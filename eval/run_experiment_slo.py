"""Experiment orchestrator: inject → wait for SLO violation → collect → RCA → score.

SLO-triggered pipeline — RCA fires as soon as the frontend SLO is violated
(plus a short observation buffer to accumulate fault-window metrics).
Injector selection:
  - cpu_hog / mem_leak / net_delay / packet_loss → chaos_inject.py (Chaos Mesh)
      Works on all services including distroless Go containers.
  - disk_hog → inject.py (kubectl exec)
      Fallback: IOChaos requires FUSE kernel support not available on kind.
      disk_hog only works on services with /bin/sh (adservice, recommendationservice,
      emailservice, paymentservice, currencyservice).

Schedule:
    1. Start load generator
    2. Baseline period (BASELINE_DURATION s)
    3. Inject fault + start SLO monitor (background)
    4. Wait until SLO violation fires OR fault duration elapses (whichever comes first)
       4a. SLO violation path: wait SLO_OBSERVATION_BUFFER s, then proceed
       4b. Duration-elapsed path: proceed immediately (same as run_experiment.py)
    5. Collect metrics (windows anchored to injection time)
    6. Run RCA
    7. Reap inject subprocess + recovery

Best fault/service combos guaranteed to trigger SLO violations:
    python eval/run_experiment_slo.py --fault net_delay  --service checkoutservice       --duration 120
    python eval/run_experiment_slo.py --fault net_delay  --service currencyservice       --duration 120
    python eval/run_experiment_slo.py --fault net_delay  --service cartservice           --duration 120
    python eval/run_experiment_slo.py --fault net_delay  --service productcatalogservice --duration 120
    python eval/run_experiment_slo.py --fault net_delay  --service emailservice          --duration 120
    python eval/run_experiment_slo.py --fault cpu_hog    --service checkoutservice       --duration 120
    python eval/run_experiment_slo.py --fault cpu_hog    --service frontend              --duration 120
"""

import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import click

# ---------------------------------------------------------------------------
# Project root on sys.path so sibling packages are importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from infra.loadgen import WorkloadGenerator
from rca_engine.metrics_client import PrometheusMetricsClient
import fault_injection.ground_truth as gt

EXPERIMENTS_DIR     = ROOT / "experiments"
CHAOS_INJECT_SCRIPT = ROOT / "fault_injection" / "chaos_inject.py"
EXEC_INJECT_SCRIPT  = ROOT / "fault_injection" / "inject.py"

# Faults handled by the old kubectl-exec injector (shell required in container).
# IOChaos needs FUSE which is unavailable on kind — fall back to shell script.
EXEC_ONLY_FAULTS = {"disk_hog"}

SLO_MULTIPLIER        = 2.2  # SLO threshold = this × baseline p95
BASELINE_DURATION     = 60    # seconds of steady-state before injection
BASELINE_END_BUFFER   = 10    # seconds trimmed from the tail of the baseline window
SLO_OBSERVATION_BUFFER = 30   # seconds to keep observing after SLO violation before running RCA
RECOVERY_WAIT         = 30    # seconds of post-fault observation before stopping loadgen
SLO_POLL_INTERVAL     = 5     # seconds between SLO polls
FRONTEND_URL          = "http://localhost:8080"
PROMETHEUS_URL        = "http://localhost:9090"
NAMESPACE             = "boutique"

# ---------------------------------------------------------------------------
# SLO monitor — background thread, signals main thread on first violation
# ---------------------------------------------------------------------------

class SLOMonitor:
    """Polls frontend p95 latency and signals the main thread on first violation.

    Sets a threading.Event the moment the SLO is first breached so the main
    orchestrator can unblock early and run RCA without waiting for the full
    fault duration.

    Every violation reported here is a measured one.  If the SLO is never
    breached the event simply never fires, the caller falls through to the
    ``duration_elapsed`` path, and ``diagnosis_latency_seconds`` is recorded
    as None.  Do not reintroduce a synthetic trigger: it silently poisons the
    diagnosis-latency numbers that the evaluation reports.
    """

    def __init__(self, gen: WorkloadGenerator, threshold_ms: float) -> None:
        self._gen = gen
        self._threshold_ms = threshold_ms
        self._violation_time: float | None = None
        self._stop = threading.Event()
        self._violation_event = threading.Event()   # set on first violation
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def violation_event(self) -> threading.Event:
        """Event that is set the moment the first SLO violation is detected."""
        return self._violation_event

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="slo-monitor"
        )
        self._thread.start()

    def stop(self) -> float | None:
        """Signal stop, wait for thread to exit, return violation timestamp (or None)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=SLO_POLL_INTERVAL + 2)
        with self._lock:
            return self._violation_time

    def _run(self) -> None:
        while not self._stop.wait(SLO_POLL_INTERVAL):
            p95 = self._gen.current_p95(window_seconds=10)
            if p95 is None:
                continue
            ms = p95 * 1000
            with self._lock:
                if self._violation_time is None and ms > self._threshold_ms:
                    self._violation_time = time.time()
                    click.echo(
                        f"  [slo] VIOLATION detected — p95={ms:.0f}ms"
                        f" (threshold={self._threshold_ms:.0f}ms)"
                    )
                    self._violation_event.set()     # unblock main thread


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _format_rca_output(logs: list, ranked: list) -> list[str]:
    """Clean RCA output: service-wise, RCA-relative timing (starts at 0)."""

    LAYER_LABELS = {
        "START_PINPOINT": "RCA Pipeline Start",
        "LAYER1_CUSUM": "Layer 1 (CUSUM Detection)",
        "LAYER3_FFT_FILTER": "Layer 3 (FFT Predictability Filter)",
        "LAYER4_ROLLBACK": "Layer 4 (Onset Rollback)",
        "FINAL_RANKING": "Final Ranking",
    }

    # --------------------------------------------------
    # Get FIRST occurrence of each layer
    # --------------------------------------------------
    layer_events = {}
    for entry in logs:
        stage = entry.get("stage")
        if stage in LAYER_LABELS:
            label = LAYER_LABELS[stage]
            ts = entry.get("timestamp", 0)
            if label not in layer_events:
                layer_events[label] = ts

    # --------------------------------------------------
    # RCA start time = earliest layer timestamp
    # --------------------------------------------------
    if not layer_events:
        return ["No RCA logs found"]

    rca_start = min(layer_events.values())

    txt_lines = []
    txt_lines.insert(0, "RCA Service Timeline")

    # --------------------------------------------------
    # SERVICE OUTPUT
    # --------------------------------------------------
    for svc in ranked:
        name = svc.get("service")
        onset = svc.get("onset_time")
        abnormal_metrics = svc.get("abnormal_metrics") or []

        # Format onset
        if isinstance(onset, (int, float)):
            onset_str = datetime.fromtimestamp(onset, timezone.utc).isoformat()
        else:
            onset_str = str(onset)

        txt_lines.append(f"Service: {name}")
        txt_lines.append(f"  Onset: {onset_str}")

        if abnormal_metrics:
            txt_lines.append("\n  Metrics:")
            txt_lines.append(f"    {', '.join(abnormal_metrics)}")

        txt_lines.append("\n  RCA Pipeline:")

        # --------------------------------------------------
        # LAYER TIMING (relative to RCA start)
        # --------------------------------------------------
        ordered_layers = [
            "RCA Pipeline Start",
            "Layer 1 (CUSUM Detection)",
            "Layer 3 (FFT Predictability Filter)",
            "Layer 4 (Onset Rollback)",
            "Final Ranking",
        ]

        for layer in ordered_layers:
            if layer in layer_events:
                delta = layer_events[layer] - rca_start
                txt_lines.append(f"    - {layer:<40} → +{delta:.3f}s")

        txt_lines.append("")

    return txt_lines

# ---------------------------------------------------------------------------
# RCA
# ---------------------------------------------------------------------------

def run_rca(
    metric_matrix: dict,
    baseline_window: tuple[float, float],
    fault_window: tuple[float, float],
    run_dir: Path,
    propagation_map_path: str | None = None,
) -> dict:
    """Call the RCA engine and return a results dict."""
    try:
        from rca_engine import fault_chain
        
        # Initialize timing collection
        start_time = time.time()
        logs: list[dict] = []
        
        ranked = fault_chain.pinpoint(
            metric_matrix=metric_matrix,
            baseline_window=baseline_window,
            fault_window=fault_window,
            propagation_map_path=propagation_map_path,
            start_time=start_time,
            logs=logs,
        )
        
        # Prepare output with timing information
        total_time = time.time() - start_time
        result = {
            "ranked_services": ranked,
            "total_time_seconds": total_time,
            "timing_logs": logs,
        }
        
        # Save timing JSON to file
        output_file = run_dir / "rca_timing.json"
        output_file.write_text(json.dumps(result, indent=2))
        click.echo(f"  [rca] timing data saved to {output_file}")
        
        # Generate clean service-centric timeline
        txt_lines = _format_rca_output(logs, ranked)
        txt_lines.insert(0, f"Total RCA time: {total_time:.3f} seconds")
        txt_lines.insert(0, "")
        txt_lines.insert(0, f"Saved: {datetime.now(timezone.utc).isoformat()}")
        txt_lines.insert(0, "RCA Service Timeline (Onset-based)")

        # ---------------------------------------------------
        # Save output
        # ---------------------------------------------------
        output_txt = run_dir / "output.txt"
        output_txt.write_text("\n".join(txt_lines))
        click.echo(f"  [rca] text summary saved to {output_txt}")

        return {"ranked_services": ranked}
    except NotImplementedError:
        click.echo("  [rca] fault_chain.pinpoint not yet implemented — saving placeholder")
        return {"status": "rca_not_implemented"}
    except Exception as exc:
        click.echo(f"  [rca] error: {exc}", err=True)
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def _ts() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

@click.command()
@click.option("--fault",      required=True,
              type=click.Choice(["cpu_hog", "mem_leak", "net_delay", "disk_hog", "packet_loss"]))
@click.option("--service",    required=True, help="Primary target service name.")
@click.option("--duration",   default=120, show_default=True,
              help="Fault duration in seconds (also the max wait for an SLO violation).")
@click.option("--run-id",     default=None,
              help="Run ID (timestamp-based ID generated if omitted).")
@click.option("--rps",        default=5.0, show_default=True,
              help="Load generator base RPS.")
@click.option("--concurrent", default=None,
              help="Comma-separated additional services to fault simultaneously.")
@click.option("--propagation-map", default=None,
              help="Path to calibration/propagation_delays.json for edge-aware RCA.")
def run(
    fault: str,
    service: str,
    duration: int,
    run_id: str | None,
    rps: float,
    concurrent: str | None,
    propagation_map: str | None,
) -> None:
    """Run a fault injection experiment triggered by SLO violation."""
    if run_id is None:
        run_id = gt.make_run_id()

    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"\n{'='*60}")
    click.echo(f"  run_id  : {run_id}")
    click.echo(f"  fault   : {fault}  into  {service}")
    click.echo(f"  duration: {duration}s  (max wait for SLO violation)")
    click.echo(f"{'='*60}\n")

    timeline: dict = {
        "run_id":           run_id,
        "fault":            fault,
        "service":          service,
        "duration_seconds": duration,
        "trigger_mode":     "slo_violation",
        "events":           {},
        "windows":          {},
        "slo":              {},
    }
    client = PrometheusMetricsClient(prometheus_url=PROMETHEUS_URL)

    # ------------------------------------------------------------------
    # 1. Start load generator
    # ------------------------------------------------------------------
    click.echo("[1/7] Starting load generator …")
    gen = WorkloadGenerator(frontend_url=FRONTEND_URL, quiet=True)
    total_gen_duration = BASELINE_DURATION + duration + RECOVERY_WAIT + 60
    gen.run(duration_seconds=total_gen_duration, base_rps=rps)
    experiment_start = _ts()
    timeline["events"]["experiment_start"] = experiment_start

    # ------------------------------------------------------------------
    # 2. Baseline period
    # ------------------------------------------------------------------
    click.echo(f"[2/7] Baseline period ({BASELINE_DURATION}s) …")
    baseline_start = _ts()
    time.sleep(BASELINE_DURATION)
    baseline_end = _ts()
    timeline["events"]["baseline_end"] = baseline_end

    # Dynamic SLO threshold: SLO_MULTIPLIER × measured baseline p95
    baseline_p95 = gen.current_p95(window_seconds=BASELINE_DURATION)
    if baseline_p95 is None:
        raise RuntimeError("Baseline p95 unavailable — loadgen produced no data during baseline window")
    dynamic_threshold_ms = baseline_p95 * 1000 * SLO_MULTIPLIER
    click.echo(
        f"  baseline p95={baseline_p95*1000:.0f}ms, SLO threshold={dynamic_threshold_ms:.0f}ms"
        f" ({SLO_MULTIPLIER}×)"
    )
    timeline["baseline_p95_ms"]  = round(baseline_p95 * 1000, 1) if baseline_p95 else None
    timeline["slo_threshold_ms"] = dynamic_threshold_ms

    # ------------------------------------------------------------------
    # 3. Inject fault + start SLO monitor in background
    # ------------------------------------------------------------------
    use_exec = fault in EXEC_ONLY_FAULTS
    inject_script = EXEC_INJECT_SCRIPT if use_exec else CHAOS_INJECT_SCRIPT
    injector_label = "exec" if use_exec else "chaos"

    click.echo(f"[3/7] Injecting fault '{fault}' into '{service}' (injector={injector_label}) …")
    inject_cmd = [
        sys.executable, str(inject_script),
        "--fault",     fault,
        "--service",   service,
        "--duration",  str(duration),
        "--run-id",    run_id,
        "--namespace", NAMESPACE,
    ]
    if concurrent:
        inject_cmd += ["--concurrent", concurrent]

    inject_proc = subprocess.Popen(
        inject_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    injection_time = _ts()
    timeline["events"]["injection_start"] = injection_time
    timeline["injector"] = injector_label
    click.echo(f"  inject PID={inject_proc.pid}")

    slo_monitor = SLOMonitor(gen, threshold_ms=dynamic_threshold_ms)
    slo_monitor.start()

    # ------------------------------------------------------------------
    # 4. Wait until SLO violation fires OR full duration elapses
    #
    #    violation_event.wait(timeout=duration) blocks until either:
    #      - The SLO monitor sets the event (violation detected) → returns True
    #      - timeout seconds pass with no violation             → returns False
    #
    #    SLO-triggered path: wait SLO_OBSERVATION_BUFFER more seconds so
    #    the fault window has enough metric samples for the RCA engine.
    #
    #    Duration-elapsed path: proceed immediately, identical to
    #    run_experiment.py — used for faults that degrade system metrics
    #    but don't surface as frontend latency spikes (e.g. mem_leak,
    #    recommendationservice faults).
    # ------------------------------------------------------------------
    click.echo(
        f"[4/7] Waiting for SLO violation (max {duration}s) …"
        f"  observation buffer after violation: {SLO_OBSERVATION_BUFFER}s"
    )
    slo_fired = slo_monitor.violation_event.wait(timeout=duration)

    if slo_fired:
        click.echo(
            f"  [trigger] SLO violation fired — collecting {SLO_OBSERVATION_BUFFER}s "
            f"of fault-window metrics before RCA …"
        )
        time.sleep(SLO_OBSERVATION_BUFFER)
        triggered_by = "slo_violation"
    else:
        click.echo(
            f"  [trigger] duration elapsed — no SLO violation in {duration}s, "
            f"running RCA anyway (fault may not be on critical latency path)"
        )
        triggered_by = "duration_elapsed"

    fault_end_time = _ts()
    timeline["events"]["fault_end"]  = fault_end_time
    timeline["triggered_by"]         = triggered_by

    # Harvest SLO result
    violation_time = slo_monitor.stop()
    timeline["slo"]["violation_time"] = violation_time
    if violation_time is not None:
        diag_latency = round(violation_time - injection_time, 2)
        timeline["slo"]["diagnosis_latency_seconds"] = diag_latency
    else:
        timeline["slo"]["diagnosis_latency_seconds"] = None

    # ------------------------------------------------------------------
    # 5. Collect metrics — windows anchored to injection time
    # ------------------------------------------------------------------
    baseline_window_start = injection_time - BASELINE_DURATION
    baseline_window_end   = injection_time - BASELINE_END_BUFFER
    fault_window_start    = injection_time
    fault_window_end      = fault_end_time

    timeline["windows"] = {
        "baseline_start": baseline_window_start,
        "baseline_end":   baseline_window_end,
        "fault_start":    fault_window_start,
        "fault_end":      fault_window_end,
    }

    click.echo(
        f"[5/7] Collecting metrics …\n"
        f"  baseline [{_iso(baseline_window_start)} to {_iso(baseline_window_end)}]\n"
        f"  fault    [{_iso(fault_window_start)} to {_iso(fault_window_end)}]"
    )
    try:
        df     = client.fetch_metrics(baseline_window_start, fault_window_end)
        matrix = client.fetch_metric_matrix(baseline_window_start, fault_window_end)
        if not df.empty:
            df.to_parquet(run_dir / "metrics.parquet", index=False)
            click.echo(
                f"  saved metrics.parquet  ({len(df):,} rows, {df['service'].nunique()} services)"
            )
        else:
            click.echo("  WARNING: no metrics returned — metrics.parquet not saved")
            matrix = {}
    except Exception as exc:
        click.echo(f"  WARNING: metrics fetch failed: {exc}", err=True)
        matrix = {}

    # ------------------------------------------------------------------
    # 6. Run RCA
    # ------------------------------------------------------------------
    click.echo("[6/7] Running RCA …")
    rca_start = _ts()
    rca_results = run_rca(
        metric_matrix=matrix,
        baseline_window=(baseline_window_start, baseline_window_end),
        fault_window=(fault_window_start, fault_window_end),
        run_dir=run_dir,
        propagation_map_path=propagation_map,
    )
    rca_end = _ts()
    timeline["events"]["rca_start"] = rca_start
    timeline["events"]["rca_end"]   = rca_end
    _save_json(run_dir / "rca_results.json", rca_results)
    click.echo(f"  rca done in {rca_end - rca_start:.1f}s")

    # ------------------------------------------------------------------
    # 7. Reap inject subprocess + recovery period
    #
    #    The inject subprocess runs for the full --duration it was given.
    #    If RCA triggered early (slo_violation path), the fault may still
    #    be active here — communicate() will block until it exits naturally.
    # ------------------------------------------------------------------
    click.echo(f"[7/7] Collecting inject output + recovery ({RECOVERY_WAIT}s) …")
    try:
        stdout, stderr = inject_proc.communicate(timeout=duration + 60)
        if stdout:
            click.echo(stdout.strip())
        if inject_proc.returncode != 0 and stderr:
            click.echo(f"  inject stderr: {stderr.strip()}", err=True)
    except subprocess.TimeoutExpired:
        click.echo("  inject script still running after extended wait — killing", err=True)
        inject_proc.kill()
        inject_proc.communicate()
    timeline["events"]["fault_stopped"] = _ts()

    time.sleep(RECOVERY_WAIT)
    gen.stop()
    experiment_end = _ts()
    timeline["events"]["recovery_end"]   = experiment_end
    timeline["events"]["experiment_end"] = experiment_end

    _save_json(run_dir / "timeline.json", timeline)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    click.echo(f"\n{'='*60}")
    click.echo(f"  Experiment complete  —  artifacts in {run_dir}")
    click.echo(f"  Total duration : {experiment_end - experiment_start:.0f}s")
    click.echo(f"  RCA trigger    : {triggered_by}")
    diag = timeline["slo"].get("diagnosis_latency_seconds")
    click.echo(
        f"  SLO violation  : t+{diag}s after injection"
        if diag is not None else
        f"  SLO violation  : not detected (p95 stayed below {dynamic_threshold_ms:.0f}ms)"
    )
    if slo_fired:
        saved_time = round(duration - diag - SLO_OBSERVATION_BUFFER, 1) if diag else None
        if saved_time and saved_time > 0:
            click.echo(f"  Time saved     : ~{saved_time}s")
    click.echo("  Files:")
    for f in sorted(run_dir.iterdir()):
        click.echo(f"    {f.name}")
    click.echo(f"{'='*60}\n")


if __name__ == "__main__":
    run()
