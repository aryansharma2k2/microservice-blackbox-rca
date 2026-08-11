"""Propagation delay calibration script.

Measures per-edge fault propagation delays between connected services in the
Online Boutique by injecting a fault into each downstream service and observing
how long it takes for the anomaly to appear in each upstream caller.

Strategy
--------
One experiment per callee service.  When a fault is injected into service X,
every service that calls X should exhibit a propagated anomaly.  The onset-time
gap between X and each of its callers gives the propagation delay for that edge.
This lets us calibrate all ``caller→X`` edges in a single experiment.

The calibration reuses:
  - WorkloadGenerator           (infra/loadgen.py)
  - PrometheusMetricsClient     (rca_engine/metrics_client.py)
  - inject_one / _delete_resource  (fault_injection/chaos_inject.py)
  - fault_chain.pinpoint()      (rca_engine/fault_chain.py)

Usage
-----
    # Full calibration — 10 targets × 3 trials ≈ 2 hours
    python calibration/calibrate.py --trials 3

    # Quick single trial — ≈ 40 minutes
    python calibration/calibrate.py --trials 1

    # Calibrate only specific downstream services
    python calibration/calibrate.py --trials 2 --services checkoutservice,paymentservice

    # Dry run — print experiment plan without injecting anything
    python calibration/calibrate.py --dry-run
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from infra.loadgen import WorkloadGenerator
from rca_engine.metrics_client import PrometheusMetricsClient
from rca_engine.dependency import get_dependency_graph
from rca_engine import fault_chain
from fault_injection.chaos_inject import inject_one, _delete_resource
from calibration.propagation_map import PropagationMap, empty_map

FRONTEND_URL   = "http://localhost:8080"
PROMETHEUS_URL = "http://localhost:9090"
NAMESPACE      = "boutique"

BASELINE_DURATION  = 60   # seconds — enough for onset detection baseline
FAULT_DURATION     = 120  # seconds — enough for propagation to manifest
RECOVERY_WAIT      = 30   # seconds between experiments
COOLDOWN_BETWEEN   = 90   # seconds between trials of the same target


def _ts() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _callee_services(graph: dict[str, list[str]]) -> list[str]:
    """Return all services that appear as a dependency (i.e. are called by someone)."""
    callees: set[str] = set()
    for deps in graph.values():
        callees.update(deps)
    # Only include services that are actually in the graph as nodes
    return sorted(callees & graph.keys())


def _callers_of(callee: str, graph: dict[str, list[str]]) -> list[str]:
    """Return all services that directly call *callee*."""
    return [svc for svc, deps in graph.items() if callee in deps]


def _run_single_calibration(
    target: str,
    fault: str,
    rps: float,
    client: PrometheusMetricsClient,
) -> dict[str, float | None]:
    """Run one calibration experiment for *target*.

    Injects *fault* into *target*, collects metrics, runs the RCA onset
    detection pipeline, and returns a dict of
    ``{caller: propagation_delay_seconds}`` for each direct caller of *target*.
    If a caller showed no onset, its value is ``None``.
    """
    graph = get_dependency_graph()
    callers = _callers_of(target, graph)

    click.echo(f"  Injecting '{fault}' into '{target}' (callers: {callers})")

    # ------------------------------------------------------------------
    # Start load generator
    # ------------------------------------------------------------------
    total_duration = BASELINE_DURATION + FAULT_DURATION + RECOVERY_WAIT + 30
    gen = WorkloadGenerator(frontend_url=FRONTEND_URL, quiet=True)
    # WorkloadGenerator is sine-only; it no longer accepts a `pattern` argument.
    gen.run(duration_seconds=total_duration, base_rps=rps)
    time.sleep(BASELINE_DURATION)

    # ------------------------------------------------------------------
    # Inject fault
    # ------------------------------------------------------------------
    injection_time = _ts()
    kind, name = inject_one(fault, target, FAULT_DURATION)
    click.echo(f"  fault active  PID-less (Chaos Mesh CR={kind}/{name})")

    time.sleep(FAULT_DURATION)
    fault_end_time = _ts()

    # Cleanup chaos resource
    _delete_resource(kind, name)

    # ------------------------------------------------------------------
    # Collect metrics
    # ------------------------------------------------------------------
    baseline_start = injection_time - BASELINE_DURATION
    baseline_end   = injection_time - 10  # trim transition buffer
    try:
        matrix = client.fetch_metric_matrix(baseline_start, fault_end_time)
    except Exception as exc:
        click.echo(f"  WARNING: metrics fetch failed: {exc}", err=True)
        gen.stop()
        return {caller: None for caller in callers}

    # ------------------------------------------------------------------
    # Run onset detection (Layers 1-5) via fault_chain.pinpoint()
    # ------------------------------------------------------------------
    onset_times: dict[str, float] = {}
    if matrix:
        results = fault_chain.pinpoint(
            metric_matrix=matrix,
            baseline_window=(baseline_start, baseline_end),
            fault_window=(injection_time, fault_end_time),
        )
        for entry in results:
            onset_times[entry["service"]] = entry["onset_time"]

    gen.stop()

    # ------------------------------------------------------------------
    # Compute per-edge delays
    # ------------------------------------------------------------------
    target_onset = onset_times.get(target)
    delays: dict[str, float | None] = {}

    for caller in callers:
        caller_onset = onset_times.get(caller)
        if target_onset is None or caller_onset is None:
            click.echo(f"    {caller}->{target}: no onset detected (target={target_onset}, caller={caller_onset})")
            delays[caller] = None
        else:
            delay = round(caller_onset - target_onset, 3)
            click.echo(f"    {caller}->{target}: onset_diff={delay:.3f}s")
            delays[caller] = max(delay, 0.0)  # clamp negative (detection noise)

    return delays


@click.command()
@click.option("--trials",   default=3,   show_default=True, help="Number of trials per target service.")
@click.option("--fault",    default="cpu_hog", show_default=True,
              type=click.Choice(["cpu_hog", "net_delay"]),
              help="Fault type to inject during calibration.")
@click.option("--services", default=None,
              help="Comma-separated list of callee services to calibrate (default: all).")
@click.option("--rps",      default=5.0, show_default=True, help="Load generator RPS.")
@click.option("--output",   default="calibration/propagation_delays.json", show_default=True,
              help="Output path for the propagation delay map.")
@click.option("--dry-run",  is_flag=True, help="Print experiment plan without injecting.")
def calibrate(
    trials: int,
    fault: str,
    services: str | None,
    rps: float,
    output: str,
    dry_run: bool,
) -> None:
    """Calibrate per-edge propagation delays for the FChain RCA algorithm."""
    graph   = get_dependency_graph()
    targets = _callee_services(graph)

    if services:
        requested = [s.strip() for s in services.split(",")]
        unknown = set(requested) - set(targets)
        if unknown:
            click.echo(f"ERROR: unknown or non-callee services: {unknown}", err=True)
            sys.exit(1)
        targets = [t for t in targets if t in requested]

    click.echo(f"\n{'='*60}")
    click.echo(f"  Propagation Delay Calibration")
    click.echo(f"  fault   : {fault}")
    click.echo(f"  targets : {targets}")
    click.echo(f"  trials  : {trials}")
    click.echo(f"  output  : {output}")
    click.echo(f"{'='*60}\n")

    if dry_run:
        total = len(targets) * trials
        click.echo(f"DRY RUN — would run {total} experiments:")
        for target in targets:
            callers = _callers_of(target, graph)
            click.echo(f"  inject {fault} into {target:30s}  calibrates edges: {callers}")
        click.echo(f"\nEstimated time: {total * (BASELINE_DURATION + FAULT_DURATION + RECOVERY_WAIT + COOLDOWN_BETWEEN) // 60} min")
        return

    prop_map = empty_map(
        default_threshold_s=2.0,
        step_seconds=1.0,
        calibration_fault=fault,
    )
    prop_map.trials = trials
    prop_map.created_utc = datetime.now(timezone.utc).isoformat()

    client = PrometheusMetricsClient(prometheus_url=PROMETHEUS_URL)
    n_total = len(targets) * trials

    experiment_idx = 0
    for target in targets:
        callers = _callers_of(target, graph)
        if not callers:
            click.echo(f"Skipping {target} — no callers in graph")
            continue

        for trial in range(1, trials + 1):
            experiment_idx += 1
            click.echo(
                f"\n[{experiment_idx}/{n_total}] target={target}  trial={trial}/{trials}"
                f"  ({_iso(_ts())})"
            )

            delays = _run_single_calibration(target, fault, rps, client)

            for caller, delay in delays.items():
                prop_map.record_observation(caller, target, delay if delay is not None else 0.0)
                if delay is None:
                    click.echo(f"  recorded {caller}->{target}: no propagation (using 0.0 s)")

            # Save intermediate results after every trial so a crash doesn't
            # lose all data
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            prop_map.save(out_path)
            click.echo(f"  saved intermediate map → {out_path}")

            if trial < trials:
                click.echo(f"  cooldown {COOLDOWN_BETWEEN}s before next trial …")
                time.sleep(COOLDOWN_BETWEEN)

        if target != targets[-1]:
            click.echo(f"\nRecovery {RECOVERY_WAIT}s before next target …")
            time.sleep(RECOVERY_WAIT)

    # Final save with summary
    prop_map.finalize()
    prop_map.save(Path(output))

    click.echo(f"\n{'='*60}")
    click.echo(f"  Calibration complete — {experiment_idx} experiments")
    click.echo(f"  Output: {output}")
    click.echo(f"\n  Per-edge thresholds:")
    for key in sorted(prop_map.edge_keys()):
        entry = prop_map._edges[key]
        med  = entry.get("median_delay_s")
        thr  = entry.get("threshold_s")
        obs  = entry.get("observed_delays_s", [])
        click.echo(f"    {key:50s}  median={med}s  threshold={thr}s  (n={len(obs)})")
    click.echo(f"{'='*60}\n")


if __name__ == "__main__":
    calibrate()
