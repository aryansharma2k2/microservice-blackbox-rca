"""Run one labeled vLLM TTFT experiment end to end.

    baseline -> inject -> observe -> collect -> diagnose -> write artifacts

Every run writes a self-contained directory that can be re-diagnosed later
with no server and no GPU (see ``eval/replay.py``). That is the point: the
capture session is expensive and one-time, checking the result is neither.

    python -m eval.run_vllm_experiment --scenario kv_cache_starved
    python -m eval.run_vllm_experiment --scenario clean --repeat 3

Injection depends on the scenario kind:

``workload``
    Switch the generator's phase. No restart, so the step change is clean and
    the baseline and fault windows are directly comparable.
``config``
    Restart the server with degraded flags. Costs a warm-up and forces a
    longer settle, but constrains the mechanism directly.
``infra``
    Starve the server of CPU, either by lowering the container's quota or
    by spawning competing busy loops. See eval/server_control.py.

Unlike the Boutique runner this does not trigger on an SLO breach. Every run
uses fixed windows, so a scenario that fails to move TTFT still produces a
scored artifact rather than being silently dropped — a fault that does not
fire is a finding about the fault, and it should be visible as one.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click
import requests

import fault_injection.ground_truth as gt
from eval.replay import score_run
from eval.server_control import detect, wait_for_health
from rca_engine import fault_chain
from rca_engine.domains import VLLM
from rca_engine.metrics_client import PrometheusMetricsClient
from rca_engine.scripts.discover_metrics import (
    check_domain,
    fetch_metric_text,
    parse_metric_surface,
)
from workload.generator import VllmWorkloadGenerator
from workload.scenarios import (
    CONFIG,
    INFRA,
    PROFILES,
    Scenario,
    apply_profile,
    get_scenario,
)

ROOT = Path(__file__).parent.parent
TRACES_DIR = ROOT / "traces" / "vllm"
CHAOS_INJECT = ROOT / "fault_injection" / "chaos_inject.py"
COMPOSE_DIR = ROOT / "deploy" / "vllm"

logger = logging.getLogger(__name__)

#: Seconds to let a restarted server warm up before its baseline starts.
#: The first requests after a restart pay for CUDA graph capture and an empty
#: prefix cache, which would otherwise land inside the baseline window and
#: poison the very statistics the fault is measured against.
WARMUP_S = 45


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def preflight(server_url: str, prometheus_url: str, strict: bool = True) -> None:
    """Fail before spending minutes on a run that cannot produce a result.

    The metric-surface check is the important one. A domain metric that does
    not resolve means its component never goes abnormal, and the pipeline
    confidently blames something else — a silent, systematic bias that is very
    hard to spot afterwards in aggregate accuracy numbers.
    """
    try:
        requests.get(f"{server_url}/health", timeout=10).raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise click.ClickException(
            f"vLLM is not healthy at {server_url}: {exc}\n"
            "Start it with one of:\n"
            f"  bash {COMPOSE_DIR / 'serve_native.sh'} up        "
            "# no Docker (RunPod Pods etc.)\n"
            f"  cd {COMPOSE_DIR} && docker compose --profile gpu up -d"
        ) from exc

    try:
        requests.get(f"{prometheus_url}/-/healthy", timeout=10).raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise click.ClickException(
            f"Prometheus is not healthy at {prometheus_url}: {exc}"
        ) from exc

    surface = parse_metric_surface(fetch_metric_text(f"{server_url}/metrics"))
    missing = check_domain(VLLM, surface)
    if not missing:
        click.echo(f"  [preflight] all {len(VLLM.metrics)} vLLM metrics resolve")
        return

    affected = sorted({VLLM.metrics[m].component for m in missing})
    message = (
        f"{len(missing)} of {len(VLLM.metrics)} domain metrics are absent from "
        f"{server_url}/metrics.\n"
        f"  Components left blind: {', '.join(affected)}\n"
        "  Run: python -m rca_engine.scripts.discover_metrics check vllm "
        f"--url {server_url}/metrics"
    )
    if strict:
        raise click.ClickException(
            message + "\n  Pass --allow-missing-metrics to capture anyway."
        )
    click.echo(f"  [preflight] WARNING — {message}")


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------


_CONTROL = None


def control():
    """The server backend, detected once per process."""
    global _CONTROL
    if _CONTROL is None:
        _CONTROL = detect(os.environ.get("VLLM_BACKEND"))
        logger.info("Server backend: %s", _CONTROL.name)
    return _CONTROL


def restart_server_with(server_args: tuple[str, ...], server_url: str) -> None:
    """Restart the server with extra flags, then wait for health."""
    click.echo(f"  [inject] restarting server with: {' '.join(server_args)}")
    control().restart_with(server_args, server_url)


def restore_server(server_url: str) -> None:
    """Restart the server back to its nominal configuration."""
    click.echo("  [recover] restoring nominal server config")
    control().restore(server_url)


def start_infra_fault(fault: str, duration_s: int, server_url: str):
    """Apply an infrastructure fault to the server."""
    if fault != "cpu_hog":
        raise click.ClickException(
            f"Infra fault '{fault}' is not implemented for the "
            f"{control().name} backend."
        )
    click.echo(f"  [inject] starving the server of CPU ({control().name})")
    return control().start_cpu_hog()


def stop_infra_fault(fault: str, token) -> None:
    """Undo an infrastructure fault."""
    if fault == "cpu_hog":
        click.echo("  [recover] restoring CPU")
        control().stop_cpu_hog(token)


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


def run_experiment(
    scenario: Scenario,
    baseline_s: int,
    fault_s: int,
    settle_s: int,
    server_url: str,
    prometheus_url: str,
    out_root: Path,
    step_seconds: float = 1.0,
    seed: int = 0,
    allow_missing_metrics: bool = False,
) -> Path:
    """Execute one scenario and return the directory it wrote."""
    preflight(server_url, prometheus_url, strict=not allow_missing_metrics)

    run_id = gt.make_run_id()
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"\n=== {scenario.name}  run_id={run_id} ===")
    click.echo(f"  {scenario.summary}")
    click.echo(f"  expect: {scenario.ground_truth or 'nothing'} "
               f"({scenario.expect_verdict})")

    gt.write_scenario(run_id, scenario, fault_s, run_dir)

    generator = VllmWorkloadGenerator(base_url=server_url, seed=seed)
    infra_token: str | None = None
    events: dict[str, float] = {}

    try:
        # 1. Warm up, then baseline -------------------------------------
        total_s = baseline_s + fault_s + settle_s + WARMUP_S + 30
        generator.run(duration_seconds=total_s, phase=scenario.baseline)

        click.echo(f"  [warmup] {WARMUP_S}s")
        time.sleep(WARMUP_S)

        baseline_start = time.time()
        events["baseline_start"] = baseline_start
        click.echo(f"  [baseline] {baseline_s}s at {scenario.baseline.describe()}")
        time.sleep(baseline_s)
        baseline_end = time.time()

        # A saturated baseline means the queue was already growing without
        # bound before anything was injected. The run still gets written — the
        # artifact is useful for inspection — but it is flagged so the
        # evaluation excludes it rather than counting it as a pipeline miss.
        health = generator.health(scenario.baseline.rps, window_seconds=baseline_s)
        baseline_valid = not health["saturated"]
        if not baseline_valid:
            click.echo(
                f"  [WARN] baseline is saturated — offered {health['offered_rps']} rps, "
                f"completed {health['completed_rps']} rps, {health['in_flight']} in "
                f"flight, {health['failure_rate']:.0%} failing.\n"
                "         There is no valid control period, so this run cannot "
                "support a conclusion. It will be marked invalid.\n"
                "         Lower the load with --profile cpu, or use a bigger box."
            )
        else:
            click.echo(
                f"  [baseline ok] completed {health['completed_rps']}/"
                f"{health['offered_rps']} rps, {health['failure_rate']:.0%} failing"
            )

        # 2. Inject ------------------------------------------------------
        if scenario.kind == CONFIG:
            # A restart drops in-flight requests, so pause the generator to
            # avoid a burst of connection errors polluting the fault window.
            generator.stop()
            restart_server_with(scenario.server_args, server_url)
            generator = VllmWorkloadGenerator(base_url=server_url, seed=seed)
            generator.run(duration_seconds=fault_s + settle_s + 30, phase=scenario.fault)
            time.sleep(WARMUP_S)
        elif scenario.kind == INFRA:
            infra_token = start_infra_fault(scenario.infra_fault, fault_s, server_url)
            generator.set_phase(scenario.fault)
        else:
            generator.set_phase(scenario.fault)

        fault_start = time.time()
        events["fault_start"] = fault_start
        click.echo(f"  [fault] {fault_s}s at {scenario.fault.describe()}")
        time.sleep(fault_s)
        fault_end = time.time()
        events["fault_end"] = fault_end

        # 3. Collect -----------------------------------------------------
        click.echo("  [collect] querying Prometheus")
        client = PrometheusMetricsClient(prometheus_url, domain=VLLM)
        frame = client.fetch_metrics(baseline_start, fault_end, step=f"{step_seconds:g}s")
        if frame.empty:
            raise click.ClickException(
                "Prometheus returned no data for the run window."
            )
        frame.to_parquet(run_dir / "metrics.parquet")
        click.echo(
            f"    {len(frame):,} rows, {frame['service'].nunique()} components, "
            f"{frame['metric'].nunique()} metrics"
        )

        # 4. Timeline ----------------------------------------------------
        workload_summary = generator.summary()
        (run_dir / "timeline.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "scenario": scenario.name,
                    "kind": scenario.kind,
                    "step_seconds": step_seconds,
                    "baseline_valid": baseline_valid,
                    "baseline_health": health,
                    "windows": {
                        "baseline": [baseline_start, baseline_end],
                        "fault": [fault_start, fault_end],
                    },
                    "events": {k: _iso(v) for k, v in events.items()},
                    "workload": {
                        "baseline": scenario.baseline.describe(),
                        "fault": scenario.fault.describe(),
                        **workload_summary,
                    },
                },
                indent=2,
            )
        )
        click.echo(
            f"    client TTFT p50/p95/p99 = "
            f"{workload_summary['ttft_p50_ms']}/{workload_summary['ttft_p95_ms']}/"
            f"{workload_summary['ttft_p99_ms']} ms "
            f"({workload_summary['ok']} ok, {workload_summary['failed']} failed)"
        )

    finally:
        generator.stop()
        if infra_token is not None:
            stop_infra_fault(scenario.infra_fault, infra_token)
        if scenario.kind == CONFIG:
            restore_server(server_url)
        click.echo(f"  [settle] {settle_s}s")
        time.sleep(settle_s)

    # 5. Diagnose ------------------------------------------------------
    # Deliberately goes through the same replay path a reader would use, so
    # the report written at capture time is by construction the report anyone
    # can re-derive from the committed artifacts.
    click.echo("  [rca] diagnosing")
    result = score_run(run_dir)
    (run_dir / "rca_report.json").write_text(
        json.dumps(
            {
                "run_id": result.run_id,
                "scenario": result.scenario,
                "domain": result.domain,
                "expected": result.expected,
                "expected_verdict": result.expected_verdict,
                "predicted": result.predicted,
                "predicted_verdict": result.predicted_verdict,
                "rank_of_expected": result.rank_of_expected,
                "ranked": result.ranked,
                "top1": result.top1,
                "top3": result.top3,
                "verdict_correct": result.verdict_correct,
            },
            indent=2,
        )
    )

    mark = "PASS" if result.top1 else ("INVALID" if not baseline_valid else "MISS")
    click.echo(
        f"  [{mark}] expected={result.expected or 'nothing'} "
        f"predicted={result.predicted or 'nothing'} "
        f"verdict={result.predicted_verdict}"
        f"{'' if result.verdict_correct else ' (verdict mismatch)'}"
    )
    if result.ranked:
        click.echo(f"         ranking: {' > '.join(result.ranked)}")
    click.echo(f"  wrote {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--scenario", "scenario_name", required=True, help="Scenario to run.")
@click.option(
    "--profile",
    "profile_name",
    type=click.Choice(sorted(PROFILES)),
    default="gpu",
    show_default=True,
    help="Rescale the workload for this deployment's capacity. Scenario "
    "ratios are preserved; only the absolute level changes.",
)
@click.option("--baseline", "baseline_s", default=120, show_default=True)
@click.option("--fault", "fault_s", default=180, show_default=True)
@click.option("--settle", "settle_s", default=60, show_default=True)
@click.option("--repeat", default=1, show_default=True, help="Consecutive repeats.")
@click.option("--server-url", default="http://localhost:8000", show_default=True)
@click.option("--prometheus-url", default="http://localhost:9090", show_default=True)
@click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=TRACES_DIR,
    show_default=True,
)
@click.option("--step", default=1.0, show_default=True, help="Prometheus step (s).")
@click.option("--seed", default=0, show_default=True)
@click.option(
    "--allow-missing-metrics",
    is_flag=True,
    help="Capture even when domain metrics are absent. Biases results; avoid.",
)
@click.option("-v", "--verbose", is_flag=True)
def main(
    scenario_name: str,
    profile_name: str,
    baseline_s: int,
    fault_s: int,
    settle_s: int,
    repeat: int,
    server_url: str,
    prometheus_url: str,
    out: Path,
    step: float,
    seed: int,
    allow_missing_metrics: bool,
    verbose: bool,
) -> None:
    """Run a labeled vLLM TTFT experiment and write a replayable artifact."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    profile = PROFILES[profile_name]
    scenario = apply_profile(get_scenario(scenario_name), profile)
    if profile_name != "gpu":
        click.echo(f"Profile '{profile.name}': {profile.note}")

    for i in range(repeat):
        run_experiment(
            scenario=scenario,
            baseline_s=baseline_s,
            fault_s=fault_s,
            settle_s=settle_s,
            server_url=server_url.rstrip("/"),
            prometheus_url=prometheus_url.rstrip("/"),
            out_root=out,
            step_seconds=step,
            seed=seed + i,
            allow_missing_metrics=allow_missing_metrics,
        )


if __name__ == "__main__":
    main()
