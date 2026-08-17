"""Run the whole vLLM experiment matrix unattended.

Built for a rented GPU: the box costs money by the hour, so this is
resumable, budgeted, and safe to interrupt.

    python -m eval.run_vllm_batch --dry-run          # plan + time estimate
    python -m eval.run_vllm_batch --budget-hours 8   # capture

Resumable by default: it counts the traces already on disk for each scenario
and runs only the shortfall. Kill it at any point, restart it, and it picks up
where it stopped — nothing captured is ever redone.

A failed run does not stop the batch. One scenario that cannot start (a bad
server flag, a chaos injector that is not installed) should cost you that
scenario, not the remaining six hours you paid for.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import click
import requests
import yaml

from eval.run_vllm_experiment import run_experiment
from workload.scenarios import PROFILES, apply_profile, get_scenario

ROOT = Path(__file__).parent.parent
DEFAULT_MATRIX = ROOT / "experiments" / "vllm_matrix.yaml"
DEFAULT_OUT = ROOT / "traces" / "vllm"

#: Extra wall clock a config-kind scenario costs, for the two server restarts
#: it needs. Rough, and only used for the estimate.
RESTART_OVERHEAD_S = 240

_stop = False
_stop_all = False


def _handle_sigint(signum, frame) -> None:
    """Finish the current run, then stop. Killing mid-run would leave a
    half-written trace directory that scores as a miss."""
    global _stop
    _stop = True
    click.echo("\n[batch] interrupt received — finishing this run, then stopping.")


@dataclass
class Planned:
    scenario: str
    kind: str
    repeats: int
    already_have: int
    requires_gpu: bool

    @property
    def todo(self) -> int:
        return max(0, self.repeats - self.already_have)


def count_existing(out_root: Path) -> dict[str, int]:
    """How many complete traces already exist, per scenario."""
    counts: dict[str, int] = {}
    for gt_file in out_root.glob("*/ground_truth.json"):
        run_dir = gt_file.parent
        # Only count runs that finished: a directory without metrics is a
        # crashed run and must be redone, not skipped.
        if not (run_dir / "metrics.parquet").exists():
            continue
        try:
            name = json.loads(gt_file.read_text()).get("scenario")
        except json.JSONDecodeError:
            continue
        if name:
            counts[name] = counts.get(name, 0) + 1
    return counts


def build_plan(
    matrix_path: Path,
    out_root: Path,
    gpu_available: bool,
    resume: bool,
) -> tuple[list[Planned], dict]:
    matrix = yaml.safe_load(matrix_path.read_text())
    defaults = matrix.get("defaults", {})
    existing = count_existing(out_root) if resume else {}

    plan = [
        Planned(
            scenario=entry["scenario"],
            kind=entry["kind"],
            repeats=entry["repeats"],
            already_have=existing.get(entry["scenario"], 0),
            requires_gpu=entry.get("requires_gpu", False),
        )
        for entry in matrix["experiments"]
    ]
    if not gpu_available:
        plan = [p for p in plan if not p.requires_gpu]
    return plan, defaults


def estimate_seconds(plan: list[Planned], defaults: dict) -> float:
    per_run = (
        defaults.get("baseline_seconds", 120)
        + defaults.get("fault_seconds", 180)
        + defaults.get("settle_seconds", 60)
        + 45  # warmup
    )
    return sum(
        p.todo * (per_run + (RESTART_OVERHEAD_S if p.kind == "config" else 0))
        for p in plan
    )


#: Consecutive failures before the batch gives up. A dead server fails every
#: remaining run in seconds, so without this a single crash silently converts
#: the rest of a paid session into a stream of identical errors — which is
#: exactly what happened on the first real capture: one crash, 32 failures,
#: and the batch "finished" in 4.7h of an 8h budget.
MAX_CONSECUTIVE_FAILURES = 3


def server_healthy(server_url: str) -> bool:
    try:
        requests.get(f"{server_url}/health", timeout=5).raise_for_status()
        return True
    except requests.exceptions.RequestException:
        return False


def revive_server(server_url: str) -> bool:
    """Try to bring a dead server back. Returns True if it came up.

    Config-kind scenarios kill and relaunch the server; if a degraded flag
    stops it from starting, both the restart and the restore can fail and the
    process stays down. Recovering here means one bad scenario costs one
    scenario, not the remainder of the session.
    """
    click.echo("  [batch] server is down — attempting restart")
    script = ROOT / "deploy" / "vllm" / "serve_native.sh"
    try:
        subprocess.run(
            ["bash", str(script), "up"],
            check=True, capture_output=True, timeout=1800,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        click.echo(f"  [batch] restart failed: {type(exc).__name__}", err=True)
        return False
    ok = server_healthy(server_url)
    click.echo(f"  [batch] server is {'back up' if ok else 'still down'}")
    return ok


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--matrix", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=DEFAULT_MATRIX, show_default=True)
@click.option("--out", type=click.Path(file_okay=False, path_type=Path),
              default=DEFAULT_OUT, show_default=True)
@click.option("--profile", "profile_name", type=click.Choice(sorted(PROFILES)),
              default="gpu", show_default=True)
@click.option("--server-url", default="http://localhost:8000", show_default=True)
@click.option("--prometheus-url", default="http://localhost:9090", show_default=True)
@click.option("--no-gpu", is_flag=True, help="Skip scenarios that need a real GPU.")
@click.option("--no-resume", is_flag=True, help="Ignore existing traces and run everything.")
@click.option("--budget-hours", type=float, default=None,
              help="Stop cleanly once this much wall clock has elapsed.")
@click.option("--max-runs", type=int, default=None, help="Stop after this many runs.")
@click.option("--only", multiple=True, help="Restrict to these scenarios. Repeatable.")
@click.option("--dry-run", is_flag=True, help="Print the plan and the estimate, run nothing.")
def main(
    matrix: Path, out: Path, profile_name: str, server_url: str, prometheus_url: str,
    no_gpu: bool, no_resume: bool, budget_hours: float | None, max_runs: int | None,
    only: tuple[str, ...], dry_run: bool,
) -> None:
    """Capture the full labeled trace corpus."""
    global _stop_all
    profile = PROFILES[profile_name]
    plan, defaults = build_plan(matrix, out, not no_gpu, resume=not no_resume)
    if only:
        plan = [p for p in plan if p.scenario in set(only)]

    todo = [p for p in plan if p.todo > 0]
    total_runs = sum(p.todo for p in todo)
    est_s = estimate_seconds(plan, defaults)

    click.echo(f"\nMatrix : {matrix}")
    click.echo(f"Profile: {profile.name} — {profile.note}")
    click.echo(f"Output : {out}\n")
    click.echo(f"{'scenario':<24}{'kind':<10}{'have':>6}{'want':>6}{'run':>6}")
    click.echo("-" * 52)
    for p in plan:
        click.echo(
            f"{p.scenario:<24}{p.kind:<10}{p.already_have:>6}{p.repeats:>6}{p.todo:>6}"
        )
    click.echo("-" * 52)
    click.echo(f"{'':<40}{total_runs:>6} runs")
    click.echo(f"\nEstimated wall clock: {est_s/3600:.1f}h")
    if budget_hours:
        click.echo(f"Budget: {budget_hours:.1f}h — will stop cleanly at the limit.")

    if dry_run:
        click.echo("\n(dry run — nothing executed)")
        return
    if total_runs == 0:
        click.echo("\nNothing to do: every scenario already has enough traces.")
        return

    signal.signal(signal.SIGINT, _handle_sigint)
    started = time.time()
    done = failed = consecutive_failures = 0

    for planned in todo:
        scenario = apply_profile(get_scenario(planned.scenario), profile)
        for i in range(planned.todo):
            if _stop or _stop_all:
                break
            if max_runs is not None and done >= max_runs:
                click.echo(f"\n[batch] reached --max-runs {max_runs}.")
                break
            elapsed_h = (time.time() - started) / 3600
            if budget_hours is not None and elapsed_h >= budget_hours:
                click.echo(f"\n[batch] reached --budget-hours {budget_hours}.")
                break

            click.echo(
                f"\n[batch] {done + failed + 1}/{total_runs}  "
                f"{planned.scenario} ({i + 1}/{planned.todo})  "
                f"elapsed {elapsed_h:.1f}h"
            )
            if not server_healthy(server_url) and not revive_server(server_url):
                click.echo(
                    "\n[batch] server will not come back — stopping so the rest of "
                    "the session is not spent on failing runs. Fix the server, then "
                    "rerun this command; captured runs are skipped automatically.",
                    err=True,
                )
                _stop_all = True
                break

            try:
                run_experiment(
                    scenario=scenario,
                    baseline_s=defaults.get("baseline_seconds", 120),
                    fault_s=defaults.get("fault_seconds", 180),
                    settle_s=defaults.get("settle_seconds", 60),
                    server_url=server_url.rstrip("/"),
                    prometheus_url=prometheus_url.rstrip("/"),
                    out_root=out,
                    seed=i,
                )
                done += 1
                consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001 - one bad scenario must not
                # cost the rest of a paid session
                failed += 1
                consecutive_failures += 1
                click.echo(f"  [batch] run FAILED: {type(exc).__name__}: {exc}", err=True)
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    click.echo(
                        f"\n[batch] {consecutive_failures} runs failed in a row — "
                        "stopping rather than burning the rest of the session. "
                        "Captured runs are kept; rerun to resume.",
                        err=True,
                    )
                    _stop_all = True
                    break
        if _stop or _stop_all or (max_runs is not None and done >= max_runs):
            break
        if budget_hours is not None and (time.time() - started) / 3600 >= budget_hours:
            break

    hours = (time.time() - started) / 3600
    click.echo(
        f"\n[batch] finished: {done} captured, {failed} failed, {hours:.2f}h elapsed."
        f"\n[batch] traces in {out}"
        f"\n[batch] score them with:  python -m eval.run_eval {out} --detail"
    )
    if failed and not done:
        sys.exit(1)


if __name__ == "__main__":
    main()
