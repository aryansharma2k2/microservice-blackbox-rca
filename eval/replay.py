"""Re-diagnose a captured run from disk, with no cluster and no GPU.

A run directory holds everything needed to reproduce its diagnosis:

    traces/vllm/<run_id>/
        ground_truth.json   what was injected, and which mechanism is to blame
        metrics.parquet     every series over [baseline_start, fault_end]
        timeline.json       window boundaries and event timestamps
        rca_report.json     the diagnosis produced at capture time

Replay rebuilds the metric matrix from the Parquet and runs the pipeline again.
Two things follow from that:

* Every number in the writeup can be re-derived by anyone who clones the repo.
  The capture session needs a GPU; checking the result does not.
* The evaluation runs over committed traces, so CI can assert that top-1
  accuracy has not regressed — an RCA project whose CI checks its own
  diagnostic accuracy.

Replay goes through the same ``metric_matrix_from_frame`` as live capture. If
it did not, a replayed result would not be evidence about the live one.

Usage
-----
    python -m eval.replay traces/vllm/20260812_120000
    python -m eval.replay traces/vllm --all
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import pandas as pd

import fault_injection.ground_truth as gt
from rca_engine import fault_chain
from rca_engine.domains import get_domain
from rca_engine.fault_chain import RcaReport
from rca_engine.metrics_client import metric_matrix_from_frame

logger = logging.getLogger(__name__)

METRICS_FILE = "metrics.parquet"
GROUND_TRUTH_FILE = "ground_truth.json"
TIMELINE_FILE = "timeline.json"
REPORT_FILE = "rca_report.json"


@dataclass
class ReplayResult:
    """One run, re-diagnosed and scored against its label."""

    run_id: str
    scenario: str
    domain: str
    #: Mechanism that was actually injected; None for a clean control.
    expected: str | None
    expected_verdict: str
    #: Highest-ranked component the pipeline named, or None if it named nothing.
    predicted: str | None
    predicted_verdict: str
    #: 1-based position of the expected component, or None if absent entirely.
    rank_of_expected: int | None
    ranked: list[str]

    @property
    def top1(self) -> bool:
        """Did the pipeline name the right mechanism first?

        For a clean control the right answer is "nothing", so naming nothing
        is the hit and naming anything is a false positive.
        """
        if self.expected is None:
            return self.predicted is None
        return self.predicted == self.expected

    @property
    def top3(self) -> bool:
        if self.expected is None:
            return self.predicted is None
        return self.rank_of_expected is not None and self.rank_of_expected <= 3

    @property
    def verdict_correct(self) -> bool:
        return self.predicted_verdict == self.expected_verdict

    @property
    def reciprocal_rank(self) -> float:
        if self.expected is None:
            return 1.0 if self.predicted is None else 0.0
        return 1.0 / self.rank_of_expected if self.rank_of_expected else 0.0


def load_run(run_dir: Path) -> tuple[dict, pd.DataFrame, dict]:
    """Load a run's label, metrics, and timeline.

    Raises FileNotFoundError with an actionable message when a run is
    incomplete — a half-written run silently scoring as a miss is exactly the
    failure mode this project exists to avoid.
    """
    run_dir = Path(run_dir)
    for name in (GROUND_TRUTH_FILE, METRICS_FILE, TIMELINE_FILE):
        if not (run_dir / name).exists():
            raise FileNotFoundError(
                f"{run_dir} is missing {name} — the run did not complete and "
                "must be excluded rather than scored."
            )

    ground_truth = gt.load(run_dir / GROUND_TRUTH_FILE)
    frame = pd.read_parquet(run_dir / METRICS_FILE)
    timeline = json.loads((run_dir / TIMELINE_FILE).read_text())
    return ground_truth, frame, timeline


def diagnose_run(run_dir: Path) -> tuple[RcaReport, dict]:
    """Re-run the pipeline over a captured run. Returns (report, ground_truth)."""
    ground_truth, frame, timeline = load_run(run_dir)

    spec = get_domain(ground_truth.get("domain", "boutique"))
    matrix = metric_matrix_from_frame(frame)
    if not matrix:
        raise ValueError(f"{run_dir}: metrics.parquet contains no usable series")

    windows = timeline["windows"]
    report = fault_chain.pinpoint_report(
        metric_matrix=matrix,
        baseline_window=tuple(windows["baseline"]),
        fault_window=tuple(windows["fault"]),
        step_seconds=timeline.get("step_seconds", 1.0),
        domain=spec,
    )
    return report, ground_truth


def score_run(run_dir: Path) -> ReplayResult:
    """Diagnose a run and score it against its ground truth."""
    report, ground_truth = diagnose_run(run_dir)

    ranked = [entry["service"] for entry in report.ranked]
    expected = ground_truth.get("root_cause")
    rank = ranked.index(expected) + 1 if expected in ranked else None

    return ReplayResult(
        run_id=ground_truth["run_id"],
        scenario=ground_truth.get("scenario", ground_truth["fault_type"]),
        domain=report.domain,
        expected=expected,
        expected_verdict=ground_truth.get("expect_verdict", ""),
        predicted=ranked[0] if ranked else None,
        predicted_verdict=report.verdict,
        rank_of_expected=rank,
        ranked=ranked,
    )


def find_runs(root: Path) -> list[Path]:
    """Every complete run directory under *root*, sorted by run id."""
    root = Path(root)
    if (root / GROUND_TRUTH_FILE).exists():
        return [root]
    return sorted(p.parent for p in root.glob(f"*/{GROUND_TRUTH_FILE}"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_one(result: ReplayResult) -> None:
    mark = "PASS" if result.top1 else "MISS"
    expected = result.expected or "(nothing — clean run)"
    predicted = result.predicted or "(nothing)"
    click.echo(f"\n[{mark}] {result.run_id}  scenario={result.scenario}")
    click.echo(f"  expected  : {expected}   verdict={result.expected_verdict}")
    click.echo(f"  predicted : {predicted}   verdict={result.predicted_verdict}")
    if not result.verdict_correct:
        click.echo("              ^ verdict mismatch")
    if result.ranked:
        click.echo(f"  ranking   : {' > '.join(result.ranked)}")
    if result.expected and result.rank_of_expected:
        click.echo(f"  expected component ranked #{result.rank_of_expected}")
    elif result.expected:
        click.echo("  expected component is absent from the ranking entirely")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--all", "replay_all", is_flag=True, help="Replay every run under PATH.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of text.")
@click.option(
    "--fail-on-miss",
    is_flag=True,
    help="Exit non-zero if any run is not top-1 correct. For CI.",
)
@click.option("-v", "--verbose", is_flag=True)
def main(
    path: Path, replay_all: bool, as_json: bool, fail_on_miss: bool, verbose: bool
) -> None:
    """Re-diagnose captured run(s) at PATH from committed telemetry."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    runs = find_runs(path) if replay_all else [path]
    if not runs:
        raise click.ClickException(
            f"No run directories under {path} (looked for */{GROUND_TRUTH_FILE})."
        )

    results: list[ReplayResult] = []
    failures: list[str] = []
    for run_dir in runs:
        try:
            results.append(score_run(run_dir))
        except (FileNotFoundError, ValueError) as exc:
            failures.append(str(exc))

    if as_json:
        click.echo(
            json.dumps(
                {
                    "results": [asdict(r) for r in results],
                    "skipped": failures,
                    "summary": _summarize(results),
                },
                indent=2,
            )
        )
    else:
        for result in results:
            _print_one(result)
        for failure in failures:
            click.echo(f"\n[SKIP] {failure}")
        if len(results) > 1:
            summary = _summarize(results)
            click.echo(
                f"\n{'=' * 60}\n"
                f"  runs           : {summary['runs']}\n"
                f"  top-1 accuracy : {summary['top1']:.1%}\n"
                f"  top-3 accuracy : {summary['top3']:.1%}\n"
                f"  MRR            : {summary['mrr']:.3f}\n"
                f"  verdict correct: {summary['verdict_accuracy']:.1%}"
            )

    if fail_on_miss and (failures or any(not r.top1 for r in results)):
        sys.exit(1)


def _summarize(results: list[ReplayResult]) -> dict:
    if not results:
        return {"runs": 0, "top1": 0.0, "top3": 0.0, "mrr": 0.0, "verdict_accuracy": 0.0}
    n = len(results)
    return {
        "runs": n,
        "top1": sum(r.top1 for r in results) / n,
        "top3": sum(r.top3 for r in results) / n,
        "mrr": sum(r.reciprocal_rank for r in results) / n,
        "verdict_accuracy": sum(r.verdict_correct for r in results) / n,
    }


if __name__ == "__main__":
    main()
