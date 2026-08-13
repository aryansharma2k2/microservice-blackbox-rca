"""Score the pipeline and every baseline over the captured traces.

    python -m eval.run_eval traces/vllm
    python -m eval.run_eval traces/vllm --json results.json

Runs entirely from committed artifacts — no server, no GPU — so every number
in the writeup can be re-derived by anyone who clones the repo, and CI can
assert that accuracy has not regressed.

The comparison is the point. A top-1 number on its own says nothing about
whether an eight-layer statistical pipeline earns its complexity; the same
number next to static threshold alerting and correlation-with-the-SLI does.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import pandas as pd

import fault_injection.ground_truth as gt
from eval.baselines import BASELINES
from eval.metrics import ScoreCard, format_card, score, _baseline_valid
from eval.replay import (
    GROUND_TRUTH_FILE,
    METRICS_FILE,
    TIMELINE_FILE,
    ReplayResult,
    find_runs,
    score_run,
)
from rca_engine.domains import get_domain
from rca_engine.metrics_client import metric_matrix_from_frame


def _result_from_ranking(
    ranked: list[str],
    ground_truth: dict,
    verdict: str = "",
) -> ReplayResult:
    """Score a plain ranked list the same way a pipeline report is scored."""
    expected = ground_truth.get("root_cause")
    return ReplayResult(
        run_id=ground_truth["run_id"],
        scenario=ground_truth.get("scenario", ground_truth["fault_type"]),
        domain=ground_truth.get("domain", "vllm"),
        expected=expected,
        expected_verdict=ground_truth.get("expect_verdict", ""),
        predicted=ranked[0] if ranked else None,
        predicted_verdict=verdict,
        rank_of_expected=(ranked.index(expected) + 1 if expected in ranked else None),
        ranked=ranked,
    )


def evaluate(root: Path, include_invalid: bool = False) -> dict[str, ScoreCard]:
    """Score the pipeline and each baseline over every valid run under *root*."""
    runs = find_runs(root)
    pipeline: list[ReplayResult] = []
    baseline_results: dict[str, list[ReplayResult]] = {k: [] for k in BASELINES}
    skipped = 0

    for run_dir in runs:
        if not include_invalid and not _baseline_valid(run_dir):
            skipped += 1
            continue
        try:
            ground_truth = gt.load(run_dir / GROUND_TRUTH_FILE)
            frame = pd.read_parquet(run_dir / METRICS_FILE)
            timeline = json.loads((run_dir / TIMELINE_FILE).read_text())
        except (FileNotFoundError, ValueError, KeyError):
            skipped += 1
            continue

        pipeline.append(score_run(run_dir))

        spec = get_domain(ground_truth.get("domain", "boutique"))
        matrix = metric_matrix_from_frame(frame)
        windows = timeline["windows"]
        step = timeline.get("step_seconds", 1.0)
        for name, fn in BASELINES.items():
            ranked = fn(
                matrix, tuple(windows["baseline"]), tuple(windows["fault"]), spec, step
            )
            baseline_results[name].append(
                _result_from_ranking(ranked, ground_truth)
            )

    cards = {"pipeline": score(pipeline, excluded_invalid=skipped)}
    for name, results in baseline_results.items():
        cards[name] = score(results, excluded_invalid=skipped)
    return cards


def format_comparison(cards: dict[str, ScoreCard]) -> str:
    """Render the head-to-head table."""
    header = (
        f"{'method':<16}{'top-1':>8}{'top-3':>8}{'MRR':>8}"
        f"{'verdict':>9}{'FPR':>8}{'fault top-1':>13}"
    )
    lines = [header, "-" * len(header)]
    for name in ["pipeline", *sorted(k for k in cards if k != "pipeline")]:
        c = cards[name]
        fpr = "-" if c.false_positive_rate is None else f"{c.false_positive_rate:.0%}"
        ftop = "-" if c.fault_top1 is None else f"{c.fault_top1:.0%}"
        verdict = f"{c.verdict_accuracy:.0%}" if name == "pipeline" else "-"
        lines.append(
            f"{name:<16}{c.top1:>8.0%}{c.top3:>8.0%}{c.mrr:>8.3f}"
            f"{verdict:>9}{fpr:>8}{ftop:>13}"
        )
    lines.append("")
    lines.append(
        "  FPR = share of clean runs where something was named. "
        "Baselines have no verdict."
    )
    return "\n".join(lines)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("traces/vllm"),
)
@click.option(
    "--json",
    "json_out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Also write the full results as JSON.",
)
@click.option("--include-invalid", is_flag=True, help="Score saturated-baseline runs too.")
@click.option(
    "--min-top1",
    type=float,
    default=None,
    help="Exit non-zero if pipeline top-1 falls below this. For CI.",
)
@click.option(
    "--min-runs",
    type=int,
    default=0,
    show_default=True,
    help="Skip the --min-top1 gate below this many scored runs. An accuracy "
    "floor asserted over a handful of runs measures noise, not regression.",
)
@click.option("--detail", is_flag=True, help="Print the per-scenario breakdown.")
def main(
    path: Path,
    json_out: Path | None,
    include_invalid: bool,
    min_top1: float | None,
    min_runs: int,
    detail: bool,
) -> None:
    """Score the pipeline and baselines over captured runs at PATH."""
    cards = evaluate(path, include_invalid=include_invalid)
    pipeline = cards["pipeline"]

    if pipeline.runs == 0:
        raise click.ClickException(
            f"No scorable runs under {path}"
            + (
                f" ({pipeline.excluded_invalid} excluded as invalid; "
                "pass --include-invalid to score them anyway)"
                if pipeline.excluded_invalid
                else ""
            )
        )

    click.echo()
    click.echo(format_comparison(cards))
    if detail:
        click.echo()
        click.echo(format_card(pipeline, "Pipeline detail"))
        if pipeline.confusion:
            click.echo("\n  confusion (expected -> predicted)")
            for expected, preds in sorted(pipeline.confusion.items()):
                got = ", ".join(f"{k} x{v}" for k, v in sorted(preds.items()))
                click.echo(f"    {expected:<24} -> {got}")

    if json_out:
        json_out.write_text(
            json.dumps({k: v.to_dict() for k, v in cards.items()}, indent=2) + "\n"
        )
        click.echo(f"\nwrote {json_out}")

    if min_top1 is None:
        return
    if pipeline.runs < min_runs:
        click.echo(
            f"\nAccuracy gate skipped: {pipeline.runs} scored run(s) is below "
            f"the --min-runs {min_runs} needed for the number to mean anything."
        )
        return
    if pipeline.top1 < min_top1:
        click.echo(
            f"\nFAIL: pipeline top-1 {pipeline.top1:.1%} over {pipeline.runs} "
            f"runs is below the {min_top1:.1%} floor.",
            err=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
