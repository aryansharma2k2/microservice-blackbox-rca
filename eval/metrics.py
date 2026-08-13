"""Scoring for RCA output.

This file was a one-line docstring for the life of the project: faults were
injected and `ground_truth.json` was written, but nothing ever compared a
diagnosis to the label, so there was not a single accuracy number anywhere.
Closing that loop is the point of the evaluation phase.

What is scored
--------------
**Top-1 / Top-3 / MRR** over the mechanism the pipeline named. For a clean
control the correct answer is *nothing*, so naming anything is a miss.

**Verdict accuracy** — capacity vs pathology vs no_anomaly. Separate from the
ranking on purpose: naming the right mechanism but calling a workload change
a defect sends an operator to fix code when they should add capacity. Getting
the ranking right and the verdict wrong is a distinct failure and is reported
as one.

**False positive rate** on clean runs. A diagnoser that always names something
is useless, and this is the number that catches it.

**Per-mechanism and confusion breakdowns.** Aggregate accuracy can look
healthy while one confusable pair is being guessed every time, which is
exactly the failure mode this project exists to study. The confusion matrix
over mechanisms is what makes that visible.

Invalid runs — those whose baseline was already saturated — are excluded and
counted separately. Scoring them would conflate "the pipeline was wrong" with
"the experiment was broken".
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from eval.replay import ReplayResult, find_runs, score_run


@dataclass
class ScoreCard:
    """Aggregate accuracy over a set of scored runs."""

    runs: int = 0
    excluded_invalid: int = 0
    top1: float = 0.0
    top3: float = 0.0
    mrr: float = 0.0
    verdict_accuracy: float = 0.0
    #: Share of clean runs on which something was named. The false positive rate.
    false_positive_rate: float | None = None
    clean_runs: int = 0
    #: Share of faulty runs whose true mechanism was named first.
    fault_top1: float | None = None
    fault_runs: int = 0
    per_scenario: dict[str, dict] = field(default_factory=dict)
    #: {expected mechanism: {predicted mechanism: count}}
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "runs": self.runs,
            "excluded_invalid": self.excluded_invalid,
            "top1": round(self.top1, 4),
            "top3": round(self.top3, 4),
            "mrr": round(self.mrr, 4),
            "verdict_accuracy": round(self.verdict_accuracy, 4),
            "false_positive_rate": (
                None if self.false_positive_rate is None
                else round(self.false_positive_rate, 4)
            ),
            "clean_runs": self.clean_runs,
            "fault_top1": (
                None if self.fault_top1 is None else round(self.fault_top1, 4)
            ),
            "fault_runs": self.fault_runs,
            "per_scenario": self.per_scenario,
            "confusion": self.confusion,
        }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def score(results: list[ReplayResult], excluded_invalid: int = 0) -> ScoreCard:
    """Aggregate a list of scored runs into a ScoreCard."""
    card = ScoreCard(runs=len(results), excluded_invalid=excluded_invalid)
    if not results:
        return card

    card.top1 = _mean([float(r.top1) for r in results])
    card.top3 = _mean([float(r.top3) for r in results])
    card.mrr = _mean([r.reciprocal_rank for r in results])
    card.verdict_accuracy = _mean([float(r.verdict_correct) for r in results])

    clean = [r for r in results if r.expected is None]
    faults = [r for r in results if r.expected is not None]
    card.clean_runs = len(clean)
    card.fault_runs = len(faults)
    if clean:
        card.false_positive_rate = _mean(
            [float(r.predicted is not None) for r in clean]
        )
    if faults:
        card.fault_top1 = _mean([float(r.top1) for r in faults])

    by_scenario: dict[str, list[ReplayResult]] = defaultdict(list)
    for r in results:
        by_scenario[r.scenario].append(r)
    for name, group in sorted(by_scenario.items()):
        card.per_scenario[name] = {
            "runs": len(group),
            "top1": round(_mean([float(r.top1) for r in group]), 4),
            "mrr": round(_mean([r.reciprocal_rank for r in group]), 4),
            "verdict_accuracy": round(
                _mean([float(r.verdict_correct) for r in group]), 4
            ),
            "expected": group[0].expected,
            "predictions": dict(
                Counter(r.predicted or "nothing" for r in group).most_common()
            ),
        }

    confusion: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        confusion[r.expected or "nothing"][r.predicted or "nothing"] += 1
    card.confusion = {k: dict(v) for k, v in confusion.items()}

    return card


def score_directory(root: Path, include_invalid: bool = False) -> ScoreCard:
    """Score every complete run under *root*.

    Runs whose baseline was saturated are skipped unless *include_invalid*,
    because a run with no valid control period cannot support a conclusion in
    either direction.
    """
    results: list[ReplayResult] = []
    skipped = 0
    for run_dir in find_runs(root):
        if not include_invalid and not _baseline_valid(run_dir):
            skipped += 1
            continue
        try:
            results.append(score_run(run_dir))
        except (FileNotFoundError, ValueError):
            skipped += 1
    return score(results, excluded_invalid=skipped)


def _baseline_valid(run_dir: Path) -> bool:
    """Runs captured before the validity guard existed have no flag; trust them."""
    timeline = run_dir / "timeline.json"
    if not timeline.exists():
        return False
    return json.loads(timeline.read_text()).get("baseline_valid", True)


def format_card(card: ScoreCard, title: str = "RCA accuracy") -> str:
    """Render a ScoreCard as plain text."""
    lines = [
        f"{title}",
        "=" * max(len(title), 52),
        f"  runs scored          : {card.runs}"
        + (f"  ({card.excluded_invalid} excluded as invalid)" if card.excluded_invalid else ""),
        f"  top-1 accuracy       : {card.top1:.1%}",
        f"  top-3 accuracy       : {card.top3:.1%}",
        f"  MRR                  : {card.mrr:.3f}",
        f"  verdict accuracy     : {card.verdict_accuracy:.1%}",
    ]
    if card.fault_top1 is not None:
        lines.append(
            f"  top-1 on faults      : {card.fault_top1:.1%}  ({card.fault_runs} runs)"
        )
    if card.false_positive_rate is not None:
        lines.append(
            f"  false positive rate  : {card.false_positive_rate:.1%}  "
            f"({card.clean_runs} clean runs)"
        )

    if card.per_scenario:
        lines += ["", "  per scenario", "  " + "-" * 50]
        for name, stats in card.per_scenario.items():
            preds = ", ".join(f"{k}x{v}" for k, v in stats["predictions"].items())
            lines.append(
                f"    {name:<22}{stats['runs']:>3} run(s)  "
                f"top1={stats['top1']:.0%}  verdict={stats['verdict_accuracy']:.0%}"
            )
            lines.append(f"      expected {stats['expected'] or 'nothing'} -> got {preds}")

    return "\n".join(lines)
