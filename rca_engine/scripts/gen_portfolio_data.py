"""Generate the portfolio dashboard's data file from real artifacts.

    python -m rca_engine.scripts.gen_portfolio_data
    python -m rca_engine.scripts.gen_portfolio_data --check

Every number the site displays is derived here from the committed traces and
the domain specs — the accuracy table by re-running the evaluation, the
mechanism graph by reading `DomainSpec.component_graph`, the case-study charts
by replaying real Parquet.

This exists because the previous version of the dashboard was hand-authored
JSON. Its metric series were invented, and it presented a run whose true root
cause was absent from the top 3 as a successful diagnosis. A site that can
disagree with the pipeline it describes is worse than no site.

Both dependency graphs are read from the code that the engine itself uses, so
the picture cannot drift from the graph that produced the ranking.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

import fault_injection.ground_truth as gt
from eval.replay import GROUND_TRUTH_FILE, METRICS_FILE, TIMELINE_FILE
from eval.run_eval import evaluate
from rca_engine.dependency import get_dependency_graph
from rca_engine.domains import get_domain
from rca_engine.fault_chain import pinpoint_report
from rca_engine.metrics_client import metric_matrix_from_frame

DEFAULT_TRACES = Path("traces/vllm")
DEFAULT_OUT = Path("portfolio/data/portfolio-data.json")

#: Case studies, in the order the site tells the story: a clean win, an honest
#: miss, and the null case. Pinned by run id so the narrative text on the page
#: always describes the run being charted.
CASE_STUDIES: tuple[tuple[str, str, str], ...] = (
    (
        "20260817_000813",
        "Two confusable mechanisms, separated",
        "Every request arrives with a unique random prefix, so the prefix "
        "cache stops helping: its hit rate collapses from 97% to zero and TTFT "
        "doubles. Cache occupancy and the running-batch size both move too — "
        "this is exactly the ambiguity the project is about. The pipeline "
        "names prefix_cache_efficacy at +3s, ahead of kv_cache_pressure at "
        "+4s and batch_composition at +6s, which are downstream of it.",
    ),
    (
        "20260817_001531",
        "A miss worth arguing with",
        "Offered load climbs from 4 to 24 requests per second. The label says "
        "kv_cache_pressure; the pipeline says batch_composition, whose onset "
        "is three seconds earlier. Under a ramp the running-sequence count "
        "does rise before the blocks those sequences hold accumulate, so the "
        "ranking is defensible and the label may be the thing that is wrong. "
        "It is scored as a miss regardless — the corpus is labelled by what "
        "was injected, not by what the engine did next.",
    ),
    (
        "20260816_235030",
        "Nothing is wrong",
        "Steady 6 rps against an unmodified server. TTFT wanders, several "
        "mechanisms cross their change-point thresholds, and the pipeline "
        "still names nothing. It does this on all eight clean runs.",
    ),
)

#: Points kept per charted series. The raw traces are ~350 samples at 1s; the
#: page renders a few hundred pixels wide, so anything finer is bytes for
#: nothing.
CHART_POINTS = 120


def _point(value: float) -> float | None:
    """One chart point, or None for a gap.

    A histogram quantile over a scrape window that saw no requests is NaN, and
    that is a real and common state in these traces — not corruption. It has to
    become JSON ``null``, because bare ``NaN`` is not valid JSON and
    ``response.json()`` rejects the whole document, blanking the page.
    """
    return None if not np.isfinite(value) else round(float(value), 4)


def _downsample(series: np.ndarray, n: int = CHART_POINTS) -> list[float | None]:
    """Reduce to *n* points by block max.

    Max rather than mean: these are latency quantiles and saturation gauges
    where the spike is the signal, and averaging is exactly what would hide the
    event the chart exists to show. ``nanmax`` so one empty scrape does not
    erase the whole block around it.
    """
    series = np.asarray(series, dtype=float)
    if len(series) <= n:
        return [_point(v) for v in series]
    points: list[float | None] = []
    for block in np.array_split(series, n):
        finite = block[np.isfinite(block)]
        points.append(_point(np.max(finite)) if finite.size else None)
    return points


def _analysis_view(
    values: np.ndarray,
    series_start: float,
    windows: dict,
    step: float,
) -> np.ndarray:
    """The baseline and fault windows, spliced, with nothing in between.

    A config-restart scenario takes vLLM down and brings it back, leaving two
    minutes of dead air between the windows. The pipeline never sees that gap —
    it is handed the two windows and nothing else — so charting the raw capture
    puts a restart artifact in the middle of the picture and pushes the actual
    diagnosis into the last third. Show the engine's view instead.
    """
    def cut(window: tuple[float, float]) -> np.ndarray:
        lo = max(0, int(round((window[0] - series_start) / step)))
        hi = min(len(values), int(round((window[1] - series_start) / step)) + 1)
        return values[lo:hi]

    return np.concatenate(
        [cut(tuple(windows["baseline"])), cut(tuple(windows["fault"]))]
    )


def _series_for(
    matrix: dict,
    component: str | None,
    series_start: float,
    windows: dict,
    step: float,
) -> dict[str, list[float | None]]:
    if not component or component not in matrix:
        return {}
    return {
        name: _downsample(
            _analysis_view(np.asarray(values, dtype=float), series_start, windows, step)
        )
        for name, values in matrix[component].items()
    }


def _case_study(run_dir: Path, title: str, note: str) -> dict:
    """Replay one run and capture what the page needs to chart it."""
    ground_truth = gt.load(run_dir / GROUND_TRUTH_FILE)
    frame = pd.read_parquet(run_dir / METRICS_FILE)
    timeline = json.loads((run_dir / TIMELINE_FILE).read_text())

    spec = get_domain(ground_truth.get("domain", "vllm"))
    matrix = metric_matrix_from_frame(frame)
    windows = timeline["windows"]
    step = timeline.get("step_seconds", 1.0)

    report = pinpoint_report(
        matrix,
        tuple(windows["baseline"]),
        tuple(windows["fault"]),
        domain=spec,
        step_seconds=step,
    )
    ranked = [e for e in report.ranked if e.get("eligible")]
    expected = ground_truth.get("root_cause")
    predicted = ranked[0]["service"] if ranked else None

    # Chart the SLI plus whichever mechanisms the run is actually about: the
    # labelled cause, and what the pipeline named instead when it differs.
    charted = [spec.sli_node, expected, predicted]
    seen: list[str] = []
    for component in charted:
        if component and component not in seen:
            seen.append(component)

    fault_start = windows["fault"][0]
    series_start = frame["timestamp"].min().timestamp()

    # The charted x-axis is baseline+fault spliced, so the boundary sits at the
    # join — the share of charted samples that are baseline.
    baseline_span = windows["baseline"][1] - windows["baseline"][0]
    fault_span = windows["fault"][1] - windows["fault"][0]

    return {
        "runId": run_dir.name,
        "title": title,
        "note": note,
        "scenario": ground_truth.get("scenario"),
        "expected": expected,
        "predicted": predicted,
        "correct": expected == predicted,
        "verdict": report.verdict,
        "expectedVerdict": ground_truth.get("expect_verdict"),
        "exogenousDrivers": list(report.exogenous_drivers),
        # Where the fault window begins, as a fraction of the charted span, so
        # the page can draw the boundary without knowing about epoch seconds.
        "faultBoundary": round(baseline_span / (baseline_span + fault_span), 4),
        "baselineSeconds": round(baseline_span),
        "faultSeconds": round(fault_span),
        "ranked": [
            {
                "component": e["service"],
                "rank": e["rank"],
                "confidence": round(float(e.get("confidence", 0.0)), 3),
                "abnormalMetrics": list(e.get("abnormal_metrics", [])),
                "onsetOffsetSeconds": round(
                    float(e.get("onset_time", fault_start)) - fault_start, 1
                ),
            }
            for e in ranked
        ],
        "series": [
            {
                "component": c,
                "metrics": _series_for(matrix, c, series_start, windows, step),
            }
            for c in seen
        ],
    }


def _graph(spec_name: str) -> dict:
    spec = get_domain(spec_name)
    return {
        "nodes": [
            {
                "name": name,
                "edges": list(edges),
                "exogenous": name in spec.exogenous,
                "sli": name == spec.sli_node,
                "metrics": list(spec.component_metrics.get(name, ())),
            }
            for name, edges in spec.component_graph.items()
        ],
    }


#: A baseline window whose SLI peak exceeds this multiple of its own median is
#: not a baseline — the previous run's load had not finished draining when the
#: window opened. See `_baseline_quiescence`.
QUIESCENCE_RATIO = 3.0


def _baseline_quiescence(traces: Path) -> dict:
    """How many scored runs start from a genuinely quiet baseline.

    The capture protocol runs baseline, inject, fault, repeat. When the
    previous run's load has not drained, the baseline window opens with p99
    TTFT already two orders of magnitude above its own median. Layer 1
    calibrates mu and sigma from that window, so an enormous sigma desensitizes
    the detector and nothing in the fault window clears threshold.

    The existing validity check looks at offered-vs-completed throughput and
    backlog growth, which those runs pass — the load generator was healthy, the
    server was still working off the last incident. This measures the thing
    that check misses. It is reported, not enforced: the headline accuracy
    still counts every scored run, because excluding a quarter of the corpus is
    a claim that deserves to be made explicitly rather than applied silently.
    """
    from eval.metrics import _baseline_valid
    from eval.replay import find_runs, score_run

    spec = get_domain("vllm")
    quiet_hits = quiet_total = dirty_hits = dirty_total = 0

    for run_dir in find_runs(traces):
        if not (run_dir / METRICS_FILE).exists() or not _baseline_valid(run_dir):
            continue
        try:
            frame = pd.read_parquet(run_dir / METRICS_FILE)
            timeline = json.loads((run_dir / TIMELINE_FILE).read_text())
        except (FileNotFoundError, ValueError, KeyError):
            continue

        matrix = metric_matrix_from_frame(frame)
        sli = matrix.get(spec.sli_node) or {}
        values = next(iter(sli.values()), None)
        if values is None:
            continue

        start = frame["timestamp"].min().timestamp()
        step = timeline.get("step_seconds", 1.0)
        window = timeline["windows"]["baseline"]
        lo = max(0, int(round((window[0] - start) / step)))
        hi = int(round((window[1] - start) / step)) + 1
        segment = np.asarray(values[lo:hi], dtype=float)
        segment = segment[np.isfinite(segment)]
        if len(segment) < 5:
            continue

        median = float(np.median(segment))
        ratio = float(np.max(segment)) / max(median, 1e-6)
        hit = int(bool(score_run(run_dir).top1))
        if ratio > QUIESCENCE_RATIO:
            dirty_total += 1
            dirty_hits += hit
        else:
            quiet_total += 1
            quiet_hits += hit

    return {
        "quietRuns": quiet_total,
        "quietTop1": round(quiet_hits / quiet_total, 4) if quiet_total else None,
        "contaminatedRuns": dirty_total,
        "contaminatedTop1": round(dirty_hits / dirty_total, 4) if dirty_total else None,
        "ratioThreshold": QUIESCENCE_RATIO,
    }


def build(traces: Path) -> dict:
    """Assemble everything the dashboard renders."""
    cards = evaluate(traces)
    pipeline = cards["pipeline"]

    methods = []
    for name in ["pipeline", *sorted(k for k in cards if k != "pipeline")]:
        c = cards[name]
        methods.append(
            {
                "name": name,
                "top1": round(c.top1, 4),
                "top3": round(c.top3, 4),
                "mrr": round(c.mrr, 4),
                "falsePositiveRate": (
                    None
                    if c.false_positive_rate is None
                    else round(c.false_positive_rate, 4)
                ),
                "verdictAccuracy": (
                    round(c.verdict_accuracy, 4) if name == "pipeline" else None
                ),
            }
        )

    case_studies = []
    for run_id, title, note in CASE_STUDIES:
        run_dir = traces / run_id
        if not (run_dir / METRICS_FILE).exists():
            raise click.ClickException(
                f"case study {run_id} has no {METRICS_FILE} under {traces}"
            )
        case_studies.append(_case_study(run_dir, title, note))

    return {
        "generatedFrom": {
            "traces": str(traces),
            "runsScored": pipeline.runs,
            "runsExcluded": pipeline.excluded_invalid,
            "cleanRuns": pipeline.clean_runs,
            "faultRuns": pipeline.runs - pipeline.clean_runs,
            "scenarios": len(pipeline.per_scenario),
            "mechanisms": len(
                {
                    s["expected"]
                    for s in pipeline.per_scenario.values()
                    if s["expected"]
                }
            ),
            "note": (
                "Regenerate with: python -m rca_engine.scripts.gen_portfolio_data"
            ),
        },
        "headline": {
            "top1": round(pipeline.top1, 4),
            "top3": round(pipeline.top3, 4),
            "mrr": round(pipeline.mrr, 4),
            "falsePositiveRate": round(pipeline.false_positive_rate or 0.0, 4),
            "verdictAccuracy": round(pipeline.verdict_accuracy, 4),
        },
        "methods": methods,
        "baselineQuiescence": _baseline_quiescence(traces),
        "perScenario": [
            {
                "scenario": name,
                "runs": s["runs"],
                "top1": s["top1"],
                "mrr": s["mrr"],
                "verdictAccuracy": s["verdict_accuracy"],
                "expected": s["expected"],
                "predictions": s["predictions"],
            }
            for name, s in sorted(pipeline.per_scenario.items())
        ],
        "confusion": {
            expected: dict(sorted(preds.items()))
            for expected, preds in sorted(pipeline.confusion.items())
        },
        "caseStudies": case_studies,
        "mechanismGraph": _graph("vllm"),
        "boutiqueGraph": {
            "nodes": [
                {"name": name, "edges": list(edges)}
                for name, edges in get_dependency_graph().items()
            ]
        },
    }


def render(traces: Path) -> str:
    # allow_nan=False on purpose. Python happily writes bare NaN/Infinity,
    # which no JSON parser accepts, so the failure would surface as a blank
    # page in a browser rather than here. Raise at generation time instead.
    return json.dumps(build(traces), indent=2, allow_nan=False) + "\n"


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--traces",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_TRACES,
    show_default=True,
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_OUT,
    show_default=True,
)
@click.option(
    "--check",
    is_flag=True,
    help="Exit non-zero if the file on disk is stale, instead of rewriting it.",
)
def main(traces: Path, out: Path, check: bool) -> None:
    """Regenerate the portfolio dashboard's data from committed traces."""
    rendered = render(traces)

    if check:
        if not out.exists():
            raise click.ClickException(f"{out} does not exist; run without --check.")
        if out.read_text() != rendered:
            click.echo(
                f"{out} is stale — the traces or the pipeline changed.\n"
                "Regenerate with: python -m rca_engine.scripts.gen_portfolio_data",
                err=True,
            )
            sys.exit(1)
        click.echo(f"{out} is up to date  ✓")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered)
    click.echo(f"Wrote {out}")


if __name__ == "__main__":
    main()
