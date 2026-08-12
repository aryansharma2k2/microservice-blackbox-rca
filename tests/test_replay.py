"""Tests for offline replay of captured runs.

Replay is what makes the project's claims checkable: the capture session needs
a GPU, but re-deriving every number must not. These tests build run
directories in exactly the format the live runner writes, then score them, so
the whole capture -> replay -> score loop is exercised without a server.
"""

import json

import numpy as np
import pandas as pd
import pytest

import fault_injection.ground_truth as gt
from eval.replay import (
    ReplayResult,
    _summarize,
    diagnose_run,
    find_runs,
    load_run,
    score_run,
)
from rca_engine.domains import VLLM
from rca_engine.fault_chain import VERDICT_CAPACITY, VERDICT_PATHOLOGY
from workload.scenarios import get_scenario

N_BASELINE = 20
N_FAULT = 80
STEP = 1.0


def _series(offset: int | None, n: int = N_BASELINE + N_FAULT) -> np.ndarray:
    """Flat, or a step change `offset` samples into the fault window."""
    values = np.ones(n) * 0.1
    if offset is not None:
        values[N_BASELINE + offset:] = 0.9
    return values


def make_run(
    run_dir,
    scenario_name: str,
    onsets: dict[str, int],
    run_id: str = "20260812_120000",
):
    """Write a synthetic run directory in the live runner's exact format.

    Components named in *onsets* step at the given offset; every other
    component of the vLLM domain stays flat.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    bl_start = 1_000_000.0
    bl_end = bl_start + N_BASELINE * STEP
    ft_start = bl_end
    ft_end = ft_start + N_FAULT * STEP

    rows = []
    for component, metrics in VLLM.component_metrics.items():
        values = _series(onsets.get(component))
        for metric in metrics:
            for i, value in enumerate(values):
                rows.append(
                    {
                        "timestamp": bl_start + i * STEP,
                        "pod": "",
                        "service": component,
                        "metric": metric,
                        "value": float(value),
                    }
                )

    frame = pd.DataFrame(rows)
    # Match the live path, which converts to datetime64 before writing.
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    frame.sort_values(["service", "metric", "timestamp"]).reset_index(
        drop=True
    ).to_parquet(run_dir / "metrics.parquet")

    gt.write_scenario(run_id, get_scenario(scenario_name), int(N_FAULT * STEP), run_dir)

    (run_dir / "timeline.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "step_seconds": STEP,
                "windows": {
                    "baseline": [bl_start, bl_end],
                    "fault": [ft_start, ft_end],
                },
            },
            indent=2,
        )
    )
    return run_dir


class TestLoadRun:

    def test_loads_all_three_artifacts(self, tmp_path):
        run = make_run(tmp_path / "r", "kv_cache_starved", {"kv_cache_pressure": 2})
        ground_truth, frame, timeline = load_run(run)
        assert ground_truth["scenario"] == "kv_cache_starved"
        assert not frame.empty
        assert "windows" in timeline

    @pytest.mark.parametrize(
        "missing", ["ground_truth.json", "metrics.parquet", "timeline.json"]
    )
    def test_incomplete_run_raises_rather_than_scoring_as_a_miss(self, tmp_path, missing):
        """A half-written run silently counting as a miss would corrupt the
        accuracy numbers in exactly the way this project exists to avoid."""
        run = make_run(tmp_path / "r", "clean", {})
        (run / missing).unlink()
        with pytest.raises(FileNotFoundError, match=missing):
            load_run(run)


class TestScoring:

    def test_correct_diagnosis_scores_top1(self, tmp_path):
        run = make_run(
            tmp_path / "r",
            "kv_cache_starved",
            {"kv_cache_pressure": 2, "preemption": 20, "queueing": 30, "ttft": 40},
        )
        result = score_run(run)
        assert result.expected == "kv_cache_pressure"
        assert result.predicted == "kv_cache_pressure"
        assert result.top1 and result.top3
        assert result.rank_of_expected == 1
        assert result.reciprocal_rank == 1.0

    def test_verdict_is_scored_separately_from_the_ranking(self, tmp_path):
        """A load spike should read as capacity even though a mechanism is
        still surfaced. Getting the ranking right but the verdict wrong is a
        different failure and must be visible as one."""
        run = make_run(
            tmp_path / "r",
            "qps_ramp",
            {"arrival_load": 2, "kv_cache_pressure": 20, "queueing": 30, "ttft": 40},
        )
        result = score_run(run)
        assert result.expected_verdict == VERDICT_CAPACITY
        assert result.predicted_verdict == VERDICT_CAPACITY
        assert result.verdict_correct

    def test_wrong_diagnosis_records_the_rank_it_did_get(self, tmp_path):
        """Partial credit matters: a run where the truth ranked #2 is a very
        different result from one where it never appeared."""
        run = make_run(
            tmp_path / "r",
            "host_cpu_hog",
            {"queueing": 2, "host_saturation": 20, "ttft": 40},
        )
        result = score_run(run)
        assert result.expected == "host_saturation"
        assert result.predicted == "queueing"
        assert not result.top1
        assert result.rank_of_expected == 2
        assert result.top3
        assert result.reciprocal_rank == 0.5

    def test_clean_run_is_correct_only_when_nothing_is_named(self, tmp_path):
        run = make_run(tmp_path / "r", "clean", {})
        result = score_run(run)
        assert result.expected is None
        assert result.predicted is None
        assert result.top1

    def test_clean_run_with_a_diagnosis_is_a_false_positive(self, tmp_path):
        """A diagnoser that always names something is useless, and nothing in
        the repo has ever measured this."""
        run = make_run(tmp_path / "r", "clean", {"queueing": 2, "ttft": 20})
        result = score_run(run)
        assert result.expected is None
        assert result.predicted is not None
        assert not result.top1

    def test_confounded_scenario_is_separated(self, tmp_path):
        """host_cpu_hog and kv_cache_starved both show a growing queue and a
        TTFT spike. The KV cache staying flat is what distinguishes them."""
        cache = make_run(
            tmp_path / "cache",
            "kv_cache_starved",
            {"kv_cache_pressure": 2, "preemption": 20, "queueing": 30, "ttft": 40},
        )
        host = make_run(
            tmp_path / "host",
            "host_cpu_hog",
            {"host_saturation": 2, "queueing": 20, "ttft": 30},
        )
        assert score_run(cache).predicted == "kv_cache_pressure"
        assert score_run(host).predicted == "host_saturation"


class TestReplayDeterminism:

    def test_replaying_twice_gives_the_same_answer(self, tmp_path):
        """Replay must be a pure function of the committed artifacts."""
        run = make_run(
            tmp_path / "r", "prefix_cache_off",
            {"prefix_cache_efficacy": 2, "prefill_cost": 20, "ttft": 30},
        )
        first, _ = diagnose_run(run)
        second, _ = diagnose_run(run)
        assert first.ranked == second.ranked
        assert first.verdict == second.verdict

    def test_replay_uses_the_domain_recorded_in_ground_truth(self, tmp_path):
        run = make_run(tmp_path / "r", "queue_starved", {"queueing": 2, "ttft": 20})
        report, ground_truth = diagnose_run(run)
        assert ground_truth["domain"] == "vllm"
        assert report.domain == "vllm"


class TestFindRuns:

    def test_finds_every_run_under_a_root(self, tmp_path):
        for i in range(3):
            make_run(tmp_path / f"run{i}", "clean", {}, run_id=f"2026081{i}_120000")
        assert len(find_runs(tmp_path)) == 3

    def test_a_run_directory_itself_is_returned(self, tmp_path):
        run = make_run(tmp_path / "solo", "clean", {})
        assert find_runs(run) == [run]

    def test_empty_root_finds_nothing(self, tmp_path):
        assert find_runs(tmp_path) == []


class TestSummary:

    def _result(self, **kw) -> ReplayResult:
        base = dict(
            run_id="r", scenario="s", domain="vllm", expected="queueing",
            expected_verdict=VERDICT_PATHOLOGY, predicted="queueing",
            predicted_verdict=VERDICT_PATHOLOGY, rank_of_expected=1,
            ranked=["queueing"],
        )
        base.update(kw)
        return ReplayResult(**base)

    def test_aggregates_the_headline_metrics(self):
        results = [
            self._result(),
            self._result(predicted="preemption", rank_of_expected=2,
                         ranked=["preemption", "queueing"]),
        ]
        summary = _summarize(results)
        assert summary["runs"] == 2
        assert summary["top1"] == 0.5
        assert summary["top3"] == 1.0
        assert summary["mrr"] == pytest.approx(0.75)

    def test_missing_component_contributes_zero_not_a_crash(self):
        summary = _summarize(
            [self._result(predicted="preemption", rank_of_expected=None,
                          ranked=["preemption"])]
        )
        assert summary["top1"] == 0.0
        assert summary["mrr"] == 0.0

    def test_empty_input_is_safe(self):
        assert _summarize([])["runs"] == 0
