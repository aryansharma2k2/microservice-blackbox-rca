"""Tests for scoring and the baselines.

These numbers are the project's headline claim, so the way they are computed
needs to be as trustworthy as the pipeline that produces them. In particular:
a clean run must be scored as "naming anything is wrong", and an invalid run
must be excluded rather than counted as a miss.
"""

import numpy as np
import pytest

from eval.baselines import BASELINES, correlation, threshold, topology
from eval.metrics import ScoreCard, format_card, score
from eval.replay import ReplayResult
from rca_engine.domains import VLLM
from rca_engine.fault_chain import VERDICT_CAPACITY, VERDICT_PATHOLOGY


def _result(**kw) -> ReplayResult:
    base = dict(
        run_id="r", scenario="s", domain="vllm", expected="queueing",
        expected_verdict=VERDICT_PATHOLOGY, predicted="queueing",
        predicted_verdict=VERDICT_PATHOLOGY, rank_of_expected=1,
        ranked=["queueing"],
    )
    base.update(kw)
    return ReplayResult(**base)


class TestScoring:

    def test_perfect_run_set(self):
        card = score([_result(), _result()])
        assert card.top1 == 1.0 and card.mrr == 1.0 and card.verdict_accuracy == 1.0

    def test_partial_credit_from_rank(self):
        card = score([
            _result(),
            _result(predicted="preemption", rank_of_expected=4,
                    ranked=["preemption", "a", "b", "queueing"]),
        ])
        assert card.top1 == 0.5
        assert card.top3 == 0.5          # rank 4 is outside the top 3
        assert card.mrr == pytest.approx(0.625)

    def test_clean_run_is_correct_only_when_nothing_is_named(self):
        hit = _result(expected=None, predicted=None, rank_of_expected=None, ranked=[])
        miss = _result(expected=None, predicted="queueing", rank_of_expected=None)
        card = score([hit, miss])
        assert card.clean_runs == 2
        assert card.false_positive_rate == 0.5
        assert card.top1 == 0.5

    def test_false_positive_rate_only_counts_clean_runs(self):
        card = score([
            _result(expected=None, predicted=None, rank_of_expected=None, ranked=[]),
            _result(),  # a fault run; must not affect FPR
        ])
        assert card.clean_runs == 1
        assert card.fault_runs == 1
        assert card.false_positive_rate == 0.0
        assert card.fault_top1 == 1.0

    def test_verdict_is_scored_independently_of_the_ranking(self):
        """Right mechanism, wrong verdict, is a real and distinct failure: it
        sends an operator to fix code when they should add capacity."""
        card = score([_result(predicted_verdict=VERDICT_CAPACITY)])
        assert card.top1 == 1.0
        assert card.verdict_accuracy == 0.0

    def test_confusion_matrix_records_what_was_predicted_instead(self):
        card = score([_result(predicted="preemption", rank_of_expected=2,
                              ranked=["preemption", "queueing"])])
        assert card.confusion["queueing"]["preemption"] == 1

    def test_per_scenario_breakdown(self):
        card = score([_result(scenario="a"), _result(scenario="b"),
                      _result(scenario="b")])
        assert card.per_scenario["b"]["runs"] == 2
        assert set(card.per_scenario) == {"a", "b"}

    def test_empty_set_is_safe(self):
        card = score([])
        assert card.runs == 0 and card.top1 == 0.0
        assert card.false_positive_rate is None

    def test_excluded_invalid_is_carried_through(self):
        card = score([_result()], excluded_invalid=3)
        assert card.excluded_invalid == 3
        assert "excluded as invalid" in format_card(card)

    def test_card_serializes(self):
        assert score([_result()]).to_dict()["top1"] == 1.0


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

N_BL, N_FT = 20, 60
BL = (1000.0, 1000.0 + N_BL)
FT = (BL[1], BL[1] + N_FT)


#: Quiet defaults, chosen so no alert rule fires unless a test asks it to.
#: preemption_rate especially: its rule is "> 0", so a nonzero default would
#: make every threshold test fire it.
_QUIET = {
    "preemption_rate": 0.0,
    "requests_waiting": 0.0,
    "kv_cache_usage": 0.1,
    "prefix_cache_hit_rate": 0.99,
    "cached_token_ratio": 0.99,
    "host_cpu_throttle_ratio": 0.0,
    "gpu_sm_active": 0.1,
    "prefill_time_p99": 0.1,
    "requests_running": 2.0,
}


def _matrix(levels: dict[str, float], fault_levels: dict[str, float] | None = None):
    """Build a vLLM matrix where named metrics hold given baseline/fault values."""
    fault_levels = fault_levels or {}
    matrix = {}
    for component, metrics in VLLM.component_metrics.items():
        matrix[component] = {}
        for name in metrics:
            b = levels.get(name, _QUIET.get(name, 0.1))
            f = fault_levels.get(name, b)
            matrix[component][name] = np.concatenate(
                [np.full(N_BL, b), np.full(N_FT, f)]
            )
    return matrix


class TestThresholdBaseline:

    def test_fires_on_a_breached_rule(self):
        ranked = threshold(_matrix({}, {"kv_cache_usage": 0.95}), BL, FT, VLLM)
        assert ranked and ranked[0] == "kv_cache_pressure"

    def test_respects_runbook_order_when_several_fire(self):
        ranked = threshold(
            _matrix({}, {"kv_cache_usage": 0.95, "requests_waiting": 50.0}),
            BL, FT, VLLM,
        )
        assert ranked[:2] == ["kv_cache_pressure", "queueing"]

    def test_names_nothing_when_no_rule_breaches(self):
        assert threshold(_matrix({"prefix_cache_hit_rate": 0.99}), BL, FT, VLLM) == []

    def test_handles_a_less_than_rule(self):
        ranked = threshold(
            _matrix({"prefix_cache_hit_rate": 0.99}, {"prefix_cache_hit_rate": 0.1}),
            BL, FT, VLLM,
        )
        assert "prefix_cache_efficacy" in ranked

    def test_never_names_an_ineligible_component(self):
        ranked = threshold(_matrix({}, {"kv_cache_usage": 0.95}), BL, FT, VLLM)
        assert not set(ranked) & VLLM.excluded_from_root_cause()


class TestCorrelationBaseline:

    def test_ranks_the_series_that_tracks_the_sli(self):
        rng = np.random.default_rng(0)
        signal = rng.normal(0, 1, N_BL + N_FT)
        matrix = _matrix({})
        matrix["ttft"]["ttft_p99"] = signal
        matrix["queueing"]["queue_time_p99"] = signal * 2 + 1     # perfectly correlated
        matrix["preemption"]["preemption_rate"] = rng.normal(0, 1, N_BL + N_FT)
        ranked = correlation(matrix, BL, FT, VLLM)
        assert ranked[0] == "queueing"

    def test_never_names_the_sli_itself(self):
        ranked = correlation(_matrix({}), BL, FT, VLLM)
        assert VLLM.sli_node not in ranked

    def test_constant_series_do_not_crash_it(self):
        assert correlation(_matrix({}), BL, FT, VLLM) == []


class TestTopologyBaseline:

    def test_is_constant_regardless_of_the_data(self):
        a = topology(_matrix({}), BL, FT, VLLM)
        b = topology(_matrix({}, {"kv_cache_usage": 0.99}), BL, FT, VLLM)
        assert a == b and a

    def test_only_ranks_eligible_components(self):
        assert not set(topology(_matrix({}), BL, FT, VLLM)) & VLLM.excluded_from_root_cause()

    def test_most_upstream_mechanism_comes_first(self):
        assert topology(_matrix({}), BL, FT, VLLM)[0] == "prefix_cache_efficacy"


class TestBaselineInterface:

    @pytest.mark.parametrize("name", sorted(BASELINES))
    def test_every_baseline_shares_the_pipeline_signature(self, name):
        ranked = BASELINES[name](_matrix({}), BL, FT, VLLM, 1.0)
        assert isinstance(ranked, list)
        assert all(isinstance(c, str) for c in ranked)
