"""Tests for the fault library and the vLLM workload generator.

The library is ground truth for the whole evaluation, so its internal
consistency matters more than usual: a scenario labelled with a component that
does not exist, or one whose fault phase is identical to its baseline, would
silently score as a pipeline miss rather than a broken experiment.
"""

import random

import pytest

from rca_engine.domains import VLLM
from rca_engine.fault_chain import (
    VERDICT_CAPACITY,
    VERDICT_NO_ANOMALY,
    VERDICT_PATHOLOGY,
)
from workload.generator import (
    TOKENS_PER_WORD,
    VllmWorkloadGenerator,
    make_text,
    tokens_to_words,
)
from workload.scenarios import (
    CONFIG,
    INFRA,
    NOMINAL,
    SCENARIOS,
    WORKLOAD,
    Phase,
    Scenario,
    confounder_pairs,
    get_scenario,
    scenarios_for,
)


class TestLibraryIntegrity:

    def test_every_ground_truth_is_a_real_component(self):
        for scenario in SCENARIOS.values():
            if scenario.ground_truth is None:
                continue
            assert scenario.ground_truth in VLLM.component_graph, (
                f"{scenario.name} is labelled with {scenario.ground_truth!r}, "
                "which is not a component of the vLLM domain"
            )

    def test_no_scenario_is_labelled_with_an_ineligible_component(self):
        """A scenario whose ground truth can never be pinpointed would score
        zero no matter how well the pipeline works."""
        excluded = VLLM.excluded_from_root_cause()
        for scenario in SCENARIOS.values():
            assert scenario.ground_truth not in excluded, (
                f"{scenario.name} is labelled with {scenario.ground_truth!r}, "
                f"which is excluded from root-cause candidacy: {sorted(excluded)}"
            )

    def test_fault_phase_actually_differs_from_baseline(self):
        for scenario in SCENARIOS.values():
            if scenario.name == "clean":
                continue
            differs = (
                scenario.fault != scenario.baseline
                or scenario.server_args
                or scenario.infra_fault
            )
            assert differs, f"{scenario.name} injects nothing"

    def test_clean_run_changes_nothing(self):
        clean = get_scenario("clean")
        assert clean.fault == clean.baseline
        assert clean.ground_truth is None
        assert clean.expect_verdict == VERDICT_NO_ANOMALY
        assert not clean.server_args and not clean.infra_fault

    def test_expected_verdicts_are_real_verdicts(self):
        valid = {VERDICT_CAPACITY, VERDICT_PATHOLOGY, VERDICT_NO_ANOMALY}
        for scenario in SCENARIOS.values():
            assert scenario.expect_verdict in valid, scenario.name

    def test_confounds_with_references_real_scenarios(self):
        for scenario in SCENARIOS.values():
            for other in scenario.confounds_with:
                assert other in SCENARIOS, f"{scenario.name} -> unknown {other!r}"
                assert other != scenario.name

    def test_library_covers_the_four_hypotheses(self):
        """KV cache pressure, batch scheduling, block churn, preemption — the
        causes the project set out to separate."""
        labelled = {s.ground_truth for s in SCENARIOS.values()}
        assert "kv_cache_pressure" in labelled
        assert "queueing" in labelled
        assert "batch_composition" in labelled
        assert "prefix_cache_efficacy" in labelled

    def test_every_kind_is_represented(self):
        kinds = {s.kind for s in SCENARIOS.values()}
        assert kinds == {WORKLOAD, CONFIG, INFRA}

    def test_confounder_pairs_are_symmetric_and_unordered(self):
        pairs = confounder_pairs()
        assert frozenset({"host_cpu_hog", "queue_starved"}) in pairs
        for pair in pairs:
            assert len(pair) == 2

    def test_most_of_the_library_needs_no_gpu(self):
        without_gpu = scenarios_for(gpu_available=False)
        assert len(without_gpu) >= len(SCENARIOS) - 1
        assert all(not s.requires_gpu for s in without_gpu)

    def test_filtering_by_kind(self):
        assert {s.name for s in scenarios_for(kind=CONFIG)} == {
            "kv_cache_starved",
            "queue_starved",
            "chunked_prefill_off",
            "prefix_cache_off",
            "block_size_mismatch",
        }

    def test_unknown_scenario_lists_alternatives(self):
        with pytest.raises(KeyError, match="qps_ramp"):
            get_scenario("nope")


class TestScenarioValidation:

    def test_config_scenario_needs_server_args(self):
        with pytest.raises(ValueError, match="server_args"):
            Scenario(
                name="x", kind=CONFIG, ground_truth="queueing",
                expect_verdict=VERDICT_PATHOLOGY, summary="", fault=NOMINAL,
            )

    def test_infra_scenario_needs_a_fault_name(self):
        with pytest.raises(ValueError, match="infra_fault"):
            Scenario(
                name="x", kind=INFRA, ground_truth="queueing",
                expect_verdict=VERDICT_PATHOLOGY, summary="", fault=NOMINAL,
            )

    def test_unknown_kind_is_rejected(self):
        with pytest.raises(ValueError, match="unknown kind"):
            Scenario(
                name="x", kind="magic", ground_truth="queueing",
                expect_verdict=VERDICT_PATHOLOGY, summary="", fault=NOMINAL,
            )

    def test_server_args_rejected_on_non_config_kinds(self):
        with pytest.raises(ValueError, match="only apply to config"):
            Scenario(
                name="x", kind=WORKLOAD, ground_truth="queueing",
                expect_verdict=VERDICT_PATHOLOGY, summary="", fault=NOMINAL,
                server_args=("--nope",),
            )


class TestPhase:

    def test_describe_mentions_prefix_reuse(self):
        assert "8 shared prefixes" in NOMINAL.describe()
        assert "unique prefixes" in Phase(
            rps=1, prompt_tokens=(1, 2), max_tokens=(1, 2), shared_prefixes=0
        ).describe()

    def test_phases_are_comparable(self):
        assert NOMINAL == Phase(
            rps=6.0, prompt_tokens=(256, 512), max_tokens=(48, 96), shared_prefixes=8
        )


class TestTextGeneration:

    def test_word_count_scales_with_token_target(self):
        assert tokens_to_words(1300) == pytest.approx(1000, rel=0.01)
        assert tokens_to_words(0) >= 1

    def test_generated_text_length_tracks_the_target(self):
        rng = random.Random(0)
        short = make_text(100, rng)
        long = make_text(1000, rng)
        assert len(long.split()) > len(short.split()) * 5

    def test_generation_is_deterministic_for_a_seed(self):
        assert make_text(50, random.Random(7)) == make_text(50, random.Random(7))

    def test_tokens_per_word_is_a_documented_estimate(self):
        # Guards against someone silently changing the constant: scenarios are
        # defined by baseline/fault ratios, but a wild value would make the
        # declared token ranges meaningless.
        assert 1.0 <= TOKENS_PER_WORD <= 2.0


class TestPromptShaping:

    def _gen(self) -> VllmWorkloadGenerator:
        return VllmWorkloadGenerator(model="test-model", seed=42, quiet=True)

    def test_shared_prefix_pool_is_reused_across_requests(self):
        """Prefix cache hits require the same prefix to actually recur."""
        gen = self._gen()
        phase = Phase(rps=1, prompt_tokens=(600, 600), max_tokens=(8, 8),
                      shared_prefixes=2)
        rng = random.Random(0)
        prefixes = {
            gen.build_prompt(phase, rng)[0].split("\n\n")[0] for _ in range(40)
        }
        assert len(prefixes) == 2

    def test_zero_shared_prefixes_gives_every_request_a_unique_one(self):
        gen = self._gen()
        phase = Phase(rps=1, prompt_tokens=(600, 600), max_tokens=(8, 8),
                      shared_prefixes=0)
        rng = random.Random(0)
        prefixes = [
            gen.build_prompt(phase, rng)[0].split("\n\n")[0] for _ in range(20)
        ]
        assert len(set(prefixes)) == 20

    def test_prefix_diversity_does_not_change_prompt_length(self):
        """The scenario's whole point: only reuse changes, not size. If length
        moved too, `request_shape` would fire and the run would be labelled
        capacity instead of a cache pathology."""
        gen = self._gen()
        shared = Phase(rps=1, prompt_tokens=(800, 800), max_tokens=(8, 8),
                       shared_prefixes=8)
        unique = Phase(rps=1, prompt_tokens=(800, 800), max_tokens=(8, 8),
                       shared_prefixes=0)
        n_shared = [
            len(gen.build_prompt(shared, random.Random(i))[0].split())
            for i in range(20)
        ]
        n_unique = [
            len(gen.build_prompt(unique, random.Random(i))[0].split())
            for i in range(20)
        ]
        assert sum(n_shared) / len(n_shared) == pytest.approx(
            sum(n_unique) / len(n_unique), rel=0.02
        )

    def test_longer_target_produces_a_longer_prompt(self):
        gen = self._gen()
        rng = random.Random(0)
        short, _ = gen.build_prompt(
            Phase(rps=1, prompt_tokens=(600, 600), max_tokens=(8, 8)), rng
        )
        long, _ = gen.build_prompt(
            Phase(rps=1, prompt_tokens=(8192, 8192), max_tokens=(8, 8)), rng
        )
        assert len(long.split()) > len(short.split()) * 5


class TestPercentiles:

    def test_p95_of_a_known_series(self):
        values = [float(i) for i in range(100)]
        assert VllmWorkloadGenerator._percentile(values, 0.95) == 95.0

    def test_percentile_never_runs_off_the_end(self):
        assert VllmWorkloadGenerator._percentile([1.0], 0.99) == 1.0
        assert VllmWorkloadGenerator._percentile([1.0, 2.0], 1.0) == 2.0

    def test_no_data_reads_as_none_not_zero(self):
        """Zero would look like a perfectly fast server to the SLO monitor."""
        gen = VllmWorkloadGenerator(model="m", quiet=True)
        assert gen.current_p95() is None
        assert gen.current_p99() is None

    def test_summary_is_safe_with_no_traffic(self):
        gen = VllmWorkloadGenerator(model="m", quiet=True)
        summary = gen.summary()
        assert summary["sent"] == 0
        assert summary["ttft_p99_ms"] is None


class TestSloMonitorCompatibility:

    def test_exposes_the_interface_slomonitor_depends_on(self):
        """eval.SLOMonitor only ever calls current_p95(window_seconds=...), so
        it must work against either generator unchanged."""
        from infra.loadgen import WorkloadGenerator

        for cls in (VllmWorkloadGenerator, WorkloadGenerator):
            assert callable(getattr(cls, "current_p95"))
            assert callable(getattr(cls, "stop"))
