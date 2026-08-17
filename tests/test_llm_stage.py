"""Tests for the LLM baseline and the explainer.

None of these call the API. What they pin down is the separation that makes the
whole evaluation trustworthy: the explainer must never be in a position to
change a diagnosis, and the LLM baseline must never score on a component the
pipeline itself could not have named.
"""

import json

import numpy as np
import pytest

# The LLM stage lives behind the optional [llm] extra. Skip the whole
# module when it is absent rather than failing collection — the rest of
# the suite must pass on a checkout without it.
pytest.importorskip("pydantic", reason="install with: pip install -e '.[llm]'")

from eval.baselines import BASELINES
from eval.llm_baseline import CACHE_DIR, MODEL, Diagnosis, summarize_evidence
from rca_engine.domains import VLLM
from rca_engine.explainer import Explanation, build_evidence, explain
from rca_engine.fault_chain import (
    VERDICT_CAPACITY,
    VERDICT_NO_ANOMALY,
    VERDICT_PATHOLOGY,
    RcaReport,
)

BL = (1000.0, 1020.0)
FT = (1020.0, 1080.0)


def _matrix(fault_levels: dict[str, float] | None = None):
    fault_levels = fault_levels or {}
    return {
        component: {
            name: np.concatenate(
                [np.full(20, 0.1), np.full(60, fault_levels.get(name, 0.1))]
            )
            for name in metrics
        }
        for component, metrics in VLLM.component_metrics.items()
    }


def _report(verdict=VERDICT_PATHOLOGY, ranked=None):
    ranked = ranked if ranked is not None else [
        {"service": "kv_cache_pressure", "onset_time": 1025.0, "confidence": 0.9,
         "abnormal_metrics": ["kv_cache_usage"], "rank": 1, "eligible": True},
        {"service": "ttft", "onset_time": 1040.0, "confidence": 0.8,
         "abnormal_metrics": ["ttft_p99"], "rank": 2, "eligible": False},
    ]
    return RcaReport(
        verdict=verdict, ranked=ranked, exogenous_drivers=[], domain="vllm"
    )


class TestEvidenceSummary:

    def test_includes_every_component_and_the_graph(self):
        text = summarize_evidence(_matrix(), BL, FT, VLLM)
        assert "kv_cache_pressure" in text
        assert "->" in text  # graph edges rendered

    def test_states_which_components_are_eligible(self):
        text = summarize_evidence(_matrix(), BL, FT, VLLM)
        assert "Eligible root causes" in text
        assert "Excluded" in text
        for excluded in VLLM.excluded_from_root_cause():
            assert excluded in text

    def test_ranks_the_largest_shift_first(self):
        text = summarize_evidence(_matrix({"kv_cache_usage": 0.9}), BL, FT, VLLM)
        rows = [l for l in text.splitlines() if "/" in l and "%" in l]
        assert "kv_cache_usage" in rows[0]

    def test_constant_series_do_not_produce_bogus_shifts(self):
        text = summarize_evidence(_matrix(), BL, FT, VLLM)
        assert "nan" not in text.lower()


class TestBaselineContract:

    def test_registered_alongside_the_statistical_baselines(self):
        assert "llm" in BASELINES

    def test_degrades_to_empty_without_sdk_or_cache(self, monkeypatch, tmp_path):
        """No API key in CI, no cached response — the eval must still run."""
        monkeypatch.setattr("eval.llm_baseline.CACHE_DIR", tmp_path)
        monkeypatch.setenv("LLM_NO_CACHE", "0")
        result = BASELINES["llm"](_matrix(), BL, FT, VLLM, 1.0)
        assert isinstance(result, list)

    def test_filters_out_ineligible_components(self, monkeypatch, tmp_path):
        """A model naming the SLI or an exogenous input must not score for it."""
        monkeypatch.setattr("eval.llm_baseline.CACHE_DIR", tmp_path)
        monkeypatch.setattr(
            "eval.llm_baseline.diagnose",
            lambda prompt, use_cache=True: Diagnosis(
                ranked_components=["ttft", "arrival_load", "kv_cache_pressure",
                                   "not_a_component"],
                verdict="pathology",
                reasoning="x",
            ),
        )
        from eval.llm_baseline import llm

        assert llm(_matrix(), BL, FT, VLLM) == ["kv_cache_pressure"]

    def test_uses_a_cached_response_when_present(self, monkeypatch, tmp_path):
        monkeypatch.setattr("eval.llm_baseline.CACHE_DIR", tmp_path)
        from eval.llm_baseline import _cache_path, diagnose

        prompt = "cached prompt"
        path = _cache_path(prompt)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "model": MODEL,
            "diagnosis": {"ranked_components": ["queueing"],
                          "verdict": "pathology", "reasoning": "cached"},
        }))
        assert diagnose(prompt).ranked_components == ["queueing"]

    def test_cache_key_changes_with_the_prompt(self, monkeypatch, tmp_path):
        monkeypatch.setattr("eval.llm_baseline.CACHE_DIR", tmp_path)
        from eval.llm_baseline import _cache_path

        assert _cache_path("a") != _cache_path("b")


class TestExplainerSeparation:
    """The pipeline decides; the model narrates. These pin that down."""

    def test_evidence_states_the_verdict_as_settled(self):
        text = build_evidence(_report(), VLLM, fault_start=1020.0)
        assert text.startswith("VERDICT: pathology")
        assert "ROOT CAUSE: kv_cache_pressure" in text

    def test_ineligible_entries_are_marked_as_evidence_only(self):
        text = build_evidence(_report(), VLLM, fault_start=1020.0)
        assert "cannot be the cause" in text
        # ...and it is the SLI row that carries the marker, not the cause.
        ttft_line = next(l for l in text.splitlines() if "ttft" in l and "onset" in l)
        assert "cannot be the cause" in ttft_line

    def test_capacity_verdict_steers_away_from_a_code_fix(self):
        text = build_evidence(_report(verdict=VERDICT_CAPACITY), VLLM, 1020.0)
        assert "NOT a code fix" in text

    def test_no_anomaly_verdict_forbids_inventing_a_problem(self):
        text = build_evidence(_report(verdict=VERDICT_NO_ANOMALY, ranked=[]), VLLM, 1020.0)
        assert "do not invent" in text.lower()
        assert "ROOT CAUSE: none" in text

    def test_onsets_are_rendered_relative_to_the_fault_start(self):
        text = build_evidence(_report(), VLLM, fault_start=1020.0)
        assert "+5s" in text   # 1025.0 - 1020.0
        assert "+20s" in text  # 1040.0 - 1020.0

    def test_falls_back_to_the_evidence_bundle_without_an_sdk(self):
        result = explain(_report(), VLLM, fault_start=1020.0)
        assert isinstance(result, Explanation)
        # Either the SDK is absent (generated=False, narrative is the bundle)
        # or a call happened; both must return something usable.
        assert result.narrative
        if not result.generated:
            assert "VERDICT:" in result.narrative

    def test_explaining_never_raises(self):
        """A capture run must not die because an explanation failed."""
        for verdict in (VERDICT_PATHOLOGY, VERDICT_CAPACITY, VERDICT_NO_ANOMALY):
            explain(_report(verdict=verdict, ranked=[]), VLLM, 1020.0)

    def test_system_prompt_forbids_rediagnosis(self):
        from rca_engine.explainer import SYSTEM

        assert "never to re-diagnose" in SYSTEM
        assert "settled facts" in SYSTEM


class TestOptionalDependency:
    """`make eval` must work on a clean checkout without the [llm] extra."""

    def test_missing_pydantic_does_not_break_the_eval(self, monkeypatch):
        import builtins

        import eval.baselines as baselines

        real_import = builtins.__import__

        def no_llm_module(name, *args, **kwargs):
            if name.startswith("eval.llm_baseline") or name == "pydantic":
                raise ImportError("No module named 'pydantic'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_llm_module)
        monkeypatch.setattr(baselines, "_llm_unavailable_logged", False)
        assert baselines.BASELINES["llm"](_matrix(), BL, FT, VLLM, 1.0) == []
