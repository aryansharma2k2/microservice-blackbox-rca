"""Tests for the domain adapter layer.

The point of this layer is that Layers 1-8 localize a fault among *components*
in a directed graph without caring what those components are.  These tests
pin that down: the same engine, given a different DomainSpec, uses that
domain's graph, thresholds, and metric counts.
"""

import numpy as np
import pytest

from rca_engine.domains import (
    VLLM,
    BOUTIQUE,
    DEFAULT_DOMAIN,
    DomainSpec,
    MetricQuery,
    get_domain,
    histogram_quantile,
    register_domain,
)
from rca_engine import fault_chain
from rca_engine.fault_chain import STEP_SECONDS, pinpoint, pinpoint_report


# -----------------------------------------------------------------------
# DomainSpec basics
# -----------------------------------------------------------------------

class TestDomainSpec:

    def test_boutique_is_the_default_and_is_valid(self):
        assert DEFAULT_DOMAIN is BOUTIQUE
        assert BOUTIQUE.validate() == []
        assert get_domain("boutique") is BOUTIQUE

    def test_boutique_declares_the_seven_fchain_metrics(self):
        assert len(BOUTIQUE.metrics) == 7
        assert "cpu_rate" in BOUTIQUE.metrics
        assert "mem_wss" in BOUTIQUE.metrics

    def test_unknown_domain_lists_alternatives(self):
        with pytest.raises(KeyError, match="boutique"):
            get_domain("nope")

    def test_graph_returns_a_mutable_copy(self):
        g1 = BOUTIQUE.graph()
        g1["frontend"].append("bogus")
        assert "bogus" not in BOUTIQUE.graph()["frontend"]

    def test_metric_count_falls_back_to_domain_wide(self):
        # Boutique declares no per-component metrics: every service carries
        # all 7, matching the original TOTAL_METRICS denominator.
        assert BOUTIQUE.component_metrics == {}
        assert BOUTIQUE.metric_count("cartservice") == 7

    def test_metric_count_prefers_per_component(self):
        spec = _toy_domain()
        assert spec.metric_count("root") == 2
        assert spec.metric_count("unknown-component") == len(spec.metrics)

    def test_resolve_component_from_labels(self):
        assert (
            BOUTIQUE.resolve_component("cpu_rate", {"pod": "cartservice-7d9b4f6c8-xkz9p"})
            == "cartservice"
        )

    def test_resolve_component_drops_unattributable_series(self):
        # cAdvisor node-level rows carry no pod label.
        assert BOUTIQUE.resolve_component("cpu_rate", {}) is None

    def test_resolve_component_prefers_a_pinned_component(self):
        spec = _toy_domain()
        # signal_a is pinned to "root" regardless of what labels say.
        assert spec.resolve_component("signal_a", {"pod": "irrelevant"}) == "root"


class TestDomainSpecValidation:

    def _spec(self, **overrides) -> DomainSpec:
        base = dict(
            name="t",
            metrics={"m": MetricQuery("up", component="a")},
            component_graph={"a": []},
        )
        base.update(overrides)
        return DomainSpec(**base)

    def test_healthy_spec_has_no_problems(self):
        assert self._spec().validate() == []

    def test_detects_edge_to_undeclared_component(self):
        spec = self._spec(component_graph={"a": ["ghost"]})
        assert any("ghost" in p for p in spec.validate())

    def test_detects_metric_pinned_to_undeclared_component(self):
        spec = self._spec(metrics={"m": MetricQuery("up", component="ghost")})
        assert any("ghost" in p for p in spec.validate())

    def test_detects_unknown_metric_in_component_metrics(self):
        spec = self._spec(component_metrics={"a": ("ghost_metric",)})
        assert any("ghost_metric" in p for p in spec.validate())

    def test_detects_exogenous_node_outside_graph(self):
        spec = self._spec(exogenous=frozenset({"ghost"}))
        assert any("ghost" in p for p in spec.validate())

    def test_detects_sli_node_outside_graph(self):
        spec = self._spec(sli_node="ghost")
        assert any("ghost" in p for p in spec.validate())

    def test_detects_unattributable_domain(self):
        spec = self._spec(metrics={"m": MetricQuery("up")})  # no component, no extractor
        assert any("attributed" in p for p in spec.validate())

    def test_register_rejects_an_invalid_spec(self):
        with pytest.raises(ValueError, match="inconsistent"):
            register_domain(self._spec(name="bad", component_graph={"a": ["ghost"]}))


class TestHistogramQuantile:

    def test_builds_a_bucket_rate_quantile(self):
        assert histogram_quantile("vllm:time_to_first_token_seconds", 0.99, "1m") == (
            "histogram_quantile(0.99, sum by (le) "
            "(rate(vllm:time_to_first_token_seconds_bucket[1m])))"
        )


# -----------------------------------------------------------------------
# The engine actually uses the supplied domain
# -----------------------------------------------------------------------

_TOY_COMPONENTS = ("root", "victim", "independent", "quiet")


def _toy_domain(**overrides) -> DomainSpec:
    """A 4-node causal graph standing in for a mechanism graph.

        root ──> victim        independent        quiet

    ``quiet`` never goes abnormal.  It is load-bearing: Layer 6 treats "every
    monitored component abnormal with one uniform trend" as an external cause
    and pinpoints nothing, which would mask any graph effect below.
    """
    base = dict(
        name="toy",
        metrics={
            "signal_a": MetricQuery("toy_a", component="root"),
            "signal_b": MetricQuery("toy_b", component="root"),
        },
        component_graph={
            "root": ["victim"],
            "victim": [],
            "independent": [],
            "quiet": [],
        },
        component_metrics={c: ("signal_a", "signal_b") for c in _TOY_COMPONENTS},
    )
    base.update(overrides)
    return DomainSpec(**base)


def _stepped(n_baseline: int, n_fault: int, onset_offset: int) -> dict[str, np.ndarray]:
    """Two metrics; both step up `onset_offset` samples into the fault window."""
    total = n_baseline + n_fault
    series = np.ones(total) * 0.1
    series[n_baseline + onset_offset:] = 0.9
    return {"signal_a": series.copy(), "signal_b": series.copy()}


def _flat(n: int) -> dict[str, np.ndarray]:
    return {"signal_a": np.ones(n) * 0.1, "signal_b": np.ones(n) * 0.1}


class TestPinpointHonoursDomain:
    """The graph Layers 7-8 filter against must come from the domain."""

    # root fires first; victim fires next but is downstream of root;
    # independent fires last with no path to anything.
    #
    # Offsets are spaced well apart because Layer 4 rolls each onset back by
    # ~8 samples: these produce detected onsets of roughly +1s, +12s, and
    # +32s, so the gaps clear Layer 7's 2s concurrency threshold.
    N_BL, N_FT = 20, 80

    def _matrix(self):
        return {
            "root": _stepped(self.N_BL, self.N_FT, 2),
            "victim": _stepped(self.N_BL, self.N_FT, 20),
            "independent": _stepped(self.N_BL, self.N_FT, 40),
            "quiet": _flat(self.N_BL + self.N_FT),
        }

    def _windows(self):
        bl_start = 1000.0
        bl_end = bl_start + self.N_BL * STEP_SECONDS
        ft_start = bl_end + 10.0
        ft_end = ft_start + self.N_FT * STEP_SECONDS
        return (bl_start, bl_end), (ft_start, ft_end)

    def test_earliest_onset_component_ranks_first(self):
        bl, ft = self._windows()
        result = pinpoint(self._matrix(), bl, ft, domain=_toy_domain())
        assert result[0]["service"] == "root"

    def test_dependency_edge_demotes_the_downstream_victim(self):
        """`victim` has an earlier onset than `independent`, but the toy graph
        explains it as propagation from `root`, so it must rank last.

        With a graph that has no such edge, ordering falls back to onset time
        and `victim` outranks `independent`. That difference is the proof the
        domain's graph — not a hardcoded one — drives Layers 7-8.
        """
        bl, ft = self._windows()
        matrix = self._matrix()

        with_edge = [e["service"] for e in pinpoint(matrix, bl, ft, domain=_toy_domain())]

        # Same components, same data, but no root -> victim edge.
        no_edge = [
            e["service"]
            for e in pinpoint(
                matrix,
                bl,
                ft,
                domain=_toy_domain(
                    component_graph={c: [] for c in _TOY_COMPONENTS}
                ),
            )
        ]

        assert with_edge.index("victim") > with_edge.index("independent")
        assert no_edge.index("victim") < no_edge.index("independent")

    def test_confidence_denominator_comes_from_the_domain(self):
        """A component backed by 2 metrics must not be scored out of 7."""
        bl, ft = self._windows()
        # Force the fallback path (no Layer 1 confidence scores) by scoring a
        # component whose metrics are all abnormal: 2/2 = 1.0 under the toy
        # domain, but would be 2/7 = 0.286 under Boutique's denominator.
        spec = _toy_domain()
        result = pinpoint(self._matrix(), bl, ft, domain=spec)
        root = next(e for e in result if e["service"] == "root")
        assert root["abnormal_metrics"] == ["signal_a", "signal_b"]
        assert root["confidence"] > 2 / 7

    def test_defaults_to_boutique_when_no_domain_given(self):
        """Omitting `domain` must behave exactly as before the layer existed."""
        bl, ft = self._windows()
        matrix = self._matrix()
        assert pinpoint(matrix, bl, ft) == pinpoint(matrix, bl, ft, domain=BOUTIQUE)


# -----------------------------------------------------------------------
# Exogenous inputs, the SLI node, and the verdict
# -----------------------------------------------------------------------

_MECH_COMPONENTS = ("load", "cache", "queueing", "latency", "quiet")


def _mechanism_domain(**overrides) -> DomainSpec:
    """A miniature mechanism graph shaped like the vLLM one.

        load ──> cache ──> queueing ──> latency          quiet

    ``load`` is exogenous (an input, not a defect) and ``latency`` is the SLI
    being explained.  Neither may be named the root cause.
    """
    base = dict(
        name="mech",
        metrics={
            "signal_a": MetricQuery("a", component="load"),
            "signal_b": MetricQuery("b", component="load"),
        },
        component_graph={
            "load": ["cache"],
            "cache": ["queueing"],
            "queueing": ["latency"],
            "latency": [],
            "quiet": [],
        },
        component_metrics={c: ("signal_a", "signal_b") for c in _MECH_COMPONENTS},
        exogenous=frozenset({"load"}),
        sli_node="latency",
    )
    base.update(overrides)
    return DomainSpec(**base)


class TestExogenousAndVerdict:

    N_BL, N_FT = 20, 80

    def _windows(self):
        bl_start = 1000.0
        bl_end = bl_start + self.N_BL * STEP_SECONDS
        ft_start = bl_end + 10.0
        ft_end = ft_start + self.N_FT * STEP_SECONDS
        return (bl_start, bl_end), (ft_start, ft_end)

    def _run(self, offsets: dict[str, int | None], **domain_kw):
        """Build a matrix from per-component step offsets and diagnose it."""
        matrix = {
            name: (
                _flat(self.N_BL + self.N_FT)
                if off is None
                else _stepped(self.N_BL, self.N_FT, off)
            )
            for name, off in offsets.items()
        }
        bl, ft = self._windows()
        return pinpoint_report(matrix, bl, ft, domain=_mechanism_domain(**domain_kw))

    def test_load_driven_incident_is_capacity_not_pathology(self):
        """Load moves first, everything else follows -> capacity."""
        report = self._run(
            {"load": 2, "cache": 20, "queueing": 40, "latency": 45, "quiet": None}
        )
        assert report.verdict == fault_chain.VERDICT_CAPACITY
        assert report.exogenous_drivers == ["load"]

    def test_capacity_verdict_never_names_the_input_as_root_cause(self):
        """`load` is real evidence but is not something to go fix."""
        report = self._run(
            {"load": 2, "cache": 20, "queueing": 40, "latency": 45, "quiet": None}
        )
        assert report.top() is not None
        assert report.top()["service"] != "load"
        # It still appears in the ranking as evidence.
        assert "load" in [e["service"] for e in report.ranked]

    def test_internal_first_mover_is_pathology(self):
        """Load stays flat; an internal component moves -> pathology."""
        report = self._run(
            {"load": None, "cache": 2, "queueing": 20, "latency": 40, "quiet": None}
        )
        assert report.verdict == fault_chain.VERDICT_PATHOLOGY
        assert report.top()["service"] == "cache"
        assert report.exogenous_drivers == []

    def test_sli_node_never_wins_even_when_it_moves_first(self):
        """TTFT is the symptom; 'latency is slow because latency is slow' is
        not a diagnosis."""
        report = self._run(
            {"load": None, "latency": 2, "cache": 20, "queueing": 40, "quiet": None}
        )
        assert report.top()["service"] != "latency"
        assert report.verdict == fault_chain.VERDICT_PATHOLOGY

    def test_only_exogenous_abnormal_pinpoints_nothing_internal(self):
        report = self._run(
            {"load": 2, "cache": None, "queueing": None, "latency": None, "quiet": None}
        )
        assert report.verdict == fault_chain.VERDICT_CAPACITY
        assert [e["service"] for e in report.ranked] == ["load"]

    def test_no_anomaly_when_everything_is_flat(self):
        report = self._run({c: None for c in _MECH_COMPONENTS})
        assert report.verdict == fault_chain.VERDICT_NO_ANOMALY
        assert report.ranked == []
        assert report.top() is None

    def test_uniform_trend_check_is_bypassed_for_mechanism_domains(self):
        """Every component abnormal and all rising would make the classic
        external-cause check bail out with nothing. A domain that declares
        exogenous nodes expresses that idea precisely instead, so it must
        still produce a ranking."""
        report = self._run(
            {"load": 2, "cache": 20, "queueing": 30, "latency": 40, "quiet": 50}
        )
        assert report.ranked, "mechanism domain should not bail out empty"
        assert report.verdict == fault_chain.VERDICT_CAPACITY

    def test_boutique_keeps_the_classic_external_cause_behaviour(self):
        """Boutique declares no exogenous nodes, so the uniform-trend check
        must still fire and pinpoint nothing."""
        n_bl, n_ft = 20, 80
        matrix = {
            svc: _stepped(n_bl, n_ft, 2)
            for svc in ("frontend", "cartservice", "adservice")
        }
        bl = (1000.0, 1000.0 + n_bl)
        ft = (bl[1] + 10.0, bl[1] + 10.0 + n_ft)
        report = pinpoint_report(matrix, bl, ft, domain=BOUTIQUE)
        assert report.verdict == fault_chain.VERDICT_EXTERNAL


class TestVllmDomain:
    """The vLLM mechanism graph, on synthetic telemetry.

    These do not prove the pipeline works on a real server — that needs the
    captured traces from Phase 4. They pin down the graph's structure and show
    the mechanism is separable in principle.
    """

    N_BL, N_FT = 20, 80

    def _windows(self):
        bl_start = 1000.0
        bl_end = bl_start + self.N_BL * STEP_SECONDS
        ft_start = bl_end + 10.0
        ft_end = ft_start + self.N_FT * STEP_SECONDS
        return (bl_start, bl_end), (ft_start, ft_end)

    def _diagnose(self, onsets: dict[str, int]):
        """Build a full vLLM metric matrix; components in `onsets` step at the
        given offset, all others stay flat."""
        matrix = {}
        for component in VLLM.component_graph:
            metrics = VLLM.component_metrics[component]
            offset = onsets.get(component)
            if offset is None:
                series = {m: np.ones(self.N_BL + self.N_FT) * 0.1 for m in metrics}
            else:
                stepped = _stepped(self.N_BL, self.N_FT, offset)["signal_a"]
                series = {m: stepped.copy() for m in metrics}
            matrix[component] = series
        bl, ft = self._windows()
        return pinpoint_report(matrix, bl, ft, domain=VLLM)

    def test_graph_is_acyclic(self):
        colour: dict[str, int] = {}

        def visit(node: str) -> bool:
            colour[node] = 1
            for nxt in VLLM.component_graph.get(node, []):
                if colour.get(nxt) == 1:
                    return True
                if colour.get(nxt) is None and visit(nxt):
                    return True
            colour[node] = 2
            return False

        assert not any(visit(n) for n in VLLM.component_graph if colour.get(n) is None)

    def test_inputs_and_symptom_cannot_be_named_the_cause(self):
        excluded = VLLM.excluded_from_root_cause()
        assert {"arrival_load", "request_shape", "ttft"} <= excluded
        # The eight real mechanisms stay eligible.
        assert "kv_cache_pressure" not in excluded
        assert "preemption" not in excluded
        assert "prefix_cache_efficacy" not in excluded
        assert "host_saturation" not in excluded

    def test_decode_health_is_excluded_as_a_co_symptom(self):
        """ITL cannot reach TTFT, so it is evidence, not a cause."""
        assert "decode_health" in VLLM.excluded_from_root_cause()

    def test_ttft_decomposes_into_queue_and_prefill(self):
        """The split that partitions admission-side from compute-side causes."""
        assert VLLM.component_graph["queueing"] == ["ttft"]
        assert VLLM.component_graph["prefill_cost"] == ["ttft"]
        assert "queue_time_p99" in VLLM.component_metrics["queueing"]
        assert "prefill_time_p99" in VLLM.component_metrics["prefill_cost"]

    # -- the confounder pair -------------------------------------------
    #
    # Both scenarios below produce a growing queue and a TTFT spike. Any rule
    # of the form "TTFT up and queue up implies KV cache pressure" gets one of
    # them wrong. Onset ordering plus the mechanism graph separates them.

    def test_kv_cache_pressure_scenario(self):
        report = self._diagnose(
            {"kv_cache_pressure": 2, "preemption": 20, "queueing": 30, "ttft": 40}
        )
        assert report.verdict == fault_chain.VERDICT_PATHOLOGY
        assert report.top()["service"] == "kv_cache_pressure"

    def test_host_saturation_scenario_is_not_blamed_on_the_cache(self):
        """Same surface symptoms, but the KV cache never moves: the API server
        and tokenizer are starved of CPU while the GPU sits idle."""
        report = self._diagnose({"host_saturation": 2, "queueing": 20, "ttft": 30})
        assert report.verdict == fault_chain.VERDICT_PATHOLOGY
        assert report.top()["service"] == "host_saturation"
        assert "kv_cache_pressure" not in [e["service"] for e in report.ranked]

    def test_prefix_cache_thrash_scenario(self):
        report = self._diagnose(
            {"prefix_cache_efficacy": 2, "prefill_cost": 20, "ttft": 30}
        )
        assert report.verdict == fault_chain.VERDICT_PATHOLOGY
        assert report.top()["service"] == "prefix_cache_efficacy"

    def test_load_spike_is_capacity_not_a_bug(self):
        report = self._diagnose(
            {"arrival_load": 2, "kv_cache_pressure": 20, "queueing": 30, "ttft": 40}
        )
        assert report.verdict == fault_chain.VERDICT_CAPACITY
        assert report.exogenous_drivers == ["arrival_load"]
        # The verdict says "overloaded", and the earliest internal mechanism is
        # still surfaced so the operator knows which limit was hit first.
        assert report.top()["service"] != "arrival_load"


class TestPinpointReportWrapper:

    def test_pinpoint_returns_the_reports_ranking(self):
        n_bl, n_ft = 20, 80
        matrix = {
            "currencyservice": _stepped(n_bl, n_ft, 2),
            "adservice": _flat(n_bl + n_ft),
        }
        bl = (1000.0, 1000.0 + n_bl)
        ft = (bl[1] + 10.0, bl[1] + 10.0 + n_ft)
        assert pinpoint(matrix, bl, ft) == pinpoint_report(matrix, bl, ft).ranked

    def test_report_records_the_domain(self):
        assert pinpoint_report({}, (0, 1), (2, 3)).domain == "boutique"
        assert (
            pinpoint_report({}, (0, 1), (2, 3), domain=_mechanism_domain()).domain
            == "mech"
        )
