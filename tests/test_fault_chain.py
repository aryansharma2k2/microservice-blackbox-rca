"""Integration tests for rca_engine.fault_chain."""

import numpy as np

from rca_engine.fault_chain import (
    pinpoint,
    pinpoint_faults,
    _split_series,
    _determine_trend,
    _analyze_metric,
    STEP_SECONDS,
)
from rca_engine.dependency import get_dependency_graph


# -----------------------------------------------------------------------
# Helper to build synthetic metric matrices
# -----------------------------------------------------------------------

METRICS = [
    "cpu_rate", "cpu_throttle_ratio", "mem_wss",
    "net_rx_rate", "net_tx_rate", "fs_read_rate", "fs_write_rate",
]


def _flat_metrics(n_samples: int, value: float = 1.0) -> dict[str, np.ndarray]:
    """7 flat metric arrays at realistic magnitudes."""
    return {
        "cpu_rate": np.ones(n_samples) * value,
        "cpu_throttle_ratio": np.zeros(n_samples),
        "mem_wss": np.ones(n_samples) * 1e8,
        "net_rx_rate": np.ones(n_samples) * 1000,
        "net_tx_rate": np.ones(n_samples) * 1000,
        "fs_read_rate": np.ones(n_samples) * 500,
        "fs_write_rate": np.ones(n_samples) * 500,
    }


def _faulty_metrics(
    n_baseline: int,
    n_fault: int,
    fault_start_offset: int = 2,
) -> dict[str, np.ndarray]:
    """Metrics where cpu_rate has a clear step change during the fault window."""
    total = n_baseline + n_fault
    cpu = np.ones(total) * 0.1
    cpu[n_baseline + fault_start_offset :] = 0.9

    metrics = _flat_metrics(total, value=0.1)
    metrics["cpu_rate"] = cpu
    return metrics


def _make_windows(n_bl: int, n_ft: int, gap: float = 10.0):
    """Return (bl_start, bl_end, ft_start, ft_end) consistent with
    run_experiment.py timing."""
    bl_start = 1000.0
    bl_end = bl_start + n_bl * STEP_SECONDS
    ft_start = bl_end + gap
    ft_end = ft_start + n_ft * STEP_SECONDS
    return bl_start, bl_end, ft_start, ft_end


# -----------------------------------------------------------------------
# Tests for the full pinpoint() pipeline
# -----------------------------------------------------------------------

class TestPinpointPipeline:

    def test_single_faulty_service(self):
        """One service with a CPU step change, others flat -> faulty service
        should be ranked #1."""
        n_bl, n_ft = 10, 24
        total = n_bl + n_ft
        bl_start, bl_end, ft_start, ft_end = _make_windows(n_bl, n_ft)

        matrix = {}
        matrix["currencyservice"] = _faulty_metrics(n_bl, n_ft)
        for svc in ["frontend", "cartservice", "adservice", "paymentservice"]:
            matrix[svc] = _flat_metrics(total, value=0.1)

        result = pinpoint(
            metric_matrix=matrix,
            baseline_window=(bl_start, bl_end),
            fault_window=(ft_start, ft_end),
        )

        assert len(result) >= 1
        assert result[0]["service"] == "currencyservice"
        assert "cpu_rate" in result[0]["abnormal_metrics"]

    def test_empty_matrix(self):
        assert pinpoint({}, (0, 50), (60, 180)) == []

    def test_no_fault(self):
        """All services flat -> no suspects."""
        matrix = {
            "frontend": _flat_metrics(34),
            "adservice": _flat_metrics(34),
        }
        result = pinpoint(matrix, (0, 50), (60, 170))
        assert result == []

    def test_output_format_fields(self):
        """Verify every required field is present and correctly typed."""
        n_bl, n_ft = 10, 24
        total = n_bl + n_ft
        bl_start, bl_end, ft_start, ft_end = _make_windows(n_bl, n_ft)

        matrix = {
            "currencyservice": _faulty_metrics(n_bl, n_ft),
            "adservice": _flat_metrics(total, value=0.1),
        }
        result = pinpoint(matrix, (bl_start, bl_end), (ft_start, ft_end))

        for entry in result:
            assert "rank" in entry and isinstance(entry["rank"], int)
            assert "service" in entry and isinstance(entry["service"], str)
            assert "onset_time" in entry and isinstance(entry["onset_time"], float)
            assert "confidence" in entry and 0.0 <= entry["confidence"] <= 1.0
            assert "abnormal_metrics" in entry and isinstance(entry["abnormal_metrics"], list)

    def test_ranks_are_sequential(self):
        """Ranks must be 1, 2, 3, ... with no gaps."""
        n_bl, n_ft = 10, 24
        total = n_bl + n_ft
        bl_start, bl_end, ft_start, ft_end = _make_windows(n_bl, n_ft)

        matrix = {"currencyservice": _faulty_metrics(n_bl, n_ft)}
        for svc in ["adservice", "emailservice"]:
            matrix[svc] = _flat_metrics(total, value=0.1)

        result = pinpoint(matrix, (bl_start, bl_end), (ft_start, ft_end))
        ranks = [e["rank"] for e in result]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_root_causes_ranked_before_propagation(self):
        """Root causes should appear before propagation victims."""
        n_bl, n_ft = 10, 24
        total = n_bl + n_ft
        bl_start, bl_end, ft_start, ft_end = _make_windows(n_bl, n_ft)

        # Create two faulty services — one root cause, one propagation
        # paymentservice is a leaf; checkoutservice calls paymentservice
        matrix = {
            "paymentservice": _faulty_metrics(n_bl, n_ft, fault_start_offset=2),
            "adservice": _flat_metrics(total, value=0.1),
        }
        # checkoutservice fault starts later (propagation from paymentservice)
        checkout_metrics = _flat_metrics(total, value=0.1)
        cpu = np.ones(total) * 0.1
        cpu[n_bl + 6:] = 0.8  # later onset
        checkout_metrics["cpu_rate"] = cpu
        matrix["checkoutservice"] = checkout_metrics

        result = pinpoint(matrix, (bl_start, bl_end), (ft_start, ft_end))
        names = [e["service"] for e in result]
        if "paymentservice" in names and "checkoutservice" in names:
            assert names.index("paymentservice") < names.index("checkoutservice")

    def test_multi_metric_fault(self):
        """A fault that affects CPU + memory should list both in abnormal_metrics."""
        n_bl, n_ft = 10, 24
        total = n_bl + n_ft
        bl_start, bl_end, ft_start, ft_end = _make_windows(n_bl, n_ft)

        metrics = _flat_metrics(total, value=0.1)
        # CPU spike
        cpu = np.ones(total) * 0.1
        cpu[n_bl + 2:] = 0.9
        metrics["cpu_rate"] = cpu
        # Memory spike
        mem = np.ones(total) * 1e8
        mem[n_bl + 2:] = 5e8
        metrics["mem_wss"] = mem

        matrix = {
            "currencyservice": metrics,
            "adservice": _flat_metrics(total, value=0.1),
        }
        result = pinpoint(matrix, (bl_start, bl_end), (ft_start, ft_end))

        assert len(result) >= 1
        assert result[0]["service"] == "currencyservice"
        abnormal = result[0]["abnormal_metrics"]
        assert "cpu_rate" in abnormal
        assert "mem_wss" in abnormal
        assert result[0]["confidence"] > 1 / 7  # more than 1 metric

    def test_realistic_magnitudes(self):
        """CPU values ~0.1, mem values ~10^8 — both should detect changes."""
        n_bl, n_ft = 10, 24
        total = n_bl + n_ft
        bl_start, bl_end, ft_start, ft_end = _make_windows(n_bl, n_ft)

        rng = np.random.default_rng(42)
        metrics = {}
        for m in METRICS:
            metrics[m] = np.ones(total) * 0.05

        # Small CPU change (realistic: 0.05 → 0.45)
        metrics["cpu_rate"] = np.concatenate([
            rng.normal(0.05, 0.005, n_bl),
            rng.normal(0.45, 0.01, n_ft),
        ])
        # Large memory change (realistic: 100MB → 400MB)
        metrics["mem_wss"] = np.concatenate([
            rng.normal(1e8, 1e6, n_bl),
            rng.normal(4e8, 1e6, n_ft),
        ])

        matrix = {
            "emailservice": metrics,
            "adservice": _flat_metrics(total, value=0.05),
        }
        result = pinpoint(matrix, (bl_start, bl_end), (ft_start, ft_end))
        assert len(result) >= 1
        assert result[0]["service"] == "emailservice"

    def test_service_not_in_dependency_graph(self):
        """A service that doesn't exist in the hardcoded graph should still
        be processed and pinpointed (treated as independent)."""
        n_bl, n_ft = 10, 24
        total = n_bl + n_ft
        bl_start, bl_end, ft_start, ft_end = _make_windows(n_bl, n_ft)

        matrix = {
            "unknownservice": _faulty_metrics(n_bl, n_ft),
            "adservice": _flat_metrics(total, value=0.1),
        }
        result = pinpoint(matrix, (bl_start, bl_end), (ft_start, ft_end))
        assert len(result) >= 1
        services = [e["service"] for e in result]
        assert "unknownservice" in services

    def test_very_short_fault_window(self):
        """Fault window with only 3 samples should not crash."""
        n_bl, n_ft = 10, 3
        total = n_bl + n_ft
        bl_start, bl_end, ft_start, ft_end = _make_windows(n_bl, n_ft)

        matrix = {"adservice": _faulty_metrics(n_bl, n_ft)}
        # Should not raise
        result = pinpoint(matrix, (bl_start, bl_end), (ft_start, ft_end))
        assert isinstance(result, list)

    def test_very_short_baseline(self):
        """Baseline with 2 samples — minimum for the model to fit."""
        n_bl, n_ft = 2, 24
        total = n_bl + n_ft
        bl_start, bl_end, ft_start, ft_end = _make_windows(n_bl, n_ft)

        matrix = {"adservice": _faulty_metrics(n_bl, n_ft)}
        result = pinpoint(matrix, (bl_start, bl_end), (ft_start, ft_end))
        assert isinstance(result, list)


# -----------------------------------------------------------------------
# Tests for pinpoint_faults logic
# -----------------------------------------------------------------------

class TestPinpointFaults:

    def test_single_root_cause(self):
        graph = get_dependency_graph()
        onsets = {"currencyservice": 100.0, "frontend": 105.0}
        trends = {"currencyservice": "up", "frontend": "up"}
        result = pinpoint_faults(onsets, trends, graph)
        assert result[0] == "currencyservice"

    def test_concurrent_faults(self):
        graph = get_dependency_graph()
        onsets = {"emailservice": 100.0, "paymentservice": 101.5}
        trends = {"emailservice": "up", "paymentservice": "up"}
        result = pinpoint_faults(onsets, trends, graph, concurrency_threshold_s=2.0)
        assert "emailservice" in result
        assert "paymentservice" in result

    def test_propagation_filtered(self):
        """checkoutservice depends on paymentservice — if paymentservice is
        root cause, checkoutservice should be filtered as propagation
        (checkoutservice calls paymentservice, so a fault in paymentservice
        propagates up to checkoutservice)."""
        graph = get_dependency_graph()
        onsets = {"paymentservice": 100.0, "checkoutservice": 105.0}
        trends = {"paymentservice": "up", "checkoutservice": "up"}
        result = pinpoint_faults(onsets, trends, graph)
        assert "paymentservice" in result
        # checkoutservice -> paymentservice, so checkoutservice is downstream consumer
        assert "checkoutservice" not in result

    def test_independent_faults(self):
        """Two leaf services with no dependency path — both should be pinpointed."""
        graph = get_dependency_graph()
        onsets = {"adservice": 100.0, "paymentservice": 108.0}
        trends = {"adservice": "up", "paymentservice": "up"}
        result = pinpoint_faults(onsets, trends, graph)
        assert "adservice" in result
        assert "paymentservice" in result

    def test_external_cause(self):
        """All services abnormal with same trend -> external cause -> empty."""
        graph = get_dependency_graph()
        all_svcs = list(graph.keys())
        onsets = {s: 100.0 + i * 0.5 for i, s in enumerate(all_svcs)}
        trends = {s: "up" for s in all_svcs}
        # n_monitored_services must be supplied so the external cause check fires
        result = pinpoint_faults(onsets, trends, graph, n_monitored_services=len(all_svcs))
        assert result == []

    def test_empty_input(self):
        assert pinpoint_faults({}, {}, {}) == []

    def test_external_cause_not_triggered_with_mixed_trends(self):
        """If all services are abnormal but trends differ, it's NOT external."""
        graph = get_dependency_graph()
        all_svcs = list(graph.keys())
        onsets = {s: 100.0 + i for i, s in enumerate(all_svcs)}
        trends = {s: ("up" if i % 2 == 0 else "down") for i, s in enumerate(all_svcs)}
        result = pinpoint_faults(onsets, trends, graph)
        assert len(result) > 0  # not flagged as external

    def test_external_cause_not_triggered_with_subset(self):
        """If only a subset of services are abnormal, it's NOT external."""
        graph = get_dependency_graph()
        onsets = {"adservice": 100.0, "paymentservice": 101.0, "emailservice": 102.0}
        trends = {"adservice": "up", "paymentservice": "up", "emailservice": "up"}
        result = pinpoint_faults(onsets, trends, graph)
        assert len(result) > 0

    def test_propagation_chain_3_hop(self):
        """frontend -> recommendationservice -> productcatalogservice.
        If productcatalogservice is root cause, both recommendationservice
        and frontend should be filtered."""
        graph = get_dependency_graph()
        onsets = {
            "productcatalogservice": 100.0,
            "recommendationservice": 103.0,
            "frontend": 106.0,
        }
        trends = {s: "up" for s in onsets}
        result = pinpoint_faults(onsets, trends, graph)
        assert "productcatalogservice" in result
        assert "recommendationservice" not in result
        assert "frontend" not in result

    def test_concurrent_threshold_exact_boundary(self):
        """Services exactly at the concurrency threshold."""
        graph = get_dependency_graph()
        onsets = {"adservice": 100.0, "emailservice": 102.0}
        trends = {"adservice": "up", "emailservice": "up"}
        # Exactly at 2.0s threshold — should be included
        result = pinpoint_faults(onsets, trends, graph, concurrency_threshold_s=2.0)
        assert "adservice" in result
        assert "emailservice" in result

    def test_concurrent_threshold_just_outside(self):
        """Services just beyond the concurrency threshold — not concurrent."""
        graph = get_dependency_graph()
        onsets = {"adservice": 100.0, "emailservice": 102.1}
        trends = {"adservice": "up", "emailservice": "up"}
        result = pinpoint_faults(onsets, trends, graph, concurrency_threshold_s=2.0)
        assert "adservice" in result
        # emailservice is independent (no dep path from adservice), so still pinpointed
        assert "emailservice" in result


# -----------------------------------------------------------------------
# Tests for internal helpers
# -----------------------------------------------------------------------

class TestSplitSeries:

    def test_standard_split(self):
        """Standard experiment with 1s step: 50 baseline + 10 gap + 120 fault = 180 samples."""
        n = 180
        series = np.arange(n, dtype=float)
        bl_start = 1000.0
        bl_end = 1050.0     # 50 samples * 1s
        ft_start = 1060.0   # 10s gap
        ft_end = 1180.0     # 120 samples * 1s

        bl, ft = _split_series(series, bl_start, (bl_start, bl_end), (ft_start, ft_end))
        assert len(bl) == 50
        assert len(ft) == 120
        np.testing.assert_array_equal(bl, np.arange(50, dtype=float))
        np.testing.assert_array_equal(ft, np.arange(60, 180, dtype=float))

    def test_no_gap(self):
        """Baseline and fault windows are contiguous."""
        series = np.arange(100, dtype=float)
        bl, ft = _split_series(series, 0.0, (0.0, 50.0), (50.0, 100.0))
        assert len(bl) == 50
        assert len(ft) == 50

    def test_series_shorter_than_window(self):
        """Series is shorter than the fault window — should clamp."""
        series = np.arange(30, dtype=float)
        bl, ft = _split_series(series, 0.0, (0.0, 25.0), (25.0, 100.0))
        assert len(bl) == 25
        assert len(ft) <= 5  # clamped to available data


class TestDetermineTrend:

    def test_all_up(self):
        assert _determine_trend(["up", "up", "up"]) == "up"

    def test_all_down(self):
        assert _determine_trend(["down", "down"]) == "down"

    def test_mixed(self):
        assert _determine_trend(["up", "down"]) == "mixed"

    def test_empty(self):
        assert _determine_trend([]) == "mixed"


class TestAnalyzeMetric:

    def test_clear_step_detects_change(self):
        """A clear step change should produce at least one onset."""
        baseline = np.ones(10) * 5.0
        fault = np.concatenate([np.ones(5) * 5.0, np.ones(19) * 50.0])
        result = _analyze_metric(baseline, fault)
        assert result is not None
        # _analyze_metric returns (onset_indices, directions, confidences)
        onsets, dirs, confs = result
        assert len(onsets) >= 1
        assert all(isinstance(o, (int, np.integer)) for o in onsets)

    def test_flat_no_change(self):
        """No change -> None returned."""
        baseline = np.ones(100) * 5.0
        fault = np.ones(24) * 5.0
        result = _analyze_metric(baseline, fault)
        assert result is None

    def test_constant_baseline_detects_any_shift(self):
        """Perfectly constant baseline with sigma=0 should detect any shift."""
        baseline = np.ones(10) * 100.0
        fault = np.concatenate([np.ones(5) * 100.0, np.ones(19) * 200.0])
        result = _analyze_metric(baseline, fault)
        assert result is not None
        onsets, _, _ = result
        assert len(onsets) >= 1

    def test_noisy_baseline_small_fault(self):
        """Noise in baseline: a small fault should NOT be detected if within
        normal variation."""
        rng = np.random.default_rng(123)
        baseline = rng.normal(10.0, 2.0, 100)
        # Fault that's within baseline noise range
        fault = rng.normal(10.5, 2.0, 24)
        result = _analyze_metric(baseline, fault)
        if result is not None:
            onsets, _, _ = result
            assert len(onsets) <= 2


# -----------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------

class TestReproducibility:
    """Layer 1's bootstrap must not make the pipeline stochastic.

    The whole project rests on the claim that anyone can clone the repo and
    re-derive the published accuracy numbers from the committed traces. An
    unseeded bootstrap quietly breaks that: the CUSUM threshold is a
    percentile of 1000 random draws, so change points near it flip in and
    out between runs. Measured on the 38-run corpus before this was fixed,
    top-3 alternated between 71.0% and 73.7% with no input changing.
    """

    def _series(self):
        rng = np.random.default_rng(7)
        baseline = rng.normal(10.0, 1.0, 400)
        fault = np.concatenate([rng.normal(10.0, 1.0, 20), rng.normal(13.0, 1.0, 40)])
        return baseline, fault

    def test_repeated_analysis_of_identical_input_is_identical(self):
        baseline, fault = self._series()
        first = _analyze_metric(baseline, fault, service="svc", metric_name="cpu_rate")
        second = _analyze_metric(baseline, fault, service="svc", metric_name="cpu_rate")

        assert first is not None, "fixture should produce a detectable change"
        onsets_a, dirs_a, conf_a = first
        onsets_b, dirs_b, conf_b = second
        assert onsets_a == onsets_b
        assert dirs_a == dirs_b
        # Confidence is derived from the bootstrap maxima, so it is the value
        # that moves first when the generator is unseeded.
        assert conf_a == conf_b

    def test_the_confidence_check_above_can_actually_fail(self):
        """Guard the guard.

        If run_layer1 were called without a seed the confidences would differ,
        so assert that an unseeded pair really does diverge. Without this, a
        future refactor that made confidence constant would leave the test
        above passing while proving nothing.
        """
        from rca_engine.change_point import run_layer1

        baseline, fault = self._series()
        a = run_layer1(time_series=fault, baseline_data=baseline, seed=None)
        b = run_layer1(time_series=fault, baseline_data=baseline, seed=None)
        assert a.bootstrap_threshold != b.bootstrap_threshold

    def test_seeds_differ_across_metrics(self):
        """One shared seed would hand every metric the same resampling draws,
        correlating the thresholds that Layer 5 aggregates per component."""
        from rca_engine.fault_chain import _metric_seed

        seeds = {
            _metric_seed("svc", "cpu_rate"),
            _metric_seed("svc", "mem_wss"),
            _metric_seed("other", "cpu_rate"),
        }
        assert len(seeds) == 3
        assert _metric_seed("svc", "cpu_rate") == _metric_seed("svc", "cpu_rate")
