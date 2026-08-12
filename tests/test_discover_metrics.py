"""Tests for the metric-surface discovery and drift check."""

from pathlib import Path

import pytest

from rca_engine.domains import VLLM, DomainSpec, MetricQuery
from rca_engine.scripts.discover_metrics import (
    base_metric_name,
    check_domain,
    parse_metric_surface,
    referenced_metrics,
)

FIXTURE = Path(__file__).parent / "fixtures" / "vllm_metrics_v1.txt"


@pytest.fixture(scope="module")
def surface() -> dict[str, str]:
    return parse_metric_surface(FIXTURE.read_text())


class TestReferencedMetrics:

    def test_extracts_a_simple_rate(self):
        assert referenced_metrics("sum(rate(vllm:prompt_tokens_total[30s]))") == {
            "vllm:prompt_tokens_total"
        }

    def test_ignores_functions_and_keywords(self):
        found = referenced_metrics(
            "histogram_quantile(0.99, sum by (le) (rate(vllm:foo_bucket[1m])))"
        )
        assert found == {"vllm:foo_bucket"}

    def test_ignores_label_matcher_contents(self):
        found = referenced_metrics(
            'rate(container_cpu_usage_seconds_total{namespace="vllm",container!=""}[30s])'
        )
        assert found == {"container_cpu_usage_seconds_total"}

    def test_ignores_labels_named_in_an_aggregation_clause(self):
        """`by (pod, namespace)` names labels, not metrics — and uses parens,
        so the label-matcher strip does not cover it."""
        found = referenced_metrics(
            "sum by (pod, namespace) (rate(container_cpu_cfs_periods_total[30s]))"
        )
        assert found == {"container_cpu_cfs_periods_total"}

    def test_handles_a_ratio_of_two_metrics(self):
        found = referenced_metrics(
            "sum(rate(vllm:prefix_cache_hits_total[30s]))"
            " / clamp_min(sum(rate(vllm:prefix_cache_queries_total[30s])), 1)"
        )
        assert found == {
            "vllm:prefix_cache_hits_total",
            "vllm:prefix_cache_queries_total",
        }


class TestBaseMetricName:

    @pytest.mark.parametrize(
        "sample,expected",
        [
            ("vllm:ttft_seconds_bucket", "vllm:ttft_seconds"),
            ("vllm:ttft_seconds_sum", "vllm:ttft_seconds"),
            ("vllm:ttft_seconds_count", "vllm:ttft_seconds"),
            ("vllm:num_requests_running", "vllm:num_requests_running"),
            # A counter's own _total suffix is part of its name, not a
            # histogram sample suffix.
            ("vllm:num_preemptions_total", "vllm:num_preemptions_total"),
        ],
    )
    def test_strips_only_sample_suffixes(self, sample, expected):
        assert base_metric_name(sample) == expected


class TestParseMetricSurface:

    def test_reads_types_from_type_headers(self, surface):
        assert surface["vllm:kv_cache_usage_perc"] == "gauge"
        assert surface["vllm:num_preemptions_total"] == "counter"
        assert surface["vllm:time_to_first_token_seconds"] == "histogram"

    def test_histogram_registers_under_its_base_name(self, surface):
        assert "vllm:request_queue_time_seconds" in surface
        # The _bucket/_sum/_count samples are not separate metrics.
        assert "vllm:request_queue_time_seconds_bucket" not in surface

    def test_picks_up_non_vllm_exporters(self, surface):
        """A domain draws from several targets; the surface unions them.

        No DCGM series here — the capture was CPU-only, which is exactly why
        the GPU metrics are marked optional in the domain.
        """
        assert surface["container_cpu_usage_seconds_total"] == "counter"
        assert not any(name.startswith("DCGM_") for name in surface)

    def test_records_untyped_samples(self):
        parsed = parse_metric_surface("some_metric_without_a_type_header 1.0\n")
        assert parsed == {"some_metric_without_a_type_header": "untyped"}

    def test_empty_input_gives_empty_surface(self):
        assert parse_metric_surface("") == {}


class TestCheckDomain:

    def test_vllm_domain_resolves_against_a_real_server(self, surface):
        """Every required metric must exist on a real server.

        The fixture is captured from `vllm/vllm-openai-cpu:latest-arm64`
        serving Qwen3-0.6B, not written by hand. An earlier hand-written
        version claimed `vllm:kv_block_reuse_gap_seconds` and
        `vllm:kv_block_idle_before_evict_seconds` exist — they appear in the
        docs but the shipped engine does not expose them. This test is what
        stops that happening again.
        """
        missing = check_domain(VLLM, surface)
        assert missing == {}, (
            "vLLM domain references metrics absent from a real server: "
            f"{missing}"
        )

    def test_exporter_gated_metrics_are_marked_optional(self, surface):
        """DCGM needs a GPU and cAdvisor's CFS counters need a real cgroup
        tree. Those absences are environmental, and must not be reported the
        same way as a wrong metric name."""
        optional_absent = set(check_domain(VLLM, surface, include_optional=True))
        assert optional_absent - set(check_domain(VLLM, surface)), (
            "expected some optional metrics to be absent from a CPU-only capture"
        )
        for name in optional_absent - set(check_domain(VLLM, surface)):
            assert VLLM.metrics[name].optional, f"{name} should be marked optional"

    def test_the_ttft_decomposition_signals_are_present(self, surface):
        """The domain's central claim: TTFT splits into queue and prefill.
        If either histogram were absent the whole approach would collapse."""
        assert "vllm:request_queue_time_seconds" in surface
        assert "vllm:request_prefill_time_seconds" in surface
        assert "vllm:time_to_first_token_seconds" in surface

    def test_detects_a_renamed_metric(self, surface):
        """The V0 -> V1 rename this check exists to catch."""
        stale = DomainSpec(
            name="stale",
            # gpu_cache_usage_perc was the V0 name for kv_cache_usage_perc.
            metrics={
                "cache": MetricQuery(
                    "sum(vllm:gpu_cache_usage_perc)", component="kv"
                )
            },
            component_graph={"kv": []},
        )
        missing = check_domain(stale, surface)
        assert missing == {"cache": ["vllm:gpu_cache_usage_perc"]}

    def test_reports_every_missing_reference_in_one_query(self, surface):
        spec = DomainSpec(
            name="bad",
            metrics={
                "ratio": MetricQuery(
                    "sum(vllm:nope_a_total) / sum(vllm:nope_b_total)", component="c"
                )
            },
            component_graph={"c": []},
        )
        assert check_domain(spec, surface) == {
            "ratio": ["vllm:nope_a_total", "vllm:nope_b_total"]
        }

    def test_empty_surface_flags_every_required_metric(self):
        required = [n for n, q in VLLM.metrics.items() if not q.optional]
        assert set(check_domain(VLLM, {})) == set(required)
        assert set(check_domain(VLLM, {}, include_optional=True)) == set(VLLM.metrics)
