"""Tests for the experiment label written by the injectors.

This record is what the evaluation scores a diagnosis against. A wrong label
is worse than a wrong diagnosis: it makes a correct pipeline look broken, and
there is no way to tell the two apart after the fact. So validation is strict
and happens at write time.
"""

import json

import pytest

import fault_injection.ground_truth as gt
from rca_engine.domains import VLLM as VLLM_DOMAIN
from workload.scenarios import SCENARIOS, get_scenario


class TestBoutiqueLabels:
    """The original schema must be untouched — two injectors depend on it."""

    def test_writes_and_reloads(self, tmp_path):
        path = gt.write("20260812_120000", "cpu_hog", ["cartservice"], 120, tmp_path)
        record = gt.load(path)
        assert record["fault_type"] == "cpu_hog"
        assert record["target_services"] == ["cartservice"]
        assert record["domain"] == "boutique"

    def test_defaults_to_the_boutique_domain(self, tmp_path):
        record = gt.load(gt.write("r", "cpu_hog", ["frontend"], 60, tmp_path))
        assert record["domain"] == "boutique"

    def test_rejects_an_unknown_fault(self, tmp_path):
        with pytest.raises(ValueError, match="unknown fault_type"):
            gt.write("r", "not_a_fault", ["frontend"], 60, tmp_path)

    def test_still_requires_a_target_service(self, tmp_path):
        with pytest.raises(ValueError, match="non-empty"):
            gt.write("r", "cpu_hog", [], 60, tmp_path)

    def test_rejects_a_non_positive_duration(self, tmp_path):
        with pytest.raises(ValueError, match="positive integer"):
            gt.write("r", "cpu_hog", ["frontend"], 0, tmp_path)

    def test_a_record_without_a_domain_key_still_validates(self):
        """Files written before the domain field existed must still load."""
        gt.validate(
            {
                "run_id": "r",
                "fault_type": "cpu_hog",
                "target_services": ["frontend"],
                "inject_time_utc": "2026-04-13T00:00:00+00:00",
                "duration_seconds": 60,
            }
        )


class TestVllmLabels:

    def test_every_scenario_produces_a_valid_label(self, tmp_path):
        """The strongest check in the file: the whole fault library round-trips
        through validation, so no scenario can be labelled with a component the
        pipeline could never name."""
        for name, scenario in SCENARIOS.items():
            path = gt.write_scenario(f"run_{name}", scenario, 180, tmp_path / name)
            record = gt.load(path)
            assert record["scenario"] == name
            assert record["root_cause"] == scenario.ground_truth
            assert record["expect_verdict"] == scenario.expect_verdict

    def test_target_services_may_be_empty(self, tmp_path):
        """A vLLM fault targets a mechanism, not a service."""
        record = gt.load(
            gt.write_scenario("r", get_scenario("qps_ramp"), 180, tmp_path)
        )
        assert record["target_services"] == []

    def test_clean_run_records_none_rather_than_omitting_the_key(self, tmp_path):
        """'The correct answer is nothing' is a label. Omitting the key would
        make a clean run indistinguishable from an unlabelled one."""
        path = gt.write_scenario("r", get_scenario("clean"), 180, tmp_path)
        raw = json.loads(path.read_text())
        assert "root_cause" in raw
        assert raw["root_cause"] is None
        assert raw["expect_verdict"] == "no_anomaly"

    def test_rejects_an_unknown_scenario(self, tmp_path):
        with pytest.raises(ValueError, match="unknown scenario"):
            gt.write(
                "r", "invented_scenario", [], 60, tmp_path,
                domain="vllm", root_cause="queueing",
            )

    def test_rejects_a_component_that_is_not_in_the_graph(self, tmp_path):
        with pytest.raises(ValueError, match="not a component"):
            gt.write(
                "r", "qps_ramp", [], 60, tmp_path,
                domain="vllm", scenario="qps_ramp", root_cause="imaginary",
            )

    @pytest.mark.parametrize("component", ["ttft", "arrival_load", "decode_health"])
    def test_rejects_a_component_that_can_never_be_pinpointed(self, tmp_path, component):
        """Labelling a run with the SLI, an exogenous input, or a co-symptom
        would guarantee a permanent zero score for a reason that has nothing to
        do with the pipeline."""
        assert component in VLLM_DOMAIN.excluded_from_root_cause()
        with pytest.raises(ValueError, match="never be pinpointed"):
            gt.write(
                "r", "qps_ramp", [], 60, tmp_path,
                domain="vllm", scenario="qps_ramp", root_cause=component,
            )

    def test_requires_root_cause_to_be_present(self, tmp_path):
        with pytest.raises(ValueError, match="must record root_cause"):
            gt.validate(
                {
                    "run_id": "r",
                    "fault_type": "qps_ramp",
                    "target_services": [],
                    "inject_time_utc": "2026-08-12T00:00:00+00:00",
                    "duration_seconds": 60,
                    "domain": "vllm",
                }
            )

    def test_rejects_an_unknown_domain(self, tmp_path):
        with pytest.raises(ValueError, match="unknown domain"):
            gt.write("r", "cpu_hog", ["frontend"], 60, tmp_path, domain="sglang")
