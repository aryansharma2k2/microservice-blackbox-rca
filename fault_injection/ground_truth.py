"""Ground-truth JSON writer and validator for fault injection experiments.

This is the label the evaluation scores against.  Until now it was written by
the injectors and never read back by anything, so nothing ever compared a
diagnosis to what was actually injected.  Closing that loop is the point of
the evaluation phase, and it starts with the label being expressive enough.

Boutique schema (unchanged)::

    {
        "run_id":            str,   # e.g. "20260326_143022"
        "fault_type":        str,   # cpu_hog | mem_leak | net_delay | disk_hog
        "target_services":   list[str],
        "inject_time_utc":   str,   # ISO-8601
        "duration_seconds":  int
    }

vLLM runs add, and relax, the following::

    {
        "domain":            "vllm",
        "scenario":          str,        # name from workload.scenarios
        "root_cause":        str | None, # mechanism component; None on a clean run
        "expect_verdict":    str,        # pathology | capacity | no_anomaly
        "target_services":   []          # may be empty: the target is a mechanism
    }

``fault_type`` carries the scenario name for vLLM runs, so the field means
"what was injected" in both domains.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_FIELDS = {"run_id", "fault_type", "target_services", "inject_time_utc", "duration_seconds"}
VALID_FAULTS = {"cpu_hog", "mem_leak", "net_delay", "disk_hog", "packet_loss"}

BOUTIQUE = "boutique"
VLLM = "vllm"
VALID_DOMAINS = {BOUTIQUE, VLLM}


def make_run_id() -> str:
    """Return a timestamp-based run ID, e.g. '20260326_143022'."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def write(
    run_id: str,
    fault_type: str,
    target_services: list[str],
    duration_seconds: int,
    output_dir: Path,
    domain: str = BOUTIQUE,
    scenario: str | None = None,
    root_cause: str | None = None,
    expect_verdict: str | None = None,
) -> Path:
    """Write a ground_truth.json into *output_dir* and return its path.

    Creates *output_dir* if it doesn't exist.
    Raises ValueError if the inputs don't pass validation.

    The domain-specific arguments default to None so the two Boutique callers
    are unaffected.
    """
    record = {
        "run_id": run_id,
        "fault_type": fault_type,
        "target_services": target_services,
        "inject_time_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration_seconds,
        "domain": domain,
    }
    if scenario is not None:
        record["scenario"] = scenario
    if expect_verdict is not None:
        record["expect_verdict"] = expect_verdict
    # Written even when None: on a clean run "the correct answer is nothing"
    # is a real label, and omitting the key would make it indistinguishable
    # from an unlabelled run.
    if domain == VLLM or root_cause is not None:
        record["root_cause"] = root_cause

    validate(record)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "ground_truth.json"
    path.write_text(json.dumps(record, indent=2))
    return path


def write_scenario(
    run_id: str,
    scenario,
    duration_seconds: int,
    output_dir: Path,
) -> Path:
    """Write ground truth for a vLLM run from a ``workload.scenarios.Scenario``.

    Preferred over :func:`write` for vLLM, because the label is then derived
    from the scenario definition rather than restated at the call site, where
    it could drift.
    """
    return write(
        run_id=run_id,
        fault_type=scenario.name,
        target_services=[],
        duration_seconds=duration_seconds,
        output_dir=output_dir,
        domain=VLLM,
        scenario=scenario.name,
        root_cause=scenario.ground_truth,
        expect_verdict=scenario.expect_verdict,
    )


def load(path: Path) -> dict:
    """Load and validate a ground_truth.json file."""
    record = json.loads(Path(path).read_text())
    validate(record)
    return record


def validate(record: dict) -> None:
    """Raise ValueError if *record* is missing fields or has invalid values."""
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        raise ValueError(f"ground_truth missing fields: {missing}")

    domain = record.get("domain", BOUTIQUE)
    if domain not in VALID_DOMAINS:
        raise ValueError(f"unknown domain '{domain}' — must be one of {VALID_DOMAINS}")

    if not isinstance(record["duration_seconds"], int) or record["duration_seconds"] <= 0:
        raise ValueError("duration_seconds must be a positive integer")

    if not isinstance(record["target_services"], list):
        raise ValueError("target_services must be a list")

    if domain == BOUTIQUE:
        if record["fault_type"] not in VALID_FAULTS:
            raise ValueError(
                f"unknown fault_type '{record['fault_type']}' — must be one of {VALID_FAULTS}"
            )
        if not record["target_services"]:
            raise ValueError("target_services must be a non-empty list")
        return

    _validate_vllm(record)


def _validate_vllm(record: dict) -> None:
    """Check a vLLM label against the scenario library and the domain graph.

    A scenario labelled with a component that does not exist, or one the
    pipeline can never pinpoint, would score as a miss forever while the real
    problem is the label. Catch it at write time.
    """
    # Imported lazily: the injectors run as standalone scripts and should not
    # pay for the domain/scenario imports unless a vLLM run needs them.
    from rca_engine.domains import VLLM as VLLM_DOMAIN
    from workload.scenarios import SCENARIOS

    scenario_name = record.get("scenario", record["fault_type"])
    if scenario_name not in SCENARIOS:
        raise ValueError(
            f"unknown scenario '{scenario_name}' — must be one of "
            f"{sorted(SCENARIOS)}"
        )

    if "root_cause" not in record:
        raise ValueError("vllm ground_truth must record root_cause (None for clean runs)")

    root_cause = record["root_cause"]
    if root_cause is None:
        return

    if root_cause not in VLLM_DOMAIN.component_graph:
        raise ValueError(
            f"root_cause '{root_cause}' is not a component of the vllm domain"
        )

    excluded = VLLM_DOMAIN.excluded_from_root_cause()
    if root_cause in excluded:
        raise ValueError(
            f"root_cause '{root_cause}' can never be pinpointed — it is "
            f"excluded from root-cause candidacy ({sorted(excluded)}). "
            "The label is wrong, or the domain graph is."
        )
