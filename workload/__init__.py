"""Workload shaping and the labeled fault library for vLLM experiments.

For an inference server the workload shape *is* most of the fault library:
changing what the client sends is enough to induce KV cache pressure,
prefill saturation, prefix-cache thrash, and batch-composition swings, with no
restarts and no privileged access.
"""

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

__all__ = [
    "CONFIG",
    "INFRA",
    "NOMINAL",
    "SCENARIOS",
    "WORKLOAD",
    "Phase",
    "Scenario",
    "confounder_pairs",
    "get_scenario",
    "scenarios_for",
]
