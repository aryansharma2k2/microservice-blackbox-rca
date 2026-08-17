"""Baselines the pipeline has to beat.

A top-1 number in isolation means nothing — the question is whether an
eight-layer statistical pipeline earns its complexity over what people
actually do, or over a one-liner. Three cheap baselines, all with the same
signature as the real diagnoser so the evaluation can score them identically:

``threshold``
    Static alert rules checked in the order an on-call engineer would check
    them. This is the honest state of the art: it is what a Grafana alert
    does, and it is what the pipeline has to be better than to be worth
    running.

``correlation``
    Rank every mechanism by how strongly its metrics correlate with the SLI
    over the fault window. The obvious data-driven approach, and the one that
    should fail precisely where this project is aimed: preemption is both a
    cause of TTFT and an effect of cache pressure, so correlation implicates
    both and separates neither.

``topology``
    Always blame the same node — whatever sits upstream in the graph.
    Ignores the telemetry entirely. Establishes the floor that any real
    method must clear, and catches a graph so lopsided that guessing wins.

Each returns a ranked list of component names, best first.
"""

from __future__ import annotations

import numpy as np

from rca_engine.domains import DomainSpec
from rca_engine.dependency import has_path


def _window_slice(
    series: np.ndarray,
    full_start: float,
    window: tuple[float, float],
    step: float,
) -> np.ndarray:
    lo = max(0, int((window[0] - full_start) / step))
    hi = min(len(series), int((window[1] - full_start) / step) + 1)
    return series[lo:hi]


def _split(matrix, baseline_window, fault_window, step):
    """Per-component {metric: (baseline_slice, fault_slice)}."""
    out: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    start = baseline_window[0]
    for component, metrics in matrix.items():
        for name, series in metrics.items():
            b = _window_slice(series, start, baseline_window, step)
            f = _window_slice(series, start, fault_window, step)
            if len(b) >= 2 and len(f) >= 2:
                out.setdefault(component, {})[name] = (b, f)
    return out


def _eligible(spec: DomainSpec) -> set[str]:
    return set(spec.component_graph) - spec.excluded_from_root_cause()


# ---------------------------------------------------------------------------
# 1. Static threshold alerting
# ---------------------------------------------------------------------------

#: (metric, comparison, threshold, component), checked in order. Values are
#: the round numbers that appear in real vLLM runbooks and dashboards.
VLLM_ALERT_RULES: tuple[tuple[str, str, float, str], ...] = (
    ("kv_cache_usage", ">", 0.90, "kv_cache_pressure"),
    ("preemption_rate", ">", 0.0, "preemption"),
    ("requests_waiting", ">", 5.0, "queueing"),
    ("prefix_cache_hit_rate", "<", 0.50, "prefix_cache_efficacy"),
    ("host_cpu_throttle_ratio", ">", 0.10, "host_saturation"),
    ("gpu_sm_active", ">", 0.95, "gpu_saturation"),
    ("prefill_time_p99", ">", 5.0, "prefill_cost"),
    ("requests_running", ">", 64.0, "batch_composition"),
)


def threshold(
    matrix, baseline_window, fault_window, spec: DomainSpec, step: float = 1.0
) -> list[str]:
    """Fire static alert rules against the fault window, in runbook order."""
    data = _split(matrix, baseline_window, fault_window, step)
    by_metric = {
        name: fault
        for metrics in data.values()
        for name, (_, fault) in metrics.items()
    }

    fired: list[str] = []
    for metric, op, limit, component in VLLM_ALERT_RULES:
        series = by_metric.get(metric)
        if series is None or component not in _eligible(spec):
            continue
        value = float(np.mean(series))
        if (op == ">" and value > limit) or (op == "<" and value < limit):
            if component not in fired:
                fired.append(component)
    return fired


# ---------------------------------------------------------------------------
# 2. Correlation with the SLI
# ---------------------------------------------------------------------------

def _is_flat(series: np.ndarray, rel_tol: float = 1e-9) -> bool:
    """Is this series effectively constant?

    Not ``std == 0``: the standard deviation of a constant float array is on
    the order of 1e-17, not zero, and ``corrcoef`` of two such arrays returns
    a spurious 1.0. Several vLLM metrics are genuinely constant at low request
    rates — a histogram quantile lands in the same coarse bucket every scrape
    — so without a tolerance this baseline reports perfect correlation between
    unrelated flat signals.
    """
    scale = max(abs(float(np.mean(series))), 1.0)
    return bool(np.std(series) <= rel_tol * scale)


def correlation(
    matrix, baseline_window, fault_window, spec: DomainSpec, step: float = 1.0
) -> list[str]:
    """Rank mechanisms by peak |Pearson r| against the SLI over the fault window."""
    if spec.sli_node is None:
        return []
    data = _split(matrix, baseline_window, fault_window, step)

    sli_metrics = data.get(spec.sli_node, {})
    if not sli_metrics:
        return []
    target = next(iter(sli_metrics.values()))[1]

    scores: dict[str, float] = {}
    for component, metrics in data.items():
        if component not in _eligible(spec):
            continue
        best = 0.0
        for _, fault in metrics.values():
            n = min(len(fault), len(target))
            a, b = fault[:n], target[:n]
            if n < 3 or _is_flat(a) or _is_flat(b):
                continue
            best = max(best, abs(float(np.corrcoef(a, b)[0, 1])))
        if best > 0:
            scores[component] = best

    return [c for c, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


# ---------------------------------------------------------------------------
# 3. Topology only
# ---------------------------------------------------------------------------

def topology(
    matrix, baseline_window, fault_window, spec: DomainSpec, step: float = 1.0
) -> list[str]:
    """Blame whatever is furthest upstream, ignoring the telemetry entirely.

    Ranked by how many other eligible components each can reach, so the most
    upstream mechanism comes first. Constant for a given domain — that is the
    point: it is the score you get for knowing the architecture and nothing
    about the incident.
    """
    eligible = _eligible(spec)
    reach = {
        c: sum(
            1 for other in eligible
            if other != c and has_path(spec.component_graph, c, other)
        )
        for c in eligible
    }
    return [c for c, _ in sorted(reach.items(), key=lambda kv: (-kv[1], kv[0]))]


def _llm(matrix, baseline_window, fault_window, spec, step=1.0):
    """Imported lazily: the LLM baseline pulls in pydantic and, on a cache
    miss, the anthropic SDK. The statistical baselines must stay runnable
    without either."""
    from eval.llm_baseline import llm

    return llm(matrix, baseline_window, fault_window, spec, step)


BASELINES = {
    "threshold": threshold,
    "correlation": correlation,
    "topology": topology,
    "llm": _llm,
}
