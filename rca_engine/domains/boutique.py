"""Online Boutique domain — 11 microservices over cAdvisor container metrics.

This is the original FChain target: components are Kubernetes services, edges
are gRPC calls, and every component carries the same 7 black-box system
metrics (FChain paper Section III-A).

Nothing here changed behaviourally when the domain layer was introduced — the
PromQL, the pod-name regex, and the dependency graph are the same values the
pipeline used before, relocated so a second domain can exist beside them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from rca_engine.aggregation import MONITORED_METRICS
from rca_engine.dependency import ONLINE_BOUTIQUE_DEPENDENCIES
from rca_engine.domains.base import DomainSpec, MetricQuery

NAMESPACE = "boutique"

# Regex to strip the two random suffixes appended to pod names, e.g.
# "cartservice-7d9b4f6c8-xkz9p"  to  "cartservice"
_POD_SUFFIX_RE = re.compile(r"-[a-f0-9]+-[a-z0-9]+$")


def pod_to_service(pod_name: str) -> str:
    """Strip the ReplicaSet hash and pod hash from *pod_name*."""
    return _POD_SUFFIX_RE.sub("", pod_name)


def _component_from_labels(labels: Mapping[str, str]) -> str | None:
    """Attribute a cAdvisor series to a service via its ``pod`` label.

    Returns None for node-level series that carry no pod label, which the
    collector then drops.
    """
    pod = labels.get("pod", "")
    if not pod:
        return None
    return pod_to_service(pod)


# PromQL expressions keyed by a short metric name.
#
# cAdvisor housekeeping interval governs how often these counters actually
# update, independent of how often Prometheus scrapes.  Rate/deriv windows use
# [30s]/[45s] so each evaluation window spans several real counter updates,
# giving stable estimates with enough samples for CUSUM to distinguish
# sustained changes from per-service noise.
METRICS: dict[str, MetricQuery] = {
    "cpu_rate": MetricQuery(
        promql=(
            f'rate(container_cpu_usage_seconds_total{{namespace="{NAMESPACE}",container!=""}}[30s])'
        ),
        description="CPU seconds consumed per second.",
    ),
    # Fraction of CFS scheduling periods where the pod was CPU-throttled.
    # Rises sharply when a cpu_hog fault hits a resource-limited container,
    # even when cpu_rate stays flat at its limit.
    # Note: cAdvisor emits this without a container label — it is pod-scoped.
    "cpu_throttle_ratio": MetricQuery(
        promql=(
            f'sum by (pod, namespace) (rate(container_cpu_cfs_throttled_periods_total{{namespace="{NAMESPACE}"}}[30s]))'
            " / "
            f'sum by (pod, namespace) (rate(container_cpu_cfs_periods_total{{namespace="{NAMESPACE}"}}[30s]))'
        ),
        description="Fraction of CFS periods the pod was throttled.",
    ),
    # Rate of memory growth (bytes/sec) over a 45s window.
    # Using deriv() instead of the raw gauge makes this metric stationary:
    # normal fluctuation stays near zero while a mem_leak shows a sustained
    # positive slope.  The raw gauge drifts upward over time under any load,
    # causing CUSUM to fire false change points for every non-memory fault.
    "mem_wss": MetricQuery(
        promql=(
            f'deriv(container_memory_working_set_bytes{{namespace="{NAMESPACE}",container!=""}}[45s])'
        ),
        description="Working-set memory growth rate (bytes/sec).",
    ),
    # interface="eth0" selects the pod's primary NIC only.  Without this
    # filter, cAdvisor returns one series per virtual interface (lo, eth0,
    # erspan0, gre0, tunl0, …) and the aggregation across all interfaces
    # produces meaningless totals.
    "net_rx_rate": MetricQuery(
        promql=(
            f'rate(container_network_receive_bytes_total{{namespace="{NAMESPACE}",interface="eth0"}}[30s])'
        ),
        description="Bytes received per second on eth0.",
    ),
    "net_tx_rate": MetricQuery(
        promql=(
            f'rate(container_network_transmit_bytes_total{{namespace="{NAMESPACE}",interface="eth0"}}[30s])'
        ),
        description="Bytes transmitted per second on eth0.",
    ),
    "fs_read_rate": MetricQuery(
        promql=(
            f'rate(container_fs_reads_bytes_total{{namespace="{NAMESPACE}",container!=""}}[30s])'
        ),
        description="Bytes read from disk per second.",
    ),
    "fs_write_rate": MetricQuery(
        promql=(
            f'rate(container_fs_writes_bytes_total{{namespace="{NAMESPACE}",container!=""}}[30s])'
        ),
        description="Bytes written to disk per second.",
    ),
}

BOUTIQUE = DomainSpec(
    name="boutique",
    metrics=METRICS,
    component_graph=ONLINE_BOUTIQUE_DEPENDENCIES,
    # Every service carries every metric, so the domain-wide count (7) is the
    # correct fallback confidence denominator — matching the original
    # TOTAL_METRICS constant.
    component_metrics={},
    # Microservice RCA has no notion of exogenous inputs: Layer 6's existing
    # "all services abnormal with a uniform trend" heuristic handles external
    # causes here.
    exogenous=frozenset(),
    sli_node=None,
    concurrency_threshold_s=2.0,
    component_from_labels=_component_from_labels,
)

# Fail fast at import time rather than mid-experiment.
_problems = BOUTIQUE.validate()
if _problems:  # pragma: no cover - guards against edits to the constants above
    raise ValueError(
        "boutique DomainSpec is inconsistent:\n  " + "\n  ".join(_problems)
    )

assert set(METRICS) == set(MONITORED_METRICS), (
    "boutique METRICS drifted from aggregation.MONITORED_METRICS: "
    f"{set(METRICS) ^ set(MONITORED_METRICS)}"
)
