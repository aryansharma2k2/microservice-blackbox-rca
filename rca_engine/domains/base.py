"""Domain specification — what the RCA engine needs to know about a system.

The FChain pipeline (Layers 1-8) is domain-agnostic: it localizes a fault
among *components* connected by a directed graph, given one or more metric
time series per component.  Nothing in Layers 1-8 assumes those components
are microservices.

A :class:`DomainSpec` supplies the three things that *are* domain-specific:

1. **What to query** — PromQL per metric name (``metrics``).
2. **How series map to components** — a pod label for microservices, a fixed
   node for a single-process engine (``MetricQuery.component`` /
   ``component_from_labels``).
3. **How components relate** — the directed graph Layers 7-8 filter against
   (``component_graph``).

Two domains ship with the engine:

* ``boutique`` — 11 Online Boutique microservices over cAdvisor metrics.
  Components are services; edges are RPC calls.
* ``vllm`` — subsystems *inside* one vLLM inference server.  Components are
  scheduler mechanisms (KV cache pressure, preemption, queueing, …); edges
  are causal ("A can cause B").

The ``exogenous`` / ``sli_node`` fields exist for the second case.  In a
mechanism graph the input nodes (arrival rate, prompt shape) almost always
move first, so ranking purely by earliest onset would blame "load" for every
incident.  Marking those nodes exogenous lets Layer 6 separate *capacity*
("you are overloaded") from *pathology* ("something inside is misbehaving").
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricQuery:
    """One PromQL expression and how to attribute its result series.

    Parameters
    ----------
    promql:
        The expression to evaluate via ``/api/v1/query_range``.
    component:
        Fixed component this query's series belong to.  Use for domains where
        a metric names its own subsystem (vLLM: ``vllm:num_preemptions_total``
        is always the ``preemption`` node).  When ``None``, the owning
        :class:`DomainSpec` derives the component from the series labels via
        ``component_from_labels`` (Boutique: from the ``pod`` label).
    description:
        Free text; surfaced in evidence bundles and the writeup.
    optional:
        True when the metric comes from an exporter that may legitimately not
        be deployed — DCGM on a CPU-only box, cAdvisor outside Kubernetes.
        The drift check warns about these instead of failing, so "this
        environment has no GPU" is not reported the same way as "this metric
        name is wrong", which is the distinction that matters when deciding
        whether it is safe to capture traces.
    """

    promql: str
    component: str | None = None
    description: str = ""
    optional: bool = False
    #: Width of the rate/quantile window this metric is computed over, in
    #: seconds. ``0`` for an instantaneous gauge. See :meth:`lag_seconds`.
    window_seconds: float = 0.0

    @property
    def lag_seconds(self) -> float:
        """How long after a real change this metric reports it.

        A value computed over ``[t-W, t]`` is a moving average: at the instant
        something changes, the window still holds W seconds of the old
        behaviour, so the reported value ramps rather than steps and crosses
        any threshold roughly half a window late. A gauge has no such delay.

        This matters because the pipeline ranks causes by onset order. In the
        vLLM domain every exogenous input is a windowed rate while the
        scheduler state is instantaneous gauges, so without compensation
        *causes are systematically detected after their own effects* — a load
        spike reads as an internal pathology, and a prefix-cache collapse gets
        blamed on the cache pressure it produced.

        Half the window is the expected offset for a step change, which is
        what an injected fault is.
        """
        return self.window_seconds / 2.0


def histogram_quantile(
    metric: str,
    quantile: float,
    window: str = "1m",
    by: str = "le",
) -> str:
    """Build a PromQL quantile expression over a histogram's ``_bucket`` series.

    Boutique's metrics are all gauges and counters, but an inference engine
    reports its latencies as histograms — TTFT, queue time, and prefill time
    are only available this way, and they are the signals the whole vLLM
    domain is built to explain.

    >>> histogram_quantile("vllm:time_to_first_token_seconds", 0.99, "1m")
    'histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[1m])))'
    """
    return (
        f"histogram_quantile({quantile}, sum by ({by}) "
        f"(rate({metric}_bucket[{window}])))"
    )


@dataclass(frozen=True)
class DomainSpec:
    """Everything the pipeline needs to analyse one kind of system.

    Attributes
    ----------
    name:
        Short identifier, e.g. ``"boutique"`` or ``"vllm"``.
    metrics:
        ``{metric_name: MetricQuery}``.  Metric names are the keys of the
        inner dict in the metric matrix.
    component_graph:
        Directed adjacency ``{component: [components it points at]}``.  For
        microservices this is the call graph; for a mechanism graph it is
        "A can cause B".
    component_metrics:
        Optional ``{component: (metric_name, ...)}`` restricting which metrics
        back each component.  Leave empty when every component carries every
        metric (Boutique) — the fallback confidence denominator then falls
        back to ``len(metrics)``.
    exogenous:
        Components representing uncontrolled inputs rather than internal
        state.  Never the actionable root cause on their own.
    sli_node:
        The component being explained (vLLM: ``"ttft"``).  Symptom, not cause.
    concurrency_threshold_s:
        Layer 7 fallback onset gap when no calibrated propagation map covers
        an edge.  FChain's default is 2.0 s.
    compensate_lag:
        Subtract each metric's measurement lag from its detected onset
        before ranking. Sound in principle — windowed metrics really do
        report late, and the vLLM domain really does measure its causes
        over windows and its effects with gauges — but it measurably hurt
        accuracy on the traces available when it was written, so it is
        off until an A/B over the full corpus says otherwise.
    min_effect_size:
        Minimum standardized shift, in baseline standard deviations, before a
        detected change point counts as real.  ``0.0`` disables the gate.

        Layers 1-4 ask whether a change is *statistically detectable*, never
        whether it is *large enough to matter*.  That is fine for a cpu_hog,
        which moves a metric by multiples, but not for an inference engine
        whose histogram quantiles are computed from few samples at low request
        rates — there, ordinary noise reliably produces change points and the
        pipeline names a cause on a perfectly healthy run.
    component_from_labels:
        Maps a Prometheus series' labels to a component, for queries that
        leave ``MetricQuery.component`` unset.  Return ``None`` to drop the
        series (Boutique uses this to skip node-level series with no pod).
    """

    name: str
    metrics: Mapping[str, MetricQuery]
    component_graph: Mapping[str, list[str]]
    component_metrics: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    exogenous: frozenset[str] = frozenset()
    sli_node: str | None = None
    concurrency_threshold_s: float = 2.0
    min_effect_size: float = 0.0
    onset_estimator: str = "last_reset"
    compensate_lag: bool = False
    component_from_labels: Callable[[Mapping[str, str]], str | None] | None = None

    # -- helpers ---------------------------------------------------------

    def graph(self) -> dict[str, list[str]]:
        """Return a mutable copy of the component graph."""
        return {k: list(v) for k, v in self.component_graph.items()}

    def metric_count(self, component: str) -> int:
        """Denominator for the fallback confidence fraction.

        Uses the component's own metric count when ``component_metrics``
        declares one, else the domain-wide metric count.  Per-component is the
        right denominator whenever components carry different numbers of
        signals, which is the norm in a mechanism graph.
        """
        backing = self.component_metrics.get(component)
        return len(backing) if backing else len(self.metrics)

    def resolve_component(
        self,
        metric_name: str,
        labels: Mapping[str, str],
    ) -> str | None:
        """Attribute one Prometheus result series to a component.

        Returns ``None`` when the series should be dropped.
        """
        query = self.metrics.get(metric_name)
        if query is not None and query.component is not None:
            return query.component
        if self.component_from_labels is not None:
            return self.component_from_labels(labels)
        return None

    def is_exogenous(self, component: str) -> bool:
        return component in self.exogenous

    def excluded_from_root_cause(self) -> frozenset[str]:
        """Components that may go abnormal but can never *be* the root cause.

        Two kinds, excluded for different reasons:

        * **Exogenous inputs** — arrival rate, prompt shape.  Real, and often
          the earliest thing to move, but not something inside the system that
          broke.  "Requests got longer" is a workload change, not a defect.
        * **The SLI node** — TTFT is the symptom being explained.  It goes
          abnormal in every single incident by construction, so letting it win
          the earliest-onset race would make every diagnosis read "TTFT is
          slow because TTFT is slow".

        A third kind is derived rather than declared:

        * **Components with no causal path to the SLI.**  If a node cannot
          reach the signal being explained, it cannot explain a regression in
          it.  vLLM's ``decode_health`` (inter-token latency) is the example:
          a genuine co-symptom, useful evidence about batch behaviour, but
          naming it the cause of a *first-token* latency spike would be
          symptom-explaining-symptom.

        Only applies when ``sli_node`` is set, so domains without an explicit
        SLI (Boutique) are unaffected.  Exclusions of this third kind are
        logged, because one can equally mean a missing edge in the graph.

        All three still appear in the ranked output as evidence; they are only
        barred from being *pinpointed* as the cause.
        """
        excluded = set(self.exogenous)
        if self.sli_node is None:
            return frozenset(excluded)

        excluded.add(self.sli_node)

        # Imported here rather than at module scope to keep the graph helpers
        # out of the import path for callers that only need the dataclasses.
        from rca_engine.dependency import has_path

        unreachable = [
            node
            for node in self.component_graph
            if node not in excluded
            and not has_path(self.component_graph, node, self.sli_node)
        ]
        if unreachable:
            logger.info(
                "Domain '%s': %s cannot reach the SLI node '%s', so cannot be "
                "named a root cause. If that is wrong, the graph is missing an "
                "edge.",
                self.name,
                ", ".join(sorted(unreachable)),
                self.sli_node,
            )
            excluded.update(unreachable)

        return frozenset(excluded)

    def validate(self) -> list[str]:
        """Return a list of internal inconsistencies; empty means healthy.

        Checked at construction time by the domain modules and by the Phase 2
        metric-discovery script, which additionally diffs ``metrics`` against
        the metric surface a live server actually exposes.
        """
        problems: list[str] = []

        known = set(self.metrics)
        for component, backing in self.component_metrics.items():
            if component not in self.component_graph:
                problems.append(
                    f"component_metrics references unknown component {component!r}"
                )
            for metric in backing:
                if metric not in known:
                    problems.append(
                        f"component {component!r} references unknown metric {metric!r}"
                    )

        for component, targets in self.component_graph.items():
            for target in targets:
                if target not in self.component_graph:
                    problems.append(
                        f"edge {component!r} -> {target!r} points at an "
                        "undeclared component"
                    )

        for node in self.exogenous:
            if node not in self.component_graph:
                problems.append(f"exogenous node {node!r} is not in the graph")

        if self.sli_node is not None and self.sli_node not in self.component_graph:
            problems.append(f"sli_node {self.sli_node!r} is not in the graph")

        fixed = {q.component for q in self.metrics.values() if q.component is not None}
        for component in fixed:
            if component not in self.component_graph:
                problems.append(
                    f"a MetricQuery is pinned to undeclared component {component!r}"
                )

        if not fixed and self.component_from_labels is None:
            problems.append(
                "no MetricQuery declares a component and component_from_labels "
                "is unset — no series could ever be attributed"
            )

        return problems
