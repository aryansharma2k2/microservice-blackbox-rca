"""Fault-chain integrated pinpointing algorithm (FChain Sections II-B/C).

Implements the full RCA pipeline described in the FChain paper:

  * Section II-B — per-service, per-metric abnormal change point selection:
      Layer 1  CUSUM + Bootstrap        change-point candidates
      Layer 2  Markov prediction model  prediction-error filtering
      Layer 3  FFT burst threshold      burst-aware abnormality filter
      Layer 4  Tangent rollback         onset-time refinement
      Layer 5  Multi-metric aggregation earliest onset across metrics

  * Section II-C — integrated fault diagnosis across services:
      Layer 6  Propagation chain        sort services by onset time
      Layer 7  Root cause candidates    earliest onset + concurrency check
      Layer 8  Dependency filter        remove spurious propagation paths

Typical usage from ``eval/run_experiment.py``::

    from rca_engine import fault_chain

    ranked = fault_chain.pinpoint(
        metric_matrix,
        baseline_window=(bl_start, bl_end),
        fault_window=(ft_start, ft_end),
        step_seconds=1.0,
    )

``ranked`` is a list of dicts ordered by FChain Section II-C ranking:
root causes first (earliest onset), then downstream propagation victims.
Each entry contains ``service``, ``onset_time``, ``confidence``,
``abnormal_metrics``, and ``rank``.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from calibration.propagation_map import PropagationMap

import numpy as np

from rca_engine.change_point import ChangePointResult, run_layer1
from rca_engine.dependency import has_path
from rca_engine.domains import DEFAULT_DOMAIN, DomainSpec
from rca_engine.logger import log_stage
from rca_engine.normal_model import NormalModel
from rca_engine.markov_checkpoint import select_checkpoint
from rca_engine.predictability_filter import filter_abnormal_change_points
from rca_engine.smoothing import smooth_series
from rca_engine.tangent_rollback import rollback_onset
from rca_engine.aggregation import MONITORED_METRICS

# PropagationMap is imported lazily inside pinpoint() — only when a
# propagation_map_path is supplied — to keep rca_engine free of a hard
# dependency on the calibration/ package.
 
logger = logging.getLogger(__name__)
 
_DEFAULT_CHECKPOINT_ROOT = Path("checkpoints/markov")
_logged_model_selections: set[tuple[str, str]] = set()
 

logger = logging.getLogger(__name__)

# Number of metric types tracked per service in the Online Boutique domain.
# Retained for backwards compatibility; the denominator now comes from
# DomainSpec.metric_count(component) so domains whose components carry
# different numbers of signals score correctly.
TOTAL_METRICS: int = len(MONITORED_METRICS)

# Default Prometheus scrape step used throughout the pipeline.
STEP_SECONDS: float = 1.0


# ---------------------------------------------------------------------------
# Verdicts — how to read the ranked list
# ---------------------------------------------------------------------------
#
# The ranked list alone is ambiguous: "arrival_load ranked first" could mean
# the system is overloaded, or that a load metric happened to twitch first.
# The verdict says which reading applies, and it changes the remediation.

#: Nothing crossed a change-point threshold.
VERDICT_NO_ANOMALY = "no_anomaly"

#: An internal component moved first — something in the system is misbehaving.
#: This is the actionable case: `ranked` names what to fix.
VERDICT_PATHOLOGY = "pathology"

#: An exogenous input (arrival rate, prompt shape) moved before anything
#: internal did.  The system is being asked for more than it can deliver;
#: nothing inside it is broken.  Remediation is capacity or admission control,
#: not a bug fix.
VERDICT_CAPACITY = "capacity"

#: Every monitored component went abnormal together with one uniform trend.
#: Classic FChain external cause — a shared-infrastructure event or a workload
#: shift that no single component explains.
VERDICT_EXTERNAL = "external"


@dataclass
class RcaReport:
    """Structured result of one diagnosis.

    ``pinpoint()`` returns just ``ranked`` for backwards compatibility; this
    is the full object, and the evidence bundle the explainer stage consumes.
    """

    verdict: str
    ranked: list[dict[str, Any]]
    #: Abnormal exogenous components ordered by onset — the drivers behind a
    #: ``capacity`` verdict. Empty for domains that declare no exogenous nodes.
    exogenous_drivers: list[str]
    domain: str

    def top(self) -> dict[str, Any] | None:
        """Highest-ranked component that could actually *be* the cause.

        Entries carry ``eligible=False`` when the domain bars them from
        root-cause candidacy — exogenous inputs, the SLI node, co-symptoms.
        They stay in ``ranked`` as evidence, but returning one here would
        answer "why is TTFT slow?" with "because TTFT is slow".

        None when nothing eligible was abnormal.
        """
        return next((e for e in self.ranked if e.get("eligible", True)), None)


def _classify(
    sorted_components: list[str],
    exogenous: frozenset[str],
    pinpointed: list[str],
    excluded: frozenset[str] = frozenset(),
    trends: dict[str, str] | None = None,
) -> str:
    """Decide how the ranked list should be read."""
    if not sorted_components:
        return VERDICT_NO_ANOMALY

    # Nothing that could *be* a cause moved.
    if excluded and all(c in excluded for c in sorted_components):
        # Inputs shifted and the system absorbed it: a workload change, not a
        # defect. With nothing internal degraded there is nothing to fix.
        if exogenous and any(c in exogenous for c in sorted_components):
            return VERDICT_CAPACITY
        # Only co-symptoms moved (the SLI, a leaf) with no eligible cause and
        # no input to blame — there is no anomaly the graph can explain.
        return VERDICT_NO_ANOMALY

    first = sorted_components[0]
    if exogenous and first in exogenous:
        # Capacity means "asked for more than can be served", so it requires
        # the input to have gone *up*. Arrival rate falling is the opposite:
        # the server slowed down and the client could not push as much
        # through — an effect of the fault, not its cause.
        if trends is None or trends.get(first) != "down":
            return VERDICT_CAPACITY

    if not pinpointed:
        return VERDICT_EXTERNAL
    return VERDICT_PATHOLOGY


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def pinpoint(
    metric_matrix: dict[str, dict[str, np.ndarray]],
    baseline_window: tuple[float, float],
    fault_window: tuple[float, float],
    step_seconds: float = 1.0,
    propagation_map_path: str | None = None,
    start_time: float | None = None,
    logs: list[dict] | None = None,
    domain: DomainSpec | None = None,
) -> list[dict[str, Any]]:
    """Run the full FChain RCA pipeline and return a ranked suspect list.

    Thin wrapper over :func:`pinpoint_report` that returns only the ranking.
    Use ``pinpoint_report`` when you also need the verdict — for a domain with
    exogenous nodes the ranking alone is ambiguous.
    """
    return pinpoint_report(
        metric_matrix=metric_matrix,
        baseline_window=baseline_window,
        fault_window=fault_window,
        step_seconds=step_seconds,
        propagation_map_path=propagation_map_path,
        start_time=start_time,
        logs=logs,
        domain=domain,
    ).ranked


def pinpoint_report(
    metric_matrix: dict[str, dict[str, np.ndarray]],
    baseline_window: tuple[float, float],
    fault_window: tuple[float, float],
    step_seconds: float = 1.0,
    propagation_map_path: str | None = None,
    start_time: float | None = None,
    logs: list[dict] | None = None,
    domain: DomainSpec | None = None,
) -> RcaReport:
    """Run the full FChain RCA pipeline and return a structured report.

    For each service in *metric_matrix*, every metric is analysed through
    Layers 1-5 (Section II-B) to produce a rollback-refined onset timestamp.
    The per-service onset times are then passed to Layers 6-8 (Section II-C)
    which build the propagation chain, identify root causes, and filter out
    services whose abnormality is explained by downstream propagation.

    Parameters
    ----------
    metric_matrix :
        ``{service: {metric: np.ndarray}}`` covering the full time range
        ``[baseline_start, fault_end]`` sampled at *step_seconds* intervals.
        Arrays must be aligned: index 0 corresponds to *baseline_window[0]*.
    baseline_window :
        ``(start_posix, end_posix)`` of the clean baseline period used to
        learn normal behaviour for each metric.
    fault_window :
        ``(start_posix, end_posix)`` of the fault period to analyse.
        The look-back window for change-point detection is drawn from here.
    step_seconds :
        Prometheus scrape step in seconds.  Must match the resolution at
        which *metric_matrix* was built.  Default 1.0.
    propagation_map_path :
        Optional path to a ``calibration/propagation_delays.json`` file
        produced by ``calibration/calibrate.py``.  When supplied, Layer 7
        uses per-edge calibrated thresholds instead of the global 2.0 s
        constant.  If the file does not exist the parameter is silently
        ignored and the global threshold is used.
    start_time :
        POSIX timestamp when RCA pipeline started. If None, initialized to current time.
    logs :
        List to accumulate timing logs. If None, initialized to empty list.
    domain :
        Which system is being diagnosed — supplies the component graph used
        by Layers 7-8, the Layer 7 concurrency threshold, and the per-component
        metric count used as the fallback confidence denominator.  Defaults to
        Online Boutique, so callers written before the domain layer existed
        behave identically.

    Returns
    -------
    RcaReport
        ``verdict``           one of the ``VERDICT_*`` constants
        ``ranked``            one entry per abnormal component, ordered by
                              FChain Section II-C ranking: root causes first
                              (by onset time), then propagation victims
        ``exogenous_drivers`` abnormal exogenous components, by onset
        ``domain``            name of the domain that was analysed

        Each ``ranked`` entry contains:

        ``service``          component name
        ``onset_time``       POSIX timestamp of the earliest detected onset
        ``confidence``       mean per-metric confidence, else the fraction of
                             the component's metrics that were abnormal
        ``abnormal_metrics`` sorted list of metrics that showed abnormality
        ``rank``             1-based position in the output list
    """
    # Initialize timing if not provided
    if start_time is None:
        start_time = time.time()
    if logs is None:
        logs = []
    spec = domain or DEFAULT_DOMAIN
    empty = RcaReport(
        verdict=VERDICT_NO_ANOMALY,
        ranked=[],
        exogenous_drivers=[],
        domain=spec.name,
    )

    # Log the START_PINPOINT stage
    stage_start = time.time()
    log_stage("START_PINPOINT", __file__, start_time, stage_start, logs)

    # Pretrained Markov checkpoints exist only for the Boutique services, so
    # every other domain logs a fallback warning per metric per run. That is
    # expected, not a problem, and at evaluation scale it buries everything
    # else.
    if spec.name != "boutique":
        logging.getLogger("rca_engine.markov_checkpoint").setLevel(logging.ERROR)

    if not metric_matrix:
        return empty

    bl_start, bl_end = baseline_window
    ft_start, ft_end = fault_window

    logger.info(
        "FChain RCA: starting pinpoint on %d services, "
        "baseline_window=[%.0f, %.0f], fault_window=[%.0f, %.0f]",
        len(metric_matrix), bl_start, bl_end, ft_start, ft_end,
    )
    full_start = bl_start  # metric arrays are aligned to baseline start

    # Total services being monitored — used by the external cause check in
    # Layer 6 to decide whether all services became abnormal simultaneously.
    n_monitored_services = len(metric_matrix)

    # ------------------------------------------------------------------
    # Layers 1-5: per-service, per-metric change-point analysis
    # ------------------------------------------------------------------
    # For each service we collect:
    #   service_onsets            — earliest rollback-refined onset (POSIX)
    #   service_trends            — overall metric trend ('up'/'down'/'mixed')
    #   service_abnormal_metrics  — metrics that showed abnormal behaviour
    #   service_metric_confidences— per-metric maximum confidence score
    service_onsets: dict[str, float] = {}
    service_trends: dict[str, str] = {}
    service_abnormal_metrics: dict[str, list[str]] = {}
    service_metric_confidences: dict[str, dict[str, float]] = {}

    for service, metrics in metric_matrix.items():
        earliest_onset: float | None = None
        all_directions: list[str] = []
        abnormal_metric_names: list[str] = []
        metric_confs: dict[str, float] = {}

        for metric_name, full_series in metrics.items():
            baseline_data, fault_data = _split_series(
                full_series, full_start, baseline_window, fault_window,
                step=step_seconds,
            )
            if len(fault_data) < 3 or len(baseline_data) < 2:
                continue

            # Apply EMA smoothing to reduce noise before change-point detection
            smoothed_baseline = smooth_series(baseline_data, method="ema", alpha=0.3)
            smoothed_fault = smooth_series(fault_data, method="ema", alpha=0.3)
            logger.debug(
                "Applied EMA smoothing to %s.%s (baseline: %d samples, fault: %d samples)",
                service, metric_name, len(smoothed_baseline), len(smoothed_fault)
            )

            metric_analysis = _analyze_metric(
                baseline_data=smoothed_baseline,
                fault_data=smoothed_fault,
                service=service,
                metric_name=metric_name,
                step_seconds=step_seconds,
                start_time=start_time,
                logs=logs,
                min_effect_size=spec.min_effect_size,
                onset_estimator=spec.onset_estimator,
            )
            if metric_analysis is None:
                continue

            onset_indices, directions, confidences = metric_analysis
            abnormal_metric_names.append(metric_name)
            all_directions.extend(directions)
            if confidences:
                metric_confs[metric_name] = max(confidences)

            # Convert the earliest onset index into a POSIX timestamp and
            # track the minimum across all metrics for this service (Layer 5).
            #
            # Subtract the metric's own measurement lag first. A value computed
            # over a rate or quantile window reports a change roughly half a
            # window late, while a gauge reports it immediately. Layers 6-8
            # rank purely on onset order, so leaving that uncorrected means a
            # cause measured over a 30s window loses to an effect measured by
            # a gauge — which is exactly how a load spike came to read as an
            # internal pathology, and a prefix-cache collapse got blamed on
            # the cache pressure it caused.
            query = spec.metrics.get(metric_name)
            lag = (
                query.lag_seconds
                if (spec.compensate_lag and query is not None)
                else 0.0
            )
            for idx in onset_indices:
                ts = ft_start + idx * step_seconds - lag
                if earliest_onset is None or ts < earliest_onset:
                    earliest_onset = ts

        if earliest_onset is not None:
            service_onsets[service] = earliest_onset
            service_trends[service] = _determine_trend(all_directions)
            service_abnormal_metrics[service] = abnormal_metric_names
            service_metric_confidences[service] = metric_confs

    if not service_onsets:
        return empty

    # ------------------------------------------------------------------
    # Layers 6-8: integrated fault diagnosis (FChain Section II-C)
    # ------------------------------------------------------------------
    dep_graph = spec.graph()
    excluded = spec.excluded_from_root_cause()

    # Load per-edge propagation map if a path was provided and the file exists.
    prop_map = None
    if propagation_map_path is not None:
        from pathlib import Path as _Path
        _map_file = _Path(propagation_map_path)
        if _map_file.exists():
            from calibration.propagation_map import PropagationMap
            prop_map = PropagationMap.load(_map_file)
            logger.info("Loaded propagation map from %s (%d edges)", _map_file, len(prop_map.edge_keys()))
        else:
            logger.warning("propagation_map_path %s not found — using global threshold", propagation_map_path)

    pinpointed = pinpoint_faults(
        service_onsets,
        service_trends,
        dep_graph,
        n_monitored_services=n_monitored_services,
        concurrency_threshold_s=spec.concurrency_threshold_s,
        propagation_map=prop_map,
        excluded=excluded,
    )

    pinpointed_set = set(pinpointed)

    # ------------------------------------------------------------------
    # Rank and format output
    # ------------------------------------------------------------------
    # Root causes are ordered before propagation victims. Within each group,
    # services are sorted by onset time (earliest first), then by confidence
    # (highest first) when onset times tie.
    def _confidence_for(svc: str) -> float:
        return _confidence(
            service_abnormal_metrics.get(svc, []),
            service_metric_confidences.get(svc, {}),
            spec.metric_count(svc),
        )

    def _sort_key(svc: str) -> tuple:
        return (
            0 if svc in pinpointed_set else 1,
            service_onsets[svc],
            -_confidence_for(svc),
        )

    sorted_services = sorted(service_onsets, key=_sort_key)
    ranked_results: list[dict[str, Any]] = []

    for i, svc in enumerate(sorted_services, 1):
        entry = _make_entry(
            svc,
            service_onsets[svc],
            service_abnormal_metrics[svc],
            metric_confidences=service_metric_confidences.get(svc, {}),
            total_metrics=spec.metric_count(svc),
        )
        entry["rank"] = i
        entry["eligible"] = svc not in excluded
        ranked_results.append(entry)

    # Sorted purely by onset, ignoring the pinpointed/victim grouping, so the
    # verdict reflects what genuinely moved first.
    by_onset = sorted(service_onsets, key=lambda s: service_onsets[s])
    verdict = _classify(
        by_onset, spec.exogenous, pinpointed, excluded, service_trends
    )
    exogenous_drivers = [svc for svc in by_onset if svc in spec.exogenous]

    if ranked_results:
        logger.info(
            "FChain RCA: verdict=%s, ranked %d components, top=%s confidence=%.3f",
            verdict,
            len(ranked_results),
            ranked_results[0].get("service", ""),
            ranked_results[0].get("confidence", 0.0),
        )
    if verdict == VERDICT_CAPACITY:
        logger.info(
            "Capacity verdict — exogenous input %s moved before any internal "
            "component; the system is being asked for more than it can serve",
            exogenous_drivers[0] if exogenous_drivers else "?",
        )

    # Log the FINAL_RANKING stage
    stage_start = time.time()
    log_stage("FINAL_RANKING", __file__, start_time, stage_start, logs)

    return RcaReport(
        verdict=verdict,
        ranked=ranked_results,
        exogenous_drivers=exogenous_drivers,
        domain=spec.name,
    )


# ---------------------------------------------------------------------------
# Pinpointing logic (FChain Section II-C)
# ---------------------------------------------------------------------------

def pinpoint_faults(
    service_onsets: dict[str, float],
    service_trends: dict[str, str],
    dependency_graph: dict[str, list[str]],
    n_monitored_services: int = 0,
    concurrency_threshold_s: float = 2.0,
    propagation_map: "PropagationMap | None" = None,
    excluded: frozenset[str] = frozenset(),
) -> list[str]:
    """Identify faulty services using the FChain Section II-C algorithm.

    Runs three sequential layers:

    Layer 6 — Build propagation chain
        Sort all abnormal services by their rollback-refined onset time.
        If every monitored service is abnormal and all share the same
        metric trend direction (all rising or all falling), the anomaly
        is attributed to an external cause (e.g. workload spike or shared
        infrastructure failure) and an empty list is returned — no
        individual service is pinpointed.

    Layer 7 — Identify root cause candidates
        The service with the earliest onset is the primary root cause.
        Any subsequent service whose onset falls within
        *concurrency_threshold_s* of the primary is treated as a
        concurrent independent fault and is also pinpointed.  The first
        service that started too late to be concurrent breaks the scan —
        all later services are assumed to be downstream victims.

    Layer 8 — Filter spurious propagation paths
        For every remaining abnormal service ``C``, check whether its
        abnormality can be explained by propagation from an already-
        pinpointed root cause ``R``.  Two propagation directions are
        considered:

        * Forward (``R → C``): the fault propagated downstream from R to C
          along a normal dependency edge.
        * Back-pressure (``C → R``): a faulty C caused its upstream
          caller R to stall, making R appear abnormal even though it is
          not the root cause.

        If neither path exists, C cannot be a propagation victim and must
        have an independent fault; it is added to the pinpointed list.

    Parameters
    ----------
    service_onsets :
        Rollback-refined onset timestamps (POSIX seconds) keyed by service.
        Only services that showed abnormal behaviour should be present.
    service_trends :
        Per-service overall metric trend direction: ``'up'``, ``'down'``,
        or ``'mixed'``.  Used for the external cause check in Layer 6.
    dependency_graph :
        Directed adjacency list ``{service: [services it calls]}``.
    n_monitored_services :
        Total number of services under observation in this experiment run.
        Pass ``0`` to skip the external cause check.
    concurrency_threshold_s :
        Fallback onset-time gap (seconds) used when no propagation map is
        available or an edge has no calibration data.  FChain default: 2.0.
    propagation_map :
        Optional per-edge delay map from ``calibration/propagation_map.py``.
        When provided, Layer 7 uses calibrated per-edge thresholds instead
        of the global *concurrency_threshold_s*.

        Semantics with a map:
          onset_diff <= edge_threshold  → within the propagation window
                                          → victim (not pinpointed here)
          onset_diff >  edge_threshold  → outside window, too slow for
                                          propagation → concurrent fault

        When the map is None the original FChain behaviour is preserved
        exactly (flat global threshold + early break).
    excluded :
        Components that may go abnormal but can never *be* the root cause —
        exogenous inputs (arrival rate, prompt shape) and the SLI node being
        explained.  See ``DomainSpec.excluded_from_root_cause``.

        Supplying a non-empty set also disables the uniform-trend external
        cause check below.  That check assumes every component moving
        together means "not one component's fault", which is right for
        microservices but wrong for a mechanism graph, where a load spike
        legitimately drives most nodes up at once.  Declaring exogenous nodes
        expresses the same idea far more precisely, so it takes over.

    Returns
    -------
    list[str]
        Services identified as root causes, ordered by onset time.
        Returns ``[]`` if an external cause is detected or no services
        are abnormal.
    """
    if not service_onsets:
        return []

    # Layer 6 — build propagation chain by sorting on onset time
    sorted_svcs = sorted(service_onsets, key=lambda s: service_onsets[s])

    # External cause check: every monitored service is abnormal AND all
    # metric changes follow the same directional trend.  A uniform upward
    # trend suggests a workload spike; uniform downward suggests a shared
    # resource collapse (e.g. NFS).  In either case no single service is
    # at fault, so we return early.
    #
    # Skipped when the domain declares exclusions — see the `excluded` docs.
    if n_monitored_services > 0 and not excluded:
        unique_trends = set(service_trends.values()) - {"mixed"}
        all_services_abnormal = len(sorted_svcs) >= n_monitored_services
        if len(unique_trends) == 1 and all_services_abnormal:
            logger.info(
                "External cause detected — all %d services abnormal "
                "with uniform trend '%s'",
                len(sorted_svcs),
                next(iter(unique_trends)),
            )
            return []

    # Root-cause candidates exclude exogenous inputs and the SLI node.  They
    # keep their place in the ranked evidence; they just cannot win.
    candidates = [svc for svc in sorted_svcs if svc not in excluded]
    if not candidates:
        logger.info(
            "Every abnormal component is exogenous or the SLI itself — "
            "no internal root cause to pinpoint"
        )
        return []

    # Layer 7 — primary root cause is the earliest-onset service.
    # Any service that started within the concurrency window is also
    # pinpointed as an independent concurrent fault rather than a victim.
    pinpointed: list[str] = [candidates[0]]
    first_onset = service_onsets[candidates[0]]

    for svc in candidates[1:]:
        onset_diff = service_onsets[svc] - first_onset

        if propagation_map is not None:
            # Edge-aware path: check whether any already-pinpointed root cause
            # can explain this service's anomaly via propagation.
            #
            #   onset_diff <= edge_threshold  → within propagation window
            #                                   → skip (victim, handled by Layer 8)
            #   onset_diff >  edge_threshold  → outside window, too slow
            #                                   → must be concurrent fault
            #   no dependency path            → cannot be propagation victim
            #                                   → pinpoint as independent fault
            #
            # Unlike the no-map path there is NO early break: a service beyond
            # the threshold might still be an independent fault with no
            # dependency path to any pinpointed root.
            is_propagation_victim = False
            for root in pinpointed:
                if has_path(dependency_graph, root, svc) or has_path(dependency_graph, svc, root):
                    path_thr = propagation_map.get_path_threshold(root, svc, dependency_graph)
                    if onset_diff <= path_thr:
                        is_propagation_victim = True
                        break
                    # onset_diff > path_thr: outside window → concurrent
            if not is_propagation_victim:
                pinpointed.append(svc)
        else:
            # Original FChain behaviour: flat global threshold + early break.
            if onset_diff <= concurrency_threshold_s:
                # Started close enough that propagation cannot explain the gap.
                pinpointed.append(svc)
            else:
                # This service started late enough to be a downstream victim.
                # All subsequent services started even later, so stop scanning.
                break

    # Layer 8 — for every remaining abnormal service, determine whether
    # its abnormality is reachable from any already-pinpointed root cause
    # via forward propagation or back-pressure.  If not, it is an
    # independent fault and is added to the pinpointed set.
    for svc in candidates:
        if svc in pinpointed:
            continue

        is_propagation_victim = any(
            has_path(dependency_graph, root, svc)   # forward propagation
            or has_path(dependency_graph, svc, root) # back-pressure
            for root in pinpointed
        )

        if not is_propagation_victim:
            pinpointed.append(svc)

    return pinpointed


# ---------------------------------------------------------------------------
# Per-metric analysis pipeline
# ---------------------------------------------------------------------------

#: Base seed for Layer 1's block bootstrap.
#:
#: The bootstrap is a Monte Carlo estimate of the CUSUM threshold, so leaving
#: its generator unseeded makes the entire pipeline stochastic: change points
#: sitting near the threshold flip in and out between runs. Measured over 15
#: evaluations of the same 38-run corpus, top-3 alternated between 71.0% and
#: 73.7% with no input changing. That is small, but it makes "clone the repo
#: and re-derive every number" false, and it means a CI accuracy floor can
#: fail on resampling noise rather than on a regression.
#:
#: The value itself is arbitrary and only has to stay fixed.
BOOTSTRAP_SEED = 0


def _metric_seed(service: str, metric_name: str) -> int:
    """A stable per-(service, metric) bootstrap seed.

    Derived rather than shared: one global seed would hand every metric the
    same resampling draws, correlating thresholds across the very metrics
    Layer 5 then aggregates into a single per-component onset. Hashing the
    identity keeps each metric independent while staying reproducible.
    """
    digest = hashlib.blake2b(
        f"{BOOTSTRAP_SEED}:{service}:{metric_name}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % (2**32)


def _analyze_metric(
    baseline_data: np.ndarray,
    fault_data: np.ndarray,
    num_bins: int = 100,
    cusum_k_factor: float = 0.5,
    fft_Q: int = 20,
    service: str = "",
    metric_name: str = "",
    step_seconds: float = 1.0,
    checkpoint_root: Path = _DEFAULT_CHECKPOINT_ROOT,
    force_window: int | None = None,
    start_time: float | None = None,
    logs: list[dict] | None = None,
    min_effect_size: float = 0.0,
    onset_estimator: str = "last_reset",
) -> tuple[list[int], list[str], list[float]] | None:
    """Run Layers 1-4 for a single metric of a single service.
 
    Parameters
    ----------
    baseline_data :
        Normal behavior window. Used to calibrate CUSUM and, if no
        checkpoint is found, to fit the Markov model on the fly.
    fault_data :
        Fault window (look-back window) being analyzed.
    num_bins :
        Number of discretisation bins for the Markov model.
    cusum_k_factor :
        CUSUM reference value k.
    fft_Q :
        Half-window size in samples for the FFT burst threshold.
    service, metric_name :
        Used to locate the pretrained checkpoint on disk.
    step_seconds :
        Prometheus scrape step. Used to convert sample count to seconds
        when selecting the appropriate checkpoint window.
    checkpoint_root :
        Root directory for checkpoint storage.
    force_window :
        If provided, always use this checkpoint window (seconds) regardless
        of how much baseline data is available.  Raises FileNotFoundError
        if the checkpoint does not exist.  Set to None to use automatic
        selection or the active_window_seconds from checkpoint_config.json.
 
    Returns
    -------
    tuple[list[int], list[str], list[float]] | None
        (onset_indices, directions, confidences) where indices are into
        fault_data. Returns None if no abnormal change points are found.
    """
 
    # ------------------------------------------------------------------
    # Layer 1: CUSUM + Bootstrap
    # ------------------------------------------------------------------
    result = run_layer1(
        time_series=fault_data,
        baseline_data=baseline_data,
        k=cusum_k_factor,
        seed=_metric_seed(service, metric_name),
        start_time=start_time,
        logs=logs,
        onset_estimator=onset_estimator,
    )
    if not result.change_points:
        return None

    # Effect-size gate. Drop change points that are detectable but too small
    # to matter, before spending Layers 2-4 on them.
    if min_effect_size > 0:
        kept = [
            cp for cp in result.change_points
            if effect_size(baseline_data, fault_data, cp) >= min_effect_size
        ]
        if not kept:
            logger.debug(
                "%s.%s: %d change point(s) all below the %.1f-sigma effect "
                "floor — treating as noise",
                service, metric_name, len(result.change_points), min_effect_size,
            )
            return None
        if len(kept) < len(result.change_points):
            dropped = set(result.change_points) - set(kept)
            result = ChangePointResult(
                **{
                    **result.__dict__,
                    "change_points": kept,
                    "directions": [
                        d for cp, d in zip(result.change_points, result.directions)
                        if cp not in dropped
                    ],
                }
            )

    g = result.cusum_combined
    h = result.bootstrap_threshold
 
    # ------------------------------------------------------------------
    # Layer 2: Markov model — load checkpoint or fit from baseline
    # ------------------------------------------------------------------
    model = _build_model(
        baseline_data   = baseline_data,
        fault_data      = fault_data,
        num_bins        = num_bins,
        service         = service,
        metric_name     = metric_name,
        step_seconds    = step_seconds,
        checkpoint_root = checkpoint_root,
        force_window    = force_window,
    )
 
    # For each onset estimate from Layer 1, find the corresponding CUSUM
    # threshold crossing — the first index at or after the onset where
    # the CUSUM score exceeds the bootstrap threshold. The prediction
    # error is evaluated at the crossing rather than the onset because
    # at the crossing the metric has deviated far enough from the baseline
    # to produce a strong error signal for both step changes and gradual
    # drifts. At the onset of a slow drift the deviation is still small
    # and the error would be indistinguishable from normal noise.
    onset_to_crossing: dict[int, int] = {}
    for onset in result.change_points:
        candidates = np.where(g[onset:] >= h)[0]
        if len(candidates) > 0:
            onset_to_crossing[onset] = onset + int(candidates[0])
        else:
            onset_to_crossing[onset] = onset
 
    crossing_indices = list(onset_to_crossing.values())
    pred_errors_at_crossing = model.prediction_errors_for(
        crossing_indices, fault_data,
    )
 
    # Remap errors to be keyed by onset index. Layer 3 centres its FFT
    # window on the dict key, so keying by onset places the window in
    # the still-calm pre-fault region, which gives a lower burstiness
    # threshold and makes detection more sensitive.
    pred_errors_by_onset: dict[int, float] = {
        onset: pred_errors_at_crossing.get(crossing, 0.0)
        for onset, crossing in onset_to_crossing.items()
    }
 
    # ------------------------------------------------------------------
    # Layer 3: FFT burst filter
    # ------------------------------------------------------------------
    abnormal_cps: list[int] = filter_abnormal_change_points(
        fault_data, pred_errors_by_onset, Q=fft_Q,
        start_time=start_time, logs=logs,
    )
    if not abnormal_cps:
        return None
 
    cp_to_direction:  dict[int, str]   = dict(
        zip(result.change_points, result.directions)
    )
    cp_to_confidence: dict[int, float] = result.confidence_scores
 
    # ------------------------------------------------------------------
    # Layer 4: Tangent-based rollback
    # ------------------------------------------------------------------
    onset_indices: list[int]   = []
    directions:    list[str]   = []
    confidences:   list[float] = []
 
    for cp in abnormal_cps:
        refined_onset = rollback_onset(
            series=fault_data,
            abnormal_cp=cp,
            all_change_points=result.change_points,
            start_time=start_time,
            logs=logs,
        )
        onset_indices.append(refined_onset)
        directions.append(cp_to_direction.get(cp, "up"))
        confidences.append(cp_to_confidence.get(cp, 0.0))
 
    if not onset_indices:
        return None
 
    return onset_indices, directions, confidences
 
 
# ---------------------------------------------------------------------------
# Model construction — checkpoint or fallback
# ---------------------------------------------------------------------------
 
def _build_model(
    baseline_data: np.ndarray,
    fault_data: np.ndarray,
    num_bins: int,
    service: str,
    metric_name: str,
    step_seconds: float,
    checkpoint_root: Path,
    force_window: int | None = None,
) -> NormalModel:
    """Return a ready-to-use NormalModel.
 
    Tries to load a pretrained checkpoint first. Falls back to fitting
    on the provided baseline data if no checkpoint is available.
    """
    if service and metric_name:
        available_seconds = len(baseline_data) * step_seconds
        checkpoint = select_checkpoint(
            service           = service,
            metric_name       = metric_name,
            available_seconds = available_seconds,
            root              = checkpoint_root,
            force_window      = force_window,
        )
        if checkpoint is not None:
            key = (service, metric_name)
            # print(
            #     "[RCA][model] "
            #     f"{service}/{metric_name}: using pretrained checkpoint "
            #     f"(window={checkpoint.window_seconds}s)"
            # )
            if key not in _logged_model_selections:
                logger.info(
                    "[rca:model] %s/%s: checkpoint window=%ds (pretrained)",
                    service,
                    metric_name,
                    checkpoint.window_seconds,
                )
                _logged_model_selections.add(key)
            logger.debug(
                "Using pretrained checkpoint for %s/%s (window=%ds)",
                service, metric_name, checkpoint.window_seconds,
            )
            return NormalModel.from_checkpoint(checkpoint)
 
        logger.debug(
            "No checkpoint found for %s/%s — fitting from baseline.",
            service, metric_name,
        )
        key = (service, metric_name)
        # print(
        #     "[RCA][model] "
        #     f"{service}/{metric_name}: no checkpoint selected, "
        #     f"training new model from baseline (available={available_seconds:.0f}s)"
        # )
        if key not in _logged_model_selections:
            logger.info(
                "[rca:model] %s/%s: fallback baseline fit (available=%.0fs)",
                service,
                metric_name,
                available_seconds,
            )
            _logged_model_selections.add(key)
 
    # Fallback: fit from the baseline data provided at call time.
    # Bin range from combined data so fault excursions are covered.
    all_data = np.concatenate([baseline_data, fault_data])
    lo = float(np.min(all_data))
    hi = float(np.max(all_data))
    if hi == lo:
        hi = lo + 1e-6
 
    return NormalModel(
        num_bins   = num_bins,
        metric_min = lo,
        metric_max = hi,
    ).fit(baseline_data)
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_series(
    series: np.ndarray,
    full_start: float,
    baseline_window: tuple[float, float],
    fault_window: tuple[float, float],
    step: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice a full-range time series into baseline and fault sub-arrays.

    Parameters
    ----------
    series :
        Full metric array aligned to *full_start* at *step*-second intervals.
    full_start :
        POSIX timestamp corresponding to ``series[0]``.
    baseline_window :
        ``(start, end)`` POSIX timestamps for the baseline slice.
    fault_window :
        ``(start, end)`` POSIX timestamps for the fault slice.
    step :
        Sample interval in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(baseline_data, fault_data)`` as contiguous views of *series*.
    """
    bl_start_idx = max(0, round((baseline_window[0] - full_start) / step))
    bl_end_idx   = round((baseline_window[1] - full_start) / step)
    ft_start_idx = round((fault_window[0] - full_start) / step)
    ft_end_idx   = min(len(series), round((fault_window[1] - full_start) / step))

    baseline = series[bl_start_idx:bl_end_idx]
    fault    = series[ft_start_idx:ft_end_idx]
    return baseline, fault


def _determine_trend(directions: list[str]) -> str:
    """Collapse a list of per-change-point directions into a single trend.

    Returns ``'up'`` if all directions are upward, ``'down'`` if all are
    downward, and ``'mixed'`` otherwise (including the empty list).
    """
    if not directions:
        return "mixed"
    ups   = directions.count("up")
    downs = directions.count("down")
    if ups > 0 and downs == 0:
        return "up"
    if downs > 0 and ups == 0:
        return "down"
    return "mixed"


#: Floors for the effect-size denominator, as a fraction of the baseline mean
#: and as an absolute value. Without them a series that is exactly constant in
#: the baseline (std == 0) makes any later movement look infinitely large.
_EFFECT_REL_FLOOR = 0.01
_EFFECT_ABS_FLOOR = 1e-9


def effect_size(
    baseline_data: np.ndarray,
    fault_data: np.ndarray,
    change_point: int,
) -> float:
    """How big is the shift at *change_point*, in baseline sigmas?

    Compares the post-change level against the baseline level, scaled by
    baseline variability. Answers "is this shift meaningful?" — a separate
    question from Layer 1's "is this shift detectable?", and the one that
    decides whether an operator would care.

    Returns 0.0 when there is not enough data on either side to judge.
    """
    after = fault_data[change_point:]
    if after.size == 0 or baseline_data.size < 2:
        return 0.0

    baseline_mean = float(np.mean(baseline_data))
    scale = max(
        float(np.std(baseline_data)),
        _EFFECT_REL_FLOOR * abs(baseline_mean),
        _EFFECT_ABS_FLOOR,
    )
    return abs(float(np.mean(after)) - baseline_mean) / scale


def _confidence(
    abnormal_metrics: list[str],
    metric_confidences: dict[str, float] | None,
    total_metrics: int,
) -> float:
    """Score how strongly one component's evidence supports it being abnormal.

    Prefers the mean of the per-metric bootstrap confidence scores produced by
    Layer 1, which is the better-calibrated signal.  Falls back to the fraction
    of the component's monitored metrics that went abnormal when Layer 1
    returned no scores.

    Shared by the ranking sort key and the emitted entry so the two can never
    disagree about what a component's confidence is.
    """
    if metric_confidences and abnormal_metrics:
        return (
            sum(metric_confidences.get(m, 0.0) for m in abnormal_metrics)
            / len(abnormal_metrics)
        )
    if total_metrics <= 0:
        return 0.0
    # Fallback: treat each abnormal metric as equally weighted evidence.
    return len(abnormal_metrics) / total_metrics


def _make_entry(
    service: str,
    onset_time: float,
    abnormal_metrics: list[str],
    metric_confidences: dict[str, float] | None = None,
    total_metrics: int = TOTAL_METRICS,
) -> dict[str, Any]:
    """Construct a ranked result dict for one service."""
    confidence = _confidence(abnormal_metrics, metric_confidences, total_metrics)

    return {
        "service":          service,
        "onset_time":       onset_time,
        "confidence":       round(confidence, 3),
        "abnormal_metrics": sorted(abnormal_metrics),
    }