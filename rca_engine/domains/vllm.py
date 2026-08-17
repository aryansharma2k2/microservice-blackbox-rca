"""vLLM domain — localizing a TTFT regression *inside* one inference server.

Where the Boutique domain localizes among services, this one localizes among
**mechanisms inside a single process**.  An inference server is one service,
so a naive port would leave Layers 6-8 with nothing to filter.  The vLLM
scheduler, though, has real internal causal structure that forms a DAG, and
onset-ordering plus graph filtering apply to it unchanged.

Why this is hard
----------------
The candidate causes of a p99 TTFT spike are *observationally confusable*.
They share symptoms and they cause one another — preemption is simultaneously
a consequence of KV cache pressure and a cause of latency, so any method that
ranks by correlation with TTFT will implicate both and distinguish neither.

Two facts about vLLM's telemetry make the problem tractable:

**TTFT decomposes natively.**  ``request_queue_time_seconds`` and
``request_prefill_time_seconds`` are reported separately, and TTFT is
essentially their sum.  That splits the hypothesis space before any inference
runs: queue-dominated points at admission (cache pressure, preemption,
capacity), prefill-dominated at compute (long prompts, prefix-cache misses,
GPU contention).

**Wasted prefill work is observable.**  PagedAttention largely eliminates
*external* fragmentation, so "memory fragmentation" in the classical sense is
close to a red herring here.  The sharp version is cache thrash: prompt tokens
that had to be recomputed because their blocks were gone.
``prompt_tokens_cached_total`` against ``prompt_tokens_total`` gives the
token-level reuse rate, and ``request_prefill_kv_computed_tokens`` gives the
KV actually computed per request.  Both rise when the cache stops working and
when preemption forces recomputation.

(An earlier draft used ``kv_block_reuse_gap_seconds`` and
``kv_block_idle_before_evict_seconds``, which appear in vLLM's metrics
documentation but are **not** exposed by the shipped V1 engine — the
discovery script caught it.  The token-level signals above are what a real
server actually reports, and they are arguably the better measurement anyway,
since they count wasted work rather than block lifetimes.)

Exogenous nodes
---------------
``arrival_load`` and ``request_shape`` are inputs, not defects.  They are
usually the first thing to move, so without marking them exogenous the
earliest-onset rule would blame "load" for every incident.  Marking them lets
Layer 6 return a *capacity* verdict ("you are asking for more than this
deployment can serve") instead of a bogus root cause.  ``ttft`` is likewise
excluded — it is the symptom being explained.

Status
------
Every ``vllm:`` metric below is confirmed present on a real server —
``vllm/vllm-openai-cpu:latest-arm64`` serving Qwen3-0.6B — and the captured
surface is committed at ``deploy/vllm/metric_surface.json``.  Re-check after
any version bump::

    python -m rca_engine.scripts.discover_metrics check vllm --url <server>/metrics

The ``container_*`` metrics come from cAdvisor and are absent unless the
monitoring sidecar is running; the ``DCGM_*`` metrics require the DCGM
exporter and are absent on the CPU profile.  Those two components
(``host_saturation``, ``gpu_saturation``) simply never go abnormal without
their exporters, which is expected rather than a drift.
"""

from __future__ import annotations

from rca_engine.domains.base import DomainSpec, MetricQuery, histogram_quantile

#: Selector for the server's own container, used by the metrics that catch
#: API-server/tokenizer starvation. Matches how cAdvisor labels containers
#: under docker-compose (``name="vllm-vllm-cpu-1"``); under Kubernetes swap
#: this for ``namespace="vllm",container!=""``.
#:
#: Note: cAdvisor cannot see per-container cgroups on Docker Desktop for
#: macOS — it reports only the root cgroup — so ``host_saturation`` is inert
#: on a Mac laptop and the ``host_cpu_hog`` scenario can only be validated on
#: Linux, which is where trace capture happens anyway.
CONTAINER_SELECTOR = 'name=~".*vllm.*"'

#: Rate window for histogram buckets.  Wide enough that a p99 is meaningful at
#: single-digit RPS, narrow enough that CUSUM still sees an onset promptly.
RATE_WINDOW = "30s"

#: Shorter window for counter *ratios*.
#:
#: A ratio over [t-W, t] is a moving average: it needs the whole window to
#: elapse before it fully reflects a change. A gauge reports the same change
#: instantly. With one shared 30s window that asymmetry inverts causality —
#: on a real prefix_diversity run, `kv_cache_usage` (gauge) moved within
#: ~2s while `prefix_cache_hit_rate` (30s ratio) took ~20s, so the effect
#: out-raced its own cause and the pipeline blamed the cache pressure the
#: cache failure had produced.
#:
#: Ratios need far fewer samples than quantiles do — a hit rate is stable over
#: tens of requests where a p99 is not — so they can afford the shorter window.
RATIO_WINDOW = "10s"

#: Quantile used for the latency signals.  p99 is the SLI teams alert on.
Q = 0.99


def _q(metric: str, quantile: float = Q) -> str:
    return histogram_quantile(metric, quantile, RATE_WINDOW)


def _secs(window: str) -> float:
    """'30s' -> 30.0"""
    return float(window.rstrip("s"))


METRICS: dict[str, MetricQuery] = {
    # -- exogenous inputs ------------------------------------------------
    "request_rate": MetricQuery(
        promql=f"sum(rate(vllm:request_success_total[{RATE_WINDOW}]))",
        component="arrival_load",
        description="Completed requests per second.",
    ),
    "prompt_token_rate": MetricQuery(
        promql=f"sum(rate(vllm:prompt_tokens_total[{RATE_WINDOW}]))",
        component="arrival_load",
        description="Prompt tokens ingested per second — offered prefill work.",
    ),
    "prompt_tokens_p99": MetricQuery(
        promql=_q("vllm:request_prompt_tokens"),
        component="request_shape",
        description="p99 prompt length. Rises on a long-prompt burst.",
    ),
    "prompt_tokens_p50": MetricQuery(
        promql=_q("vllm:request_prompt_tokens", 0.50),
        component="request_shape",
        description="Median prompt length. Separates a shifted "
        "distribution from a heavy tail.",
    ),
    # Output length is as much a workload property as prompt length, and it
    # is what drives KV growth over a request's lifetime. Without it, a
    # long-output burst would be invisible to the exogenous check and get
    # misreported as an internal pathology.
    # What clients *asked for*, which is the workload property. Distinct from
    # request_generation_tokens, which is what was actually produced and is
    # therefore partly an outcome of the server's own behaviour.
    "requested_max_tokens_p99": MetricQuery(
        promql=_q("vllm:request_params_max_tokens"),
        component="request_shape",
        description="p99 requested output length. Rises when clients ask for "
        "more tokens; each one holds KV blocks for longer.",
    ),
    # -- cache and memory ------------------------------------------------
    "kv_cache_usage": MetricQuery(
        promql="sum(vllm:kv_cache_usage_perc)",
        component="kv_cache_pressure",
        description="Fraction of KV cache blocks in use (0-1).",
    ),
    "prefix_cache_hit_rate": MetricQuery(
        promql=(
            f"sum(rate(vllm:prefix_cache_hits_total[{RATIO_WINDOW}]))"
            " / "
            f"clamp_min(sum(rate(vllm:prefix_cache_queries_total[{RATIO_WINDOW}])), 1)"
        ),
        component="prefix_cache_efficacy",
        description="Block-level prefix cache hit ratio. Collapses under "
        "prefix-diversity pressure, forcing full re-prefill of every request.",
    ),
    "cached_token_ratio": MetricQuery(
        promql=(
            f"sum(rate(vllm:prompt_tokens_cached_total[{RATIO_WINDOW}]))"
            " / "
            f"clamp_min(sum(rate(vllm:prompt_tokens_total[{RATIO_WINDOW}])), 1)"
        ),
        component="prefix_cache_efficacy",
        description="Fraction of prompt tokens served from cache. The "
        "token-level view of the same effect, and the one that translates "
        "directly into prefill work avoided.",
    ),
    # -- scheduler -------------------------------------------------------
    "preemption_rate": MetricQuery(
        promql=f"sum(rate(vllm:num_preemptions_total[{RATE_WINDOW}]))",
        component="preemption",
        description="Preemptions per second. Each one discards computed KV "
        "state that must be recomputed later.",
    ),
    "requests_waiting": MetricQuery(
        promql="sum(vllm:num_requests_waiting)",
        component="queueing",
        description="Requests admitted to the queue but not yet running.",
    ),
    "queue_time_p99": MetricQuery(
        promql=_q("vllm:request_queue_time_seconds"),
        component="queueing",
        description="p99 time spent waiting before the first forward pass. "
        "The admission-side half of TTFT.",
    ),
    "requests_running": MetricQuery(
        promql="sum(vllm:num_requests_running)",
        component="batch_composition",
        description="Sequences in the running batch.",
    ),
    "iteration_tokens_p50": MetricQuery(
        promql=_q("vllm:iteration_tokens_total", 0.50),
        component="batch_composition",
        description="Median tokens processed per scheduler step. Shows how "
        "the step budget splits between prefill and decode — the direct view "
        "of batch shape that requests_running alone cannot give.",
    ),
    # -- compute ---------------------------------------------------------
    "prefill_time_p99": MetricQuery(
        promql=_q("vllm:request_prefill_time_seconds"),
        component="prefill_cost",
        description="p99 prefill duration. The compute-side half of TTFT.",
    ),
    "prefill_kv_computed_p99": MetricQuery(
        promql=_q("vllm:request_prefill_kv_computed_tokens"),
        component="prefill_cost",
        description="p99 KV tokens actually computed during prefill. Rises "
        "when the cache stops serving reuse and when preemption forces "
        "recomputation — wasted work, measured directly.",
    ),
    "inter_token_latency_p99": MetricQuery(
        promql=_q("vllm:inter_token_latency_seconds"),
        component="decode_health",
        description="p99 inter-token latency. Co-symptom that separates "
        "decode starvation from a purely admission-side problem.",
    ),
    "generation_token_rate": MetricQuery(
        promql=f"sum(rate(vllm:generation_tokens_total[{RATE_WINDOW}]))",
        component="decode_health",
        description="Output tokens per second — realized decode throughput.",
    ),
    # -- hardware and host ------------------------------------------------
    "gpu_sm_active": MetricQuery(
        promql="avg(DCGM_FI_PROF_SM_ACTIVE)",
        component="gpu_saturation",
        description="Fraction of time SMs are active. Requires the DCGM "
        "exporter; absent on the CPU backend.",
        optional=True,
    ),
    "gpu_sm_clock": MetricQuery(
        promql="avg(DCGM_FI_DEV_SM_CLOCK)",
        component="gpu_saturation",
        description="SM clock. Drops on thermal or power throttling.",
        optional=True,
    ),
    # The API server, tokenizer, and detokenizer run on CPU. Starving them
    # spikes TTFT while the KV cache sits idle — the case that breaks every
    # "TTFT up + queue up implies cache pressure" heuristic.
    "host_cpu_rate": MetricQuery(
        promql=(
            f"sum(rate(container_cpu_usage_seconds_total{{{CONTAINER_SELECTOR}}}[{RATE_WINDOW}]))"
        ),
        component="host_saturation",
        description="CPU seconds/sec consumed by the server container.",
        optional=True,
    ),
    "host_cpu_throttle_ratio": MetricQuery(
        promql=(
            f"sum(rate(container_cpu_cfs_throttled_periods_total{{{CONTAINER_SELECTOR}}}[{RATE_WINDOW}]))"
            " / "
            f"clamp_min(sum(rate(container_cpu_cfs_periods_total{{{CONTAINER_SELECTOR}}}[{RATE_WINDOW}])), 1)"
        ),
        component="host_saturation",
        description="Fraction of CFS periods the container was throttled.",
        optional=True,
    ),
    # -- the SLI ----------------------------------------------------------
    "ttft_p99": MetricQuery(
        promql=_q("vllm:time_to_first_token_seconds"),
        component="ttft",
        description="p99 time to first token — the signal being explained.",
    ),
}


#: "A can cause B".  Read an edge as a mechanism, not a call.
MECHANISM_GRAPH: dict[str, list[str]] = {
    # Inputs push on the scheduler.
    "arrival_load": ["kv_cache_pressure", "queueing", "batch_composition", "gpu_saturation"],
    "request_shape": ["prefill_cost", "kv_cache_pressure"],
    # A prefix-cache miss means the prompt must actually be prefilled, and
    # its blocks must actually be allocated.
    "prefix_cache_efficacy": ["prefill_cost", "kv_cache_pressure"],
    # Not enough blocks -> the scheduler evicts running sequences and stops
    # admitting new ones.
    "kv_cache_pressure": ["preemption", "queueing"],
    # A preempted sequence re-enters the queue and must redo its prefill.
    "preemption": ["queueing", "prefill_cost", "decode_health"],
    # Batch shape decides how prefill and decode contend for each step.
    "batch_composition": ["prefill_cost", "decode_health", "gpu_saturation"],
    # A slow or throttled GPU makes every forward pass cost more.
    "gpu_saturation": ["prefill_cost", "decode_health"],
    # Starving the API server / tokenizer delays admission and the response
    # path without touching the GPU at all.
    "host_saturation": ["queueing", "ttft"],
    # The two halves of TTFT.
    "queueing": ["ttft"],
    "prefill_cost": ["ttft"],
    # Co-symptom, deliberately a leaf: slow decode is its own SLI (ITL) and is
    # evidence about batch behaviour, not a cause of first-token latency.
    "decode_health": [],
    "ttft": [],
}

# Declare each metric's measurement window from the PromQL it actually uses,
# so onset lag compensation cannot drift out of sync with the queries.
for _name, _q_obj in list(METRICS.items()):
    if f"[{RATE_WINDOW}]" in _q_obj.promql:
        _w = _secs(RATE_WINDOW)
    elif f"[{RATIO_WINDOW}]" in _q_obj.promql:
        _w = _secs(RATIO_WINDOW)
    else:
        _w = 0.0  # instantaneous gauge
    METRICS[_name] = MetricQuery(
        promql=_q_obj.promql,
        component=_q_obj.component,
        description=_q_obj.description,
        optional=_q_obj.optional,
        window_seconds=_w,
    )


VLLM = DomainSpec(
    name="vllm",
    metrics=METRICS,
    component_graph=MECHANISM_GRAPH,
    component_metrics={
        "arrival_load": ("request_rate", "prompt_token_rate"),
        "request_shape": (
            "prompt_tokens_p99",
            "prompt_tokens_p50",
            "requested_max_tokens_p99",
        ),
        "prefix_cache_efficacy": ("prefix_cache_hit_rate", "cached_token_ratio"),
        "kv_cache_pressure": ("kv_cache_usage",),
        "preemption": ("preemption_rate",),
        "queueing": ("requests_waiting", "queue_time_p99"),
        "batch_composition": ("requests_running", "iteration_tokens_p50"),
        "prefill_cost": ("prefill_time_p99", "prefill_kv_computed_p99"),
        "decode_health": ("inter_token_latency_p99", "generation_token_rate"),
        "gpu_saturation": ("gpu_sm_active", "gpu_sm_clock"),
        "host_saturation": ("host_cpu_rate", "host_cpu_throttle_ratio"),
        "ttft": ("ttft_p99",),
    },
    exogenous=frozenset({"arrival_load", "request_shape"}),
    sli_node="ttft",
    # Mechanisms inside one process propagate far faster than an RPC hop:
    # cache exhaustion to preemption to queue growth is sub-second. FChain's
    # 2.0s default would lump nearly the whole graph into one concurrency
    # window. Calibrate properly in Phase 4.
    concurrency_threshold_s=0.5,
    # Layers 1-4 test whether a change is detectable, never whether it is big
    # enough to matter. On a verified-clean run — 1.02/1.02 rps, zero
    # failures, no backlog growth — the pipeline still named a cause, because
    # ordinary drift in a histogram quantile is easily "detectable" when the
    # baseline is nearly constant. Requiring a 20-sigma shift removes that
    # false positive while leaving real faults (which move 50-2000 sigmas)
    # untouched.
    #
    # Calibrated against a single clean trace, so treat it as provisional:
    # Phase 5 should refit it across the full set of clean runs.
    min_effect_size=20.0,
    # "last_reset" estimates onset as where CUSUM evidence began accumulating.
    # For any metric that starts drifting at the window edge that collapses to
    # index ~0, so several mechanisms tie at +1s and the onset ordering Layers
    # 6-8 depend on carries no information. On the prefix_diversity trace it
    # put the effect (prefill_cost) ahead of its own cause.
    #
    # "crossing" uses the point where evidence became conclusive — a
    # consistent definition across metrics, which is what *relative* ordering
    # needs. It is biased later for small shifts, but the effect-size gate
    # already removes those. Boutique keeps last_reset.
    onset_estimator="crossing",
)

_problems = VLLM.validate()
if _problems:  # pragma: no cover - guards against edits to the constants above
    raise ValueError("vllm DomainSpec is inconsistent:\n  " + "\n  ".join(_problems))
