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

**Block churn is observable.**  PagedAttention largely eliminates *external*
fragmentation, so "memory fragmentation" in the classical sense is close to a
red herring here.  The sharp version — prefix-cache eviction thrash — is
directly measurable via ``kv_block_idle_before_evict_seconds`` and
``kv_block_reuse_gap_seconds``.

Exogenous nodes
---------------
``arrival_load`` and ``prompt_shape`` are inputs, not defects.  They are
usually the first thing to move, so without marking them exogenous the
earliest-onset rule would blame "load" for every incident.  Marking them lets
Layer 6 return a *capacity* verdict ("you are asking for more than this
deployment can serve") instead of a bogus root cause.  ``ttft`` is likewise
excluded — it is the symptom being explained.

Status
------
The metric names here follow vLLM's V1 engine.  They have **not** yet been
validated against a running server; ``scripts/discover_metrics.py`` does that
and fails loudly on drift.  Names moved between V0 and V1
(``gpu_cache_usage_perc`` -> ``kv_cache_usage_perc``,
``time_in_queue_requests`` -> ``request_queue_time_seconds``) and published
sources disagree about ``_total`` suffixes on the prefix-cache counters, so
treat this module as a hypothesis until discovery confirms it.
"""

from __future__ import annotations

from rca_engine.domains.base import DomainSpec, MetricQuery, histogram_quantile

#: Kubernetes namespace the server runs in, for the container-level metrics
#: that catch API-server/tokenizer starvation.
NAMESPACE = "vllm"

#: Rate window for counters and histogram buckets.  Wide enough that a p99 is
#: meaningful at single-digit RPS, narrow enough that CUSUM still sees an
#: onset promptly.  Revisit against real traces in Phase 4.
RATE_WINDOW = "30s"

#: Quantile used for the latency signals.  p99 is the SLI teams alert on.
Q = 0.99


def _q(metric: str, quantile: float = Q) -> str:
    return histogram_quantile(metric, quantile, RATE_WINDOW)


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
        component="prompt_shape",
        description="p99 prompt length. Rises on a long-prompt burst.",
    ),
    "prompt_tokens_p50": MetricQuery(
        promql=_q("vllm:request_prompt_tokens", 0.50),
        component="prompt_shape",
        description="Median prompt length. Separates a shifted "
        "distribution from a heavy tail.",
    ),
    # -- cache and memory ------------------------------------------------
    "kv_cache_usage": MetricQuery(
        promql="sum(vllm:kv_cache_usage_perc)",
        component="kv_cache_pressure",
        description="Fraction of KV cache blocks in use (0-1).",
    ),
    "kv_block_reuse_gap_p50": MetricQuery(
        promql=_q("vllm:kv_block_reuse_gap_seconds", 0.50),
        component="kv_cache_pressure",
        description="Time between reuses of a KV block. Collapses when the "
        "cache is churning.",
    ),
    "prefix_cache_hit_rate": MetricQuery(
        promql=(
            f"sum(rate(vllm:prefix_cache_hits_total[{RATE_WINDOW}]))"
            " / "
            f"clamp_min(sum(rate(vllm:prefix_cache_queries_total[{RATE_WINDOW}])), 1)"
        ),
        component="prefix_cache_efficacy",
        description="Prefix cache hit ratio. Collapses under prefix-diversity "
        "pressure, forcing full re-prefill of every request.",
    ),
    "kv_block_idle_before_evict_p50": MetricQuery(
        promql=_q("vllm:kv_block_idle_before_evict_seconds", 0.50),
        component="prefix_cache_efficacy",
        description="How long a block sits idle before eviction. Short means "
        "eviction thrash — blocks are evicted while still useful.",
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
    # -- compute ---------------------------------------------------------
    "prefill_time_p99": MetricQuery(
        promql=_q("vllm:request_prefill_time_seconds"),
        component="prefill_cost",
        description="p99 prefill duration. The compute-side half of TTFT.",
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
    ),
    "gpu_sm_clock": MetricQuery(
        promql="avg(DCGM_FI_DEV_SM_CLOCK)",
        component="gpu_saturation",
        description="SM clock. Drops on thermal or power throttling.",
    ),
    # The API server, tokenizer, and detokenizer run on CPU. Starving them
    # spikes TTFT while the KV cache sits idle — the case that breaks every
    # "TTFT up + queue up implies cache pressure" heuristic.
    "host_cpu_rate": MetricQuery(
        promql=(
            f'sum(rate(container_cpu_usage_seconds_total{{namespace="{NAMESPACE}",container!=""}}[{RATE_WINDOW}]))'
        ),
        component="host_saturation",
        description="CPU seconds/sec consumed by the server container.",
    ),
    "host_cpu_throttle_ratio": MetricQuery(
        promql=(
            f'sum(rate(container_cpu_cfs_throttled_periods_total{{namespace="{NAMESPACE}"}}[{RATE_WINDOW}]))'
            " / "
            f'clamp_min(sum(rate(container_cpu_cfs_periods_total{{namespace="{NAMESPACE}"}}[{RATE_WINDOW}])), 1)'
        ),
        component="host_saturation",
        description="Fraction of CFS periods the container was throttled.",
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
    "prompt_shape": ["prefill_cost", "kv_cache_pressure"],
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

VLLM = DomainSpec(
    name="vllm",
    metrics=METRICS,
    component_graph=MECHANISM_GRAPH,
    component_metrics={
        "arrival_load": ("request_rate", "prompt_token_rate"),
        "prompt_shape": ("prompt_tokens_p99", "prompt_tokens_p50"),
        "prefix_cache_efficacy": (
            "prefix_cache_hit_rate",
            "kv_block_idle_before_evict_p50",
        ),
        "kv_cache_pressure": ("kv_cache_usage", "kv_block_reuse_gap_p50"),
        "preemption": ("preemption_rate",),
        "queueing": ("requests_waiting", "queue_time_p99"),
        "batch_composition": ("requests_running",),
        "prefill_cost": ("prefill_time_p99",),
        "decode_health": ("inter_token_latency_p99", "generation_token_rate"),
        "gpu_saturation": ("gpu_sm_active", "gpu_sm_clock"),
        "host_saturation": ("host_cpu_rate", "host_cpu_throttle_ratio"),
        "ttft": ("ttft_p99",),
    },
    exogenous=frozenset({"arrival_load", "prompt_shape"}),
    sli_node="ttft",
    # Mechanisms inside one process propagate far faster than an RPC hop:
    # cache exhaustion to preemption to queue growth is sub-second. FChain's
    # 2.0s default would lump nearly the whole graph into one concurrency
    # window. Calibrate properly in Phase 4.
    concurrency_threshold_s=0.5,
)

_problems = VLLM.validate()
if _problems:  # pragma: no cover - guards against edits to the constants above
    raise ValueError("vllm DomainSpec is inconsistent:\n  " + "\n  ".join(_problems))
