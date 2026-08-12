"""The labeled fault library for vLLM TTFT experiments.

Each scenario is a controlled way to make p99 TTFT get worse, paired with the
mechanism that is genuinely responsible.  That pairing is the ground truth the
evaluation scores against, and it is what the current repo has never had:
faults were injected, but nothing ever compared the diagnosis to the label.

Three kinds, in descending order of how much infrastructure they need:

``workload``
    Change only what the client sends.  No restarts, no chaos tooling, no
    privileged access — just a different request mix.  Cheap, deterministic,
    and far more reproducible than the CPU/memory hogs the Boutique domain
    relies on.  Most of the library is this kind.

``config``
    Restart the server with a degraded flag.  Each restart *is* the fault.
    These produce the cleanest signals because the mechanism is constrained
    directly rather than induced.

``infra``
    Reuse the existing Chaos Mesh injectors against the server's container or
    its GPU.  The one case that needs the Kubernetes path.

Why the confounders matter
--------------------------
Several scenarios are designed to look alike on the surface.
``kv_cache_starved`` and ``host_cpu_hog`` both show a growing queue and a TTFT
spike, but in the second the KV cache never moves — the API server and
tokenizer are simply starved of CPU while the GPU idles.  Any rule of the form
"TTFT up and queue up implies cache pressure" gets one of them wrong.  Those
pairs are the experiment, not decoration.

Expected verdicts
-----------------
Every scenario declares both the mechanism it should surface and the verdict
it should produce.  A workload change that the exogenous metrics can see
(more requests, longer prompts) should read as ``capacity`` — nothing is
broken, the deployment is being asked for more than it can serve.  A genuine
internal fault should read as ``pathology``.  Scoring both is stricter than
scoring the ranking alone, and the distinction is what an operator acts on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rca_engine.fault_chain import (
    VERDICT_CAPACITY,
    VERDICT_NO_ANOMALY,
    VERDICT_PATHOLOGY,
)

WORKLOAD = "workload"
CONFIG = "config"
INFRA = "infra"


@dataclass(frozen=True)
class Phase:
    """A steady-state workload shape.

    Token counts are approximate — see ``workload.generator.tokens_to_words``.
    Exact counts do not matter; what matters is the *ratio* between a
    scenario's baseline and fault phases, which is robust to the estimate.
    """

    #: Requests per second, held constant. Deliberately not the sine wave the
    #: Boutique generator uses: a periodic component would put a strong
    #: frequency in every series and give the FFT burst filter something to
    #: chew on that has nothing to do with the injected fault.
    rps: float
    #: Prompt length range in tokens, sampled uniformly per request.
    prompt_tokens: tuple[int, int]
    #: Output length cap in tokens, sampled uniformly per request.
    max_tokens: tuple[int, int]
    #: Number of shared prefixes to draw from. ``0`` gives every request a
    #: unique prefix, which drives the prefix cache hit rate to zero.
    shared_prefixes: int = 8

    def describe(self) -> str:
        prefix = (
            "unique prefixes"
            if self.shared_prefixes == 0
            else f"{self.shared_prefixes} shared prefixes"
        )
        return (
            f"{self.rps:g} rps, prompt {self.prompt_tokens[0]}-{self.prompt_tokens[1]}tok, "
            f"output {self.max_tokens[0]}-{self.max_tokens[1]}tok, {prefix}"
        )


#: The steady state every scenario departs from. Modest load, short prompts,
#: short outputs, high prefix reuse — a well-behaved chat workload.
NOMINAL = Phase(
    rps=6.0,
    prompt_tokens=(256, 512),
    max_tokens=(48, 96),
    shared_prefixes=8,
)


@dataclass(frozen=True)
class Scenario:
    """One labeled way to degrade TTFT."""

    name: str
    kind: str
    #: The mechanism component that should be ranked first. ``None`` for clean
    #: runs, where the correct answer is "nothing".
    ground_truth: str | None
    #: The verdict the pipeline should reach.
    expect_verdict: str
    summary: str
    fault: Phase
    baseline: Phase = NOMINAL
    #: ``config`` kind: extra server flags for the degraded restart.
    server_args: tuple[str, ...] = ()
    #: ``infra`` kind: fault name understood by fault_injection/chaos_inject.py.
    infra_fault: str | None = None
    #: Scenarios that look like this one on the surface. Used by the
    #: evaluation to report a confusion matrix over the pairs that matter.
    confounds_with: tuple[str, ...] = ()
    #: Set when a scenario is known to stress something the graph models
    #: imperfectly. Phase 5 reports these separately rather than hiding them.
    caveat: str = ""
    #: Only runnable with a real GPU.
    requires_gpu: bool = False

    def __post_init__(self) -> None:
        if self.kind not in (WORKLOAD, CONFIG, INFRA):
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}")
        if self.kind == CONFIG and not self.server_args:
            raise ValueError(f"{self.name}: config scenarios need server_args")
        if self.kind == INFRA and not self.infra_fault:
            raise ValueError(f"{self.name}: infra scenarios need infra_fault")
        if self.kind != CONFIG and self.server_args:
            raise ValueError(f"{self.name}: server_args only apply to config kind")


_SCENARIOS: tuple[Scenario, ...] = (
    # ---------------------------------------------------------------
    # Workload — change only what the client sends
    # ---------------------------------------------------------------
    Scenario(
        name="qps_ramp",
        kind=WORKLOAD,
        ground_truth="kv_cache_pressure",
        expect_verdict=VERDICT_CAPACITY,
        summary="4x the request rate at a fixed request shape. The textbook "
        "overload: nothing is broken, there is simply more work than the "
        "deployment can absorb.",
        fault=Phase(rps=24.0, prompt_tokens=(256, 512), max_tokens=(48, 96)),
        confounds_with=("kv_cache_starved",),
    ),
    Scenario(
        name="long_prompt_burst",
        kind=WORKLOAD,
        ground_truth="prefill_cost",
        expect_verdict=VERDICT_CAPACITY,
        summary="Same rate, prompts ~30x longer. Prefill work per request "
        "explodes and long prefills block the scheduler head-of-line.",
        fault=Phase(rps=6.0, prompt_tokens=(8192, 16384), max_tokens=(48, 96)),
        confounds_with=("prefix_diversity", "gpu_contention"),
    ),
    Scenario(
        name="long_output_burst",
        kind=WORKLOAD,
        ground_truth="kv_cache_pressure",
        expect_verdict=VERDICT_CAPACITY,
        summary="Same rate and prompt size, but each request generates ~15x "
        "more tokens. KV blocks are held far longer, so the cache fills from "
        "the decode side rather than the admission side.",
        fault=Phase(rps=6.0, prompt_tokens=(256, 512), max_tokens=(768, 1536)),
        confounds_with=("qps_ramp", "kv_cache_starved"),
    ),
    Scenario(
        name="prefix_diversity",
        kind=WORKLOAD,
        ground_truth="prefix_cache_efficacy",
        expect_verdict=VERDICT_PATHOLOGY,
        summary="Identical rate and prompt length, but every request carries a "
        "unique prefix. The prefix cache hit rate collapses and every request "
        "must be prefilled from scratch.",
        fault=Phase(
            rps=6.0,
            prompt_tokens=(256, 512),
            max_tokens=(48, 96),
            shared_prefixes=0,
        ),
        confounds_with=("long_prompt_burst",),
        caveat=(
            "This is a workload change the exogenous metrics cannot see — rate "
            "and length are unchanged — so it reads as pathology rather than "
            "capacity. Arguably correct: the useful answer is that the cache "
            "stopped working. Worth reporting explicitly."
        ),
    ),
    Scenario(
        name="bimodal_mix",
        kind=WORKLOAD,
        ground_truth="batch_composition",
        expect_verdict=VERDICT_CAPACITY,
        summary="Short and very long prompts interleaved. Batch composition "
        "swings step to step and short requests queue behind long prefills.",
        fault=Phase(rps=6.0, prompt_tokens=(128, 16384), max_tokens=(48, 96)),
        confounds_with=("long_prompt_burst",),
    ),
    # ---------------------------------------------------------------
    # Config — restart the server degraded
    # ---------------------------------------------------------------
    Scenario(
        name="kv_cache_starved",
        kind=CONFIG,
        ground_truth="kv_cache_pressure",
        expect_verdict=VERDICT_PATHOLOGY,
        summary="Nominal workload against a server given a fraction of the "
        "usual KV cache. Cache pressure and preemption at a load the "
        "deployment should handle comfortably.",
        fault=NOMINAL,
        # Constrains the cache directly rather than via total memory, so it
        # means the same thing on the CPU and GPU backends —
        # --gpu-memory-utilization is meaningless on CPU.
        server_args=("--kv-cache-memory-bytes=268435456",),
        confounds_with=("qps_ramp", "long_output_burst"),
    ),
    Scenario(
        name="queue_starved",
        kind=CONFIG,
        ground_truth="queueing",
        expect_verdict=VERDICT_PATHOLOGY,
        summary="Batch width capped far below what the load needs. Requests "
        "wait on admission while the GPU is nowhere near saturated.",
        fault=NOMINAL,
        server_args=("--max-num-seqs=4",),
        confounds_with=("host_cpu_hog",),
    ),
    Scenario(
        name="chunked_prefill_off",
        kind=CONFIG,
        ground_truth="batch_composition",
        expect_verdict=VERDICT_PATHOLOGY,
        summary="Chunked prefill disabled, so a long prefill monopolises its "
        "step instead of being split. Decode stalls behind prefill.",
        fault=Phase(rps=6.0, prompt_tokens=(2048, 4096), max_tokens=(48, 96)),
        server_args=("--no-enable-chunked-prefill",),
        confounds_with=("bimodal_mix",),
    ),
    Scenario(
        name="prefix_cache_off",
        kind=CONFIG,
        ground_truth="prefix_cache_efficacy",
        expect_verdict=VERDICT_PATHOLOGY,
        summary="Prefix caching disabled server-side, against a workload with "
        "high prefix reuse. The mirror image of prefix_diversity: same "
        "mechanism, opposite side of the client/server boundary.",
        fault=NOMINAL,
        server_args=("--no-enable-prefix-caching",),
        confounds_with=("prefix_diversity",),
    ),
    Scenario(
        name="block_size_mismatch",
        kind=CONFIG,
        ground_truth="kv_cache_pressure",
        expect_verdict=VERDICT_PATHOLOGY,
        summary="Large KV block size against short sequences. Every sequence "
        "wastes most of its last block — internal fragmentation, the honest "
        "version of the 'memory fragmentation' hypothesis.",
        fault=Phase(rps=10.0, prompt_tokens=(32, 96), max_tokens=(16, 32)),
        server_args=("--block-size=32",),
        caveat=(
            "PagedAttention has no external fragmentation, so the effect is "
            "bounded by block_size relative to sequence length and may be "
            "weak. If it does not perturb kv_cache_pressure, that is a finding "
            "about the hypothesis, not a pipeline miss."
        ),
    ),
    # ---------------------------------------------------------------
    # Infra — chaos against the container or the GPU
    # ---------------------------------------------------------------
    Scenario(
        name="host_cpu_hog",
        kind=INFRA,
        ground_truth="host_saturation",
        expect_verdict=VERDICT_PATHOLOGY,
        summary="CPU starvation on the server container. The API server, "
        "tokenizer, and detokenizer stall while the GPU sits idle. TTFT and "
        "queue time spike with the KV cache untouched — the case that breaks "
        "every cache-centric heuristic.",
        fault=NOMINAL,
        infra_fault="cpu_hog",
        confounds_with=("queue_starved", "kv_cache_starved"),
    ),
    Scenario(
        name="gpu_contention",
        kind=INFRA,
        ground_truth="gpu_saturation",
        expect_verdict=VERDICT_PATHOLOGY,
        summary="A co-tenant process competing for the GPU. Every forward "
        "pass costs more, so prefill and decode both slow with no change in "
        "scheduler state.",
        fault=NOMINAL,
        infra_fault="gpu_cotenant",
        confounds_with=("long_prompt_burst",),
        requires_gpu=True,
    ),
    # ---------------------------------------------------------------
    # Control
    # ---------------------------------------------------------------
    Scenario(
        name="clean",
        kind=WORKLOAD,
        ground_truth=None,
        expect_verdict=VERDICT_NO_ANOMALY,
        summary="Nominal load throughout, no fault. Measures the false "
        "positive rate — a diagnoser that always names something is useless, "
        "and nothing in the repo has ever measured this.",
        fault=NOMINAL,
    ),
)

SCENARIOS: dict[str, Scenario] = {s.name: s for s in _SCENARIOS}


def get_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError:
        raise KeyError(
            f"Unknown scenario {name!r}. Available: {', '.join(sorted(SCENARIOS))}"
        ) from None


def scenarios_for(
    *,
    kind: str | None = None,
    gpu_available: bool = True,
) -> list[Scenario]:
    """Select runnable scenarios, optionally filtered by kind."""
    return [
        s
        for s in _SCENARIOS
        if (kind is None or s.kind == kind)
        and (gpu_available or not s.requires_gpu)
    ]


def confounder_pairs() -> set[frozenset[str]]:
    """Scenario pairs that look alike on the surface.

    The evaluation reports these separately: overall accuracy can look healthy
    while every confounded pair is being guessed.
    """
    pairs = set()
    for scenario in _SCENARIOS:
        for other in scenario.confounds_with:
            pairs.add(frozenset({scenario.name, other}))
    return pairs
