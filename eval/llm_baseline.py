"""LLM-only baseline — the comparison the whole project turns on.

Give a frontier model exactly the evidence the pipeline gets (per-component
metric shifts, the mechanism graph, the eligible candidates) and ask it to name
the root cause. No CUSUM, no onset ordering, no graph filtering — just the
model.

"Does an eight-layer statistical pipeline beat asking Claude?" is the question
a reader will have immediately, and it deserves a measured answer rather than
an assumption. Either result is publishable: if the pipeline wins, its
complexity is justified; if the LLM wins, that is the more interesting finding
and the writeup is stronger for reporting it.

Responses are cached to disk keyed by a hash of the prompt, so `make eval`
re-derives the same numbers without an API key or repeated spend. The cache is
committed alongside the traces — that is what keeps the published comparison
reproducible by anyone who clones the repo.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from eval.baselines import _eligible, _is_flat, _split
from rca_engine.domains import DomainSpec

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "traces" / "llm_cache"

#: Opus 5 — the strongest model available, so the comparison is against the
#: best case for "just ask a model" rather than a cheap strawman.
MODEL = "claude-opus-5"


class Diagnosis(BaseModel):
    """Structured answer, so parsing never becomes the failure mode."""

    # additionalProperties: false is required by structured outputs.
    model_config = {"extra": "forbid"}

    ranked_components: list[str] = Field(
        description=(
            "Candidate root causes, most likely first. Use only names from the "
            "eligible list. Empty if nothing is wrong."
        )
    )
    verdict: str = Field(
        description="One of: pathology, capacity, no_anomaly, external."
    )
    reasoning: str = Field(description="Two or three sentences of justification.")


SYSTEM = """\
You diagnose latency regressions inside a vLLM inference server.

You are given, for one incident, how every monitored signal changed between a \
clean baseline window and a fault window, plus the causal graph of the \
server's internal mechanisms. Name which mechanism is the ROOT CAUSE.

The hard part is that these mechanisms cause one another. Preemption is both a \
consequence of KV cache pressure and a cause of latency. A prefix-cache \
collapse raises cache usage. Rank the cause above its own effects.

Some components can never be the answer:
- Exogenous inputs (arrival rate, request shape) are workload, not defects.
- The SLI itself (ttft) is the symptom being explained.
- Co-symptoms with no causal path to the SLI.
Only rank components from the eligible list you are given.

Verdicts:
- pathology  — something inside the server is misbehaving
- capacity   — an input rose; the server is being asked for more than it can \
serve, and nothing is broken
- no_anomaly — nothing meaningfully changed
- external   — everything moved together with no single explanation\
"""


def summarize_evidence(
    matrix, baseline_window, fault_window, spec: DomainSpec, step: float = 1.0
) -> str:
    """Render the same evidence the pipeline sees, as text.

    Deliberately includes the *shift* and the baseline noise level rather than
    raw series: dumping ~20 series x 300 points would bury the signal and make
    the comparison about context handling rather than diagnosis.
    """
    data = _split(matrix, baseline_window, fault_window, step)
    eligible = sorted(_eligible(spec))

    lines = ["## Signal changes (baseline -> fault)", ""]
    lines.append(f"{'component / metric':<48}{'baseline':>12}{'fault':>12}{'shift':>10}")
    lines.append("-" * 82)

    rows = []
    for component in sorted(data):
        for name, (base, fault) in sorted(data[component].items()):
            b, f = float(np.mean(base)), float(np.mean(fault))
            if _is_flat(base) and _is_flat(fault) and abs(f - b) < 1e-12:
                shift = 0.0
            else:
                shift = (f - b) / abs(b) if abs(b) > 1e-12 else float("inf")
            rows.append((component, name, b, f, shift))

    for component, name, b, f, shift in sorted(rows, key=lambda r: -abs(r[4])):
        pct = "n/a" if shift == float("inf") else f"{shift:+.0%}"
        lines.append(f"{component + '/' + name:<48}{b:>12.3f}{f:>12.3f}{pct:>10}")

    lines += ["", "## Causal graph (A -> B means A can cause B)", ""]
    for component, targets in sorted(spec.component_graph.items()):
        lines.append(f"  {component} -> {', '.join(targets) if targets else '(leaf)'}")

    lines += [
        "",
        "## Eligible root causes (rank only these)",
        "  " + ", ".join(eligible),
        "",
        f"## Excluded (evidence only, never the answer)",
        "  " + ", ".join(sorted(spec.excluded_from_root_cause())),
    ]
    return "\n".join(lines)


def _cache_path(prompt: str) -> Path:
    digest = hashlib.sha256((MODEL + prompt).encode()).hexdigest()[:16]
    return CACHE_DIR / f"{digest}.json"


def diagnose(prompt: str, use_cache: bool = True) -> Diagnosis | None:
    """Ask the model once. Returns None if it declined or the SDK is absent."""
    cached = _cache_path(prompt)
    if use_cache and cached.exists():
        return Diagnosis(**json.loads(cached.read_text())["diagnosis"])

    try:
        import anthropic
    except ImportError:
        logger.warning(
            "anthropic SDK not installed and no cached response — skipping the "
            "LLM baseline. Install with: pip install -e '.[llm]'"
        )
        return None

    client = anthropic.Anthropic()
    try:
        # Structured outputs via output_config.format rather than the .parse()
        # helper: refusal fallbacks live on the beta namespace, and combining
        # them is the documented shape for messages.create.
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": Diagnosis.model_json_schema(),
                }
            },
            # Opus 5's safety classifiers can decline; routing by refusal
            # category recovers the request instead of losing the data point.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except Exception as exc:  # noqa: BLE001 - one failed call must not end the eval
        logger.warning("LLM baseline call failed: %s: %s", type(exc).__name__, exc)
        return None

    # Always check before reading content: a refusal returns HTTP 200 with no
    # usable answer, and indexing into it would crash mid-evaluation.
    if response.stop_reason == "refusal":
        logger.warning("LLM baseline refused: %s", response.stop_details)
        return None

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        diagnosis = Diagnosis(**json.loads(text))
    except (ValueError, TypeError) as exc:
        logger.warning("LLM baseline returned unparseable output: %s", exc)
        return None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_text(
        json.dumps(
            {"model": MODEL, "prompt": prompt, "diagnosis": diagnosis.model_dump()},
            indent=2,
        )
        + "\n"
    )
    return diagnosis


def llm(
    matrix, baseline_window, fault_window, spec: DomainSpec, step: float = 1.0
) -> list[str]:
    """Baseline entry point — same signature as the statistical baselines."""
    prompt = summarize_evidence(matrix, baseline_window, fault_window, spec, step)
    diagnosis = diagnose(prompt, use_cache=os.environ.get("LLM_NO_CACHE") != "1")
    if diagnosis is None:
        return []
    # Drop anything ineligible, so a hallucinated or excluded name cannot
    # flatter the score.
    eligible = _eligible(spec)
    return [c for c in diagnosis.ranked_components if c in eligible]
