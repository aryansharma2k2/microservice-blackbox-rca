"""Stage 9 — turn a diagnosis into something an on-call engineer can act on.

**The pipeline decides; the model only narrates.** The ranked list, the
verdict, and the onset ordering are all produced by Layers 1-8 before this
stage runs. The model receives them as settled facts and writes the
explanation and the remediation. It is never asked which mechanism is at
fault, and its output cannot change the diagnosis.

That separation is the whole point, and it is worth stating plainly in the
README: an LLM that *decides* would make every number in the evaluation a
measurement of the model rather than of the pipeline, and would not be
reproducible from the committed traces. An LLM that *writes* leaves the
evidence chain intact — anyone can re-derive the diagnosis offline and check
the narrative against it.

    from rca_engine.explainer import explain
    print(explain(report, evidence).narrative)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rca_engine.domains import DomainSpec
from rca_engine.fault_chain import (
    VERDICT_CAPACITY,
    VERDICT_EXTERNAL,
    VERDICT_NO_ANOMALY,
    VERDICT_PATHOLOGY,
    RcaReport,
)

logger = logging.getLogger(__name__)

MODEL = "claude-opus-5"

SYSTEM = """\
You write incident explanations for engineers running LLM inference servers.

A root-cause analysis pipeline has ALREADY diagnosed the incident. Its verdict \
and ranking are settled facts — your job is to explain them and recommend a \
fix, never to re-diagnose. Do not second-guess the ranking, propose a \
different root cause, or hedge about whether the diagnosis is right.

Write for someone paged at 3am who needs to act:
- Open with what broke and why, in one or two sentences.
- Explain the causal chain using the onset times given, so the reader can see \
why the cause precedes its effects.
- End with a concrete remediation: a specific flag, a capacity change, or a \
client-side fix. Name the actual vLLM setting where one applies.

Be direct. No preamble, no bullet-point walls, no restating the input back.\
"""

_VERDICT_FRAMING = {
    VERDICT_PATHOLOGY: (
        "Something inside the server is misbehaving. Recommend a fix."
    ),
    VERDICT_CAPACITY: (
        "Nothing is broken — the deployment is being asked for more than it can "
        "serve. Recommend capacity or admission control, NOT a code fix."
    ),
    VERDICT_EXTERNAL: (
        "Everything moved together; no single component explains it. Say so, "
        "and suggest what to check next."
    ),
    VERDICT_NO_ANOMALY: (
        "Nothing was abnormal. Say so plainly and briefly; do not invent a "
        "problem to explain."
    ),
}


@dataclass
class Explanation:
    narrative: str
    model: str
    #: False when the SDK is missing, the call failed, or the model declined —
    #: the caller still has the full structured report to fall back on.
    generated: bool


def build_evidence(report: RcaReport, spec: DomainSpec, fault_start: float) -> str:
    """Render the diagnosis as the evidence bundle the model explains."""
    lines = [
        f"VERDICT: {report.verdict}",
        f"GUIDANCE: {_VERDICT_FRAMING.get(report.verdict, '')}",
        "",
    ]

    if report.exogenous_drivers:
        lines += [
            f"WORKLOAD INPUTS THAT MOVED: {', '.join(report.exogenous_drivers)}",
            "",
        ]

    top = report.top()
    lines.append(
        f"ROOT CAUSE: {top['service'] if top else 'none — nothing eligible was abnormal'}"
    )
    lines += ["", "RANKED COMPONENTS (onset relative to fault start):", ""]

    for entry in report.ranked:
        onset = entry.get("onset_time")
        offset = (
            f"+{onset - fault_start:.0f}s" if isinstance(onset, (int, float)) else "?"
        )
        tag = "" if entry.get("eligible", True) else "  [evidence only — cannot be the cause]"
        metrics = ", ".join(entry.get("abnormal_metrics") or []) or "-"
        lines.append(
            f"  {entry.get('rank', '?')}. {entry.get('service')}  onset {offset}  "
            f"confidence {entry.get('confidence', 0):.2f}{tag}"
        )
        lines.append(f"       abnormal: {metrics}")

    lines += ["", "MECHANISM GRAPH (A -> B means A can cause B):", ""]
    for component, targets in sorted(spec.component_graph.items()):
        lines.append(f"  {component} -> {', '.join(targets) if targets else '(leaf)'}")

    return "\n".join(lines)


def explain(
    report: RcaReport,
    spec: DomainSpec,
    fault_start: float,
    effort: str = "medium",
) -> Explanation:
    """Write an operator-facing narrative for an already-settled diagnosis.

    Never raises: a missing SDK, a failed call, or a refusal degrades to the
    structured evidence bundle rather than taking down a capture run.
    """
    evidence = build_evidence(report, spec, fault_start)

    try:
        import anthropic
    except ImportError:
        logger.info("anthropic SDK not installed — returning the evidence bundle")
        return Explanation(narrative=evidence, model="", generated=False)

    try:
        response = anthropic.Anthropic().beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            messages=[{"role": "user", "content": evidence}],
            # Narration is not the intelligence-sensitive part of this system —
            # the diagnosis already happened. Medium keeps it cheap and terse.
            output_config={"effort": effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except Exception as exc:  # noqa: BLE001 - explanation is a nicety, not the result
        logger.warning("Explainer call failed: %s: %s", type(exc).__name__, exc)
        return Explanation(narrative=evidence, model=MODEL, generated=False)

    if response.stop_reason == "refusal":
        logger.warning("Explainer refused: %s", response.stop_details)
        return Explanation(narrative=evidence, model=MODEL, generated=False)

    text = "\n".join(b.text for b in response.content if b.type == "text")
    return Explanation(narrative=text, model=response.model, generated=True)
