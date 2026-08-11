"""Domain adapters — what the RCA engine needs to know about a target system.

Layers 1-8 localize a fault among components in a directed graph and make no
assumption about what those components are.  A :class:`DomainSpec` supplies
the parts that differ per system: the PromQL to run, how result series map to
components, and how components relate.

    from rca_engine.domains import BOUTIQUE, get_domain

    spec = get_domain("boutique")
"""

from __future__ import annotations

from rca_engine.domains.base import DomainSpec, MetricQuery, histogram_quantile
from rca_engine.domains.boutique import BOUTIQUE
from rca_engine.domains.vllm import VLLM

#: Registry of built-in domains, keyed by ``DomainSpec.name``.
DOMAINS: dict[str, DomainSpec] = {
    BOUTIQUE.name: BOUTIQUE,
    VLLM.name: VLLM,
}

#: Used wherever a domain is not supplied explicitly, preserving the
#: engine's behaviour from before the domain layer existed.
DEFAULT_DOMAIN: DomainSpec = BOUTIQUE


def get_domain(name: str) -> DomainSpec:
    """Look up a registered domain by name.

    Raises
    ------
    KeyError
        If *name* is not registered, with the available names in the message.
    """
    try:
        return DOMAINS[name]
    except KeyError:
        raise KeyError(
            f"Unknown domain {name!r}. Available: {', '.join(sorted(DOMAINS))}"
        ) from None


def register_domain(spec: DomainSpec) -> DomainSpec:
    """Add *spec* to the registry, validating it first."""
    problems = spec.validate()
    if problems:
        raise ValueError(
            f"DomainSpec {spec.name!r} is inconsistent:\n  " + "\n  ".join(problems)
        )
    DOMAINS[spec.name] = spec
    return spec


__all__ = [
    "BOUTIQUE",
    "VLLM",
    "DEFAULT_DOMAIN",
    "DOMAINS",
    "DomainSpec",
    "MetricQuery",
    "get_domain",
    "histogram_quantile",
    "register_domain",
]
