"""Snapshot a server's Prometheus metric surface and check a domain against it.

Every metric name a :class:`DomainSpec` references is a guess until something
confirms the server actually exposes it.  vLLM renamed metrics between its V0
and V1 engines (``gpu_cache_usage_perc`` -> ``kv_cache_usage_perc``,
``time_in_queue_requests`` -> ``request_queue_time_seconds``) and published
sources disagree about ``_total`` suffixes on the prefix-cache counters.  A
silently-missing metric is the worst failure mode available: the component it
backs simply never goes abnormal, and the pipeline confidently blames
something else.

So: scrape ``/metrics``, record what is really there, and fail loudly when the
domain references something absent.

Usage
-----
    # Record what a running server exposes
    python -m rca_engine.scripts.discover_metrics snapshot \\
        --url http://localhost:8000/metrics \\
        --out deploy/vllm/metric_surface.json

    # Check a domain against that snapshot (no server needed)
    python -m rca_engine.scripts.discover_metrics check vllm \\
        --surface deploy/vllm/metric_surface.json

    # Or check straight against a live server
    python -m rca_engine.scripts.discover_metrics check vllm \\
        --url http://localhost:8000/metrics

Exits non-zero when a referenced metric is missing, so it can gate CI.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import requests

from rca_engine.domains import DOMAINS, DomainSpec, get_domain

# PromQL identifiers that are functions, keywords, or aggregation modifiers
# rather than metric names.
_NOT_METRICS = frozenset(
    {
        "abs", "absent", "avg", "avg_over_time", "by", "ceil", "changes",
        "clamp", "clamp_max", "clamp_min", "count", "count_over_time", "delta",
        "deriv", "exp", "floor", "group_left", "group_right", "histogram_quantile",
        "holt_winters", "idelta", "ignoring", "increase", "irate", "label_join",
        "label_replace", "ln", "log10", "log2", "max", "max_over_time", "min",
        "min_over_time", "offset", "on", "predict_linear", "quantile",
        "quantile_over_time", "rate", "resets", "round", "scalar", "sgn",
        "sort", "sort_desc", "sqrt", "stddev", "stdvar", "sum", "sum_over_time",
        "time", "timestamp", "topk", "bottomk", "unless", "vector", "without",
        "and", "or", "bool", "le", "inf", "nan",
    }
)

_LABEL_BLOCK_RE = re.compile(r"\{[^}]*\}")
# Range selectors like [30s] and [1m]: the duration unit is a bare letter that
# would otherwise be picked up as a metric name.
_RANGE_RE = re.compile(r"\[[^\]]*\]")
# Aggregation clauses name *labels*, not metrics, and use parentheses rather
# than braces: `sum by (pod, namespace) (...)`.
_AGG_CLAUSE_RE = re.compile(
    r"\b(?:by|without|on|ignoring|group_left|group_right)\s*\([^)]*\)"
)
_IDENT_RE = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")
_TYPE_RE = re.compile(r"^#\s*TYPE\s+(\S+)\s+(\S+)", re.MULTILINE)
_SAMPLE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)", re.MULTILINE)

#: Suffixes Prometheus appends to the samples of a histogram or summary. The
#: base name is what a domain references, so strip these when matching.
_DERIVED_SUFFIXES = ("_bucket", "_sum", "_count", "_created")


def referenced_metrics(promql: str) -> set[str]:
    """Extract the metric names a PromQL expression reads.

    Deliberately approximate — it strips label matchers, pulls identifiers,
    and drops anything that is a known function or keyword or a bare number.
    Good enough to catch a typo or a renamed metric, which is the job.

    >>> sorted(referenced_metrics('sum(rate(vllm:foo_total[30s])) / vllm:bar'))
    ['vllm:bar', 'vllm:foo_total']
    >>> sorted(referenced_metrics('sum by (pod, namespace) (up)'))
    ['up']
    """
    stripped = promql
    for pattern in (_LABEL_BLOCK_RE, _RANGE_RE, _AGG_CLAUSE_RE):
        stripped = pattern.sub(" ", stripped)
    names = set()
    for token in _IDENT_RE.findall(stripped):
        if token in _NOT_METRICS:
            continue
        names.add(token)
    return names


def base_metric_name(name: str) -> str:
    """Strip a histogram/summary sample suffix to get the declared metric."""
    for suffix in _DERIVED_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def parse_metric_surface(text: str) -> dict[str, str]:
    """Parse Prometheus exposition text into ``{metric_name: type}``.

    Uses ``# TYPE`` lines where present, and falls back to recording bare
    sample names as ``"untyped"`` so a server that omits TYPE headers still
    produces a usable surface.
    """
    surface: dict[str, str] = {}
    for name, kind in _TYPE_RE.findall(text):
        surface[name] = kind

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.match(line)
        if not match:
            continue
        sample = match.group(1)
        base = base_metric_name(sample)
        if base not in surface and sample not in surface:
            surface[base] = "untyped"
    return surface


def fetch_metric_text(url: str, timeout: float = 10.0) -> str:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def check_domain(spec: DomainSpec, surface: dict[str, str]) -> dict[str, list[str]]:
    """Compare a domain's referenced metrics against an observed surface.

    Returns ``{metric_name: [missing_series, ...]}`` for every domain metric
    whose PromQL references something the surface does not expose.
    """
    known = set(surface)
    missing: dict[str, list[str]] = {}
    for metric_name, query in spec.metrics.items():
        absent = sorted(
            ref
            for ref in referenced_metrics(query.promql)
            if base_metric_name(ref) not in known and ref not in known
        )
        if absent:
            missing[metric_name] = absent
    return missing


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Record and verify a server's Prometheus metric surface."""


@cli.command()
@click.option("--url", default="http://localhost:8000/metrics", show_default=True)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("deploy/vllm/metric_surface.json"),
    show_default=True,
)
@click.option("--source", default="", help="Free-text note, e.g. 'vllm 0.11 on L4'.")
def snapshot(url: str, out: Path, source: str) -> None:
    """Scrape URL and write the observed metric surface to OUT."""
    try:
        text = fetch_metric_text(url)
    except requests.exceptions.RequestException as exc:
        raise click.ClickException(f"Cannot scrape {url}: {exc}") from exc

    surface = parse_metric_surface(text)
    if not surface:
        raise click.ClickException(f"{url} returned no parseable metrics.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "url": url,
                "source": source,
                "metrics": dict(sorted(surface.items())),
            },
            indent=2,
        )
        + "\n"
    )

    vllm_metrics = sorted(m for m in surface if m.startswith("vllm:"))
    click.echo(f"Wrote {len(surface)} metrics to {out}")
    if vllm_metrics:
        click.echo(f"  {len(vllm_metrics)} vllm: metrics, including:")
        for name in vllm_metrics[:10]:
            click.echo(f"    {name}  ({surface[name]})")
        if len(vllm_metrics) > 10:
            click.echo(f"    … and {len(vllm_metrics) - 10} more")


@cli.command()
@click.argument("domain_name", type=click.Choice(sorted(DOMAINS)))
@click.option(
    "--surface",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Snapshot file to check against.",
)
@click.option("--url", default=None, help="Scrape a live server instead.")
def check(domain_name: str, surface: Path | None, url: str | None) -> None:
    """Verify DOMAIN_NAME's metrics exist in a snapshot or on a live server."""
    if surface is None and url is None:
        raise click.UsageError("Pass either --surface or --url.")

    if url is not None:
        try:
            observed = parse_metric_surface(fetch_metric_text(url))
        except requests.exceptions.RequestException as exc:
            raise click.ClickException(f"Cannot scrape {url}: {exc}") from exc
        origin = url
    else:
        assert surface is not None
        observed = json.loads(surface.read_text())["metrics"]
        origin = str(surface)

    spec = get_domain(domain_name)
    missing = check_domain(spec, observed)

    total = len(spec.metrics)
    click.echo(f"Domain '{domain_name}': {total} metrics checked against {origin}")

    if not missing:
        click.echo(f"  all {total} resolve against the observed surface  ✓")
        return

    click.echo(f"\n  {len(missing)} of {total} reference metrics that are absent:\n")
    for metric_name, absent in sorted(missing.items()):
        component = spec.metrics[metric_name].component or "(from labels)"
        click.echo(f"    {metric_name}  -> component '{component}'")
        for ref in absent:
            click.echo(f"        missing: {ref}")

    affected = sorted(
        {spec.metrics[m].component for m in missing if spec.metrics[m].component}
    )
    if affected:
        click.echo(
            "\n  Components left with missing signals: " + ", ".join(affected)
        )
    click.echo(
        "\n  A component whose metrics never resolve never goes abnormal, so the\n"
        "  pipeline will blame something else instead. Fix the domain spec or\n"
        "  pin the server version before capturing traces."
    )
    sys.exit(1)


if __name__ == "__main__":
    cli()
