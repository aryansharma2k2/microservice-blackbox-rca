"""CLI entry point: ``python -m rca_engine``.

Runs the FChain pipeline against a live Prometheus over a recent time window
and prints the ranked suspect list.  This is the "something just went wrong,
tell me what" invocation — it does not inject anything, it only diagnoses.

The window is expressed relative to now::

    |<-- baseline -->|<-- fault -->|
    t0               t1            now

so ``--baseline 300 --fault 120`` learns normal behaviour from the 5 minutes
ending 2 minutes ago, then analyses the last 2 minutes against it.

Examples
--------
    python -m rca_engine
    python -m rca_engine --baseline 600 --fault 180
    python -m rca_engine --propagation-map calibration/propagation_delays.json
    python -m rca_engine --json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone

import click
import requests

from rca_engine import fault_chain
from rca_engine.domains import DEFAULT_DOMAIN, DOMAINS, get_domain
from rca_engine.metrics_client import PrometheusMetricsClient


#: How to read each verdict, in one line, for the human at the terminal.
_VERDICT_BLURB = {
    fault_chain.VERDICT_PATHOLOGY: (
        "pathology — an internal component moved first; the ranking below "
        "names what to fix"
    ),
    fault_chain.VERDICT_CAPACITY: (
        "capacity — an input moved before anything internal did; the system is "
        "being asked for more than it can serve, nothing is broken"
    ),
    fault_chain.VERDICT_EXTERNAL: (
        "external — every component moved together; no single one explains it"
    ),
    fault_chain.VERDICT_NO_ANOMALY: "no anomaly detected in this window",
}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _require_prometheus(url: str) -> None:
    """Fail fast with one clean message if Prometheus is unreachable.

    ``fetch_metrics`` catches connection errors per-metric and logs a warning
    for each, so without this probe an unreachable server produces a wall of
    stack traces before the real error.
    """
    try:
        requests.get(f"{url.rstrip('/')}/-/healthy", timeout=5).raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise click.ClickException(
            f"Cannot reach Prometheus at {url}: {exc}\n"
            "Start it with `bash infra/deploy-monitoring.sh`, or pass "
            "--prometheus-url."
        ) from exc


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--prometheus-url",
    default="http://localhost:9090",
    show_default=True,
    help="Base URL of the Prometheus server to query.",
)
@click.option(
    "--baseline",
    "baseline_seconds",
    default=300,
    show_default=True,
    type=int,
    help="Length of the clean baseline window, in seconds.",
)
@click.option(
    "--fault",
    "fault_seconds",
    default=120,
    show_default=True,
    type=int,
    help="Length of the fault window to analyse, ending now, in seconds.",
)
@click.option(
    "--step",
    default=1.0,
    show_default=True,
    type=float,
    help="Prometheus query step in seconds.",
)
@click.option(
    "--propagation-map",
    "propagation_map_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Optional calibrated per-edge propagation delays (Layer 7).",
)
@click.option(
    "--domain",
    "domain_name",
    type=click.Choice(sorted(DOMAINS)),
    default=DEFAULT_DOMAIN.name,
    show_default=True,
    help="Which system to diagnose.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of a table.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(
    prometheus_url: str,
    baseline_seconds: int,
    fault_seconds: int,
    step: float,
    propagation_map_path: str | None,
    domain_name: str,
    as_json: bool,
    verbose: bool,
) -> None:
    """Diagnose the most recent anomaly window from Prometheus telemetry."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    _require_prometheus(prometheus_url)
    spec = get_domain(domain_name)

    now = time.time()
    fault_window = (now - fault_seconds, now)
    baseline_window = (fault_window[0] - baseline_seconds, fault_window[0])

    client = PrometheusMetricsClient(prometheus_url, domain=spec)
    # pinpoint() expects arrays aligned to baseline_window[0] and spanning
    # through the end of the fault window, so fetch the whole range at once.
    metric_matrix = client.fetch_metric_matrix(
        baseline_window[0], fault_window[1], step=f"{step:g}s"
    )

    if not metric_matrix:
        raise click.ClickException(
            f"No metrics returned from {prometheus_url} for domain "
            f"'{domain_name}'. Is Prometheus running and scraping the target? "
            "Check the metric surface with:\n"
            f"  python -m rca_engine.scripts.discover_metrics check {domain_name} "
            "--url <server>/metrics"
        )

    report = fault_chain.pinpoint_report(
        metric_matrix=metric_matrix,
        baseline_window=baseline_window,
        fault_window=fault_window,
        step_seconds=step,
        propagation_map_path=propagation_map_path,
        domain=spec,
    )
    ranked = report.ranked

    if as_json:
        click.echo(
            json.dumps(
                {
                    "domain": report.domain,
                    "verdict": report.verdict,
                    "baseline_window": list(baseline_window),
                    "fault_window": list(fault_window),
                    "components_observed": len(metric_matrix),
                    "exogenous_drivers": report.exogenous_drivers,
                    "ranked": ranked,
                },
                indent=2,
            )
        )
        return

    click.echo(
        f"\nDomain   : {report.domain}"
        f"\nBaseline : {_iso(baseline_window[0])} → {_iso(baseline_window[1])}"
        f"\nFault    : {_iso(fault_window[0])} → {_iso(fault_window[1])}"
        f"\nObserved : {len(metric_matrix)} components"
        f"\nVerdict  : {_VERDICT_BLURB.get(report.verdict, report.verdict)}\n"
    )

    if report.exogenous_drivers:
        click.echo(
            "Exogenous drivers (inputs, not defects): "
            + ", ".join(report.exogenous_drivers)
            + "\n"
        )

    if not ranked:
        click.echo("Nothing was abnormal in this window.")
        return

    click.echo(f"{'RANK':<6}{'COMPONENT':<28}{'ONSET':<12}{'CONF':<8}ABNORMAL METRICS")
    click.echo("-" * 96)
    for entry in ranked:
        onset = entry.get("onset_time")
        offset = f"+{onset - fault_window[0]:.0f}s" if isinstance(onset, (int, float)) else "-"
        confidence = entry.get("confidence")
        conf = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "-"
        metrics = ", ".join(entry.get("abnormal_metrics") or []) or "-"
        click.echo(
            f"{entry.get('rank', '?'):<6}{entry.get('service', '?'):<28}"
            f"{offset:<12}{conf:<8}{metrics}"
        )
    click.echo()


if __name__ == "__main__":
    main()
