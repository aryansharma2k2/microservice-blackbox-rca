"""Prometheus metrics client.

Queries Prometheus for the metrics a :class:`~rca_engine.domains.DomainSpec`
declares and returns them as pandas DataFrames or nested dicts ready for
downstream analysis.

The client is domain-agnostic: which PromQL to run and how to attribute each
result series to a component both come from the spec.  It defaults to the
Online Boutique domain, so existing callers behave exactly as before.
"""

import time
from typing import Any, cast

import logging

import numpy as np
import pandas as pd
import requests

from rca_engine.domains import DEFAULT_DOMAIN, DomainSpec
from rca_engine.domains.boutique import METRICS as _BOUTIQUE_METRICS
from rca_engine.domains.boutique import pod_to_service as _pod_to_service

logger = logging.getLogger(__name__)


# Backwards-compatible view of the Boutique PromQL, kept so existing callers
# and notebooks that imported ``QUERIES`` keep working.  The definitions now
# live in rca_engine/domains/boutique.py alongside the rest of that domain.
QUERIES: dict[str, str] = {
    name: query.promql for name, query in _BOUTIQUE_METRICS.items()
}

# ``_pod_to_service`` is re-exported from the boutique domain for callers that
# imported it from here before the domain layer existed.
__all__ = ["QUERIES", "PrometheusMetricsClient"]


class PrometheusMetricsClient:
    """HTTP client for pulling range metrics from a Prometheus instance.

    Parameters
    ----------
    prometheus_url:
        Base URL of the Prometheus server.
    domain:
        Which system's metrics to collect.  Defaults to Online Boutique so
        that callers written before the domain layer keep working unchanged.
    """

    def __init__(
        self,
        prometheus_url: str = "http://localhost:9090",
        domain: DomainSpec | None = None,
    ) -> None:
        self.prometheus_url = prometheus_url.rstrip("/")
        self._range_endpoint = f"{self.prometheus_url}/api/v1/query_range"
        self.domain = domain or DEFAULT_DOMAIN

    # Internal helpers

    def _query_range(
        self,
        query: str,
        start: float,
        end: float,
        step: str,
    ) -> list[dict[str, Any]]:
        """Execute a PromQL range query and return the raw result list.

        Raises ``ConnectionError`` if Prometheus is unreachable.
        Raises ``RuntimeError`` if the API returns a non-success status.
        """
        params = {"query": query, "start": start, "end": end, "step": step}
        try:
            resp = requests.get(self._range_endpoint, params=params, timeout=30)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Cannot reach Prometheus at {self.prometheus_url}: {exc}"
            ) from exc

        body = resp.json()
        if body.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {body.get('error', body)}")

        return body["data"]["result"]  # list of {metric: {...}, values: [[ts, val], ...]}

    # Public API

    def fetch_metrics(
        self,
        start_time: float,
        end_time: float,
        step: str = "1s",
    ) -> pd.DataFrame:
        """Fetch every metric this client's domain declares over the window.

        Args:
            start_time: POSIX timestamp (seconds) for the start of the window.
            end_time:   POSIX timestamp (seconds) for the end of the window.
            step:       Prometheus step string, e.g. ``"5s"``, ``"15s"``.

        Returns:
            DataFrame with columns: [timestamp, pod, service, metric, value].
            The ``service`` column holds the component the series was
            attributed to — a microservice for the Boutique domain, a
            subsystem node for a mechanism graph.  Series the domain cannot
            attribute (e.g. cAdvisor node-level rows with no pod label) are
            dropped.
        """
        rows: list[dict[str, Any]] = []

        logger.info(
            "Fetching Prometheus metrics from %s for domain '%s', "
            "window [%s, %s] step=%s",
            self.prometheus_url,
            self.domain.name,
            start_time,
            end_time,
            step,
        )

        for metric_name, query in self.domain.metrics.items():
            try:
                results = self._query_range(query.promql, start_time, end_time, step)
            except (ConnectionError, RuntimeError) as exc:
                logger.warning("Skipping metric '%s': %s", metric_name, exc)
                continue

            for series in results:
                labels = series["metric"]
                component = self.domain.resolve_component(metric_name, labels)
                if component is None:
                    continue  # series the domain cannot attribute
                for ts_str, val_str in series["values"]:
                    rows.append(
                        {
                            "timestamp": float(ts_str),
                            "pod": labels.get("pod", ""),
                            "service": component,
                            "metric": metric_name,
                            "value": float(val_str),
                        }
                    )

        if not rows:
            return pd.DataFrame(columns=["timestamp", "pod", "service", "metric", "value"])

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        return df.sort_values(["service", "metric", "timestamp"]).reset_index(drop=True)

    def fetch_metric_matrix(
        self,
        start_time: float,
        end_time: float,
        step: str = "1s",
    ) -> dict[str, dict[str, np.ndarray]]:
        """Return metrics as a nested dict for algorithmic processing.

        Returns:
            ``{service_name: {metric_name: np.ndarray of float values}}``

            Values are averaged across pods belonging to the same service so
            that each entry is a single 1-D array aligned to a common time axis.
        """
        df = self.fetch_metrics(start_time, end_time, step)
        if df.empty:
            return {}

        matrix: dict[str, dict[str, np.ndarray]] = {}
        for key, group in df.groupby(["service", "metric"]):
            # Pandas typing exposes group keys as Hashable, so avoid direct tuple unpacking.
            if not isinstance(key, tuple) or len(key) != 2:
                continue
            service, metric = str(key[0]), str(key[1])
            # Average over pods — keeps the time axis consistent
            mean_by_timestamp = cast(
                pd.Series, group.groupby("timestamp")["value"].mean()
            )
            averaged = mean_by_timestamp.sort_index()
            matrix.setdefault(service, {})[metric] = averaged.to_numpy()

        self._log_matrix_summary(matrix)
        return matrix

    def _log_matrix_summary(self, matrix: dict[str, dict[str, np.ndarray]]) -> None:
        total_series = sum(len(metrics) for metrics in matrix.values())
        logger.info(
            "Built metric matrix with %d services and %d service-metric streams",
            len(matrix),
            total_series,
        )


# Demo
if __name__ == "__main__":
    END = time.time()
    START = END - 300  # last 5 minutes

    client = PrometheusMetricsClient()
    logger.info("Fetching metrics from %s …", client.prometheus_url)

    try:
        df = client.fetch_metrics(START, END)
    except ConnectionError as exc:
        logger.error("ERROR: %s", exc)
        raise SystemExit(1)

    if df.empty:
        logger.warning("No data returned — is Prometheus running and scraping the cluster?")
    else:
        logger.info(
            "Rows: %d  |  Services: %d  |  Metrics: %d",
            len(df),
            df['service'].nunique(),
            df['metric'].nunique(),
        )
        logger.info("Per-service, per-metric summary (mean value):")
        summary_mean = cast(pd.Series, df.groupby(["service", "metric"])["value"].mean())
        logger.info("\n%s", summary_mean.unstack(fill_value=0).round(4).to_string())
