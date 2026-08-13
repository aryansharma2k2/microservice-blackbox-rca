"""Structured logging and timing for RCA pipeline stages."""

import logging

logger = logging.getLogger(__name__)


def log_stage(stage, file, start_time, current_time, logs):
    """Record the first occurrence of a pipeline stage.

    Goes through logging rather than print: the evaluation calls the pipeline
    once per captured run, and stage banners on stdout drown the results
    table. Callers that want the timings still get them in ``logs``.
    """
    if any(entry["stage"] == stage for entry in logs):
        return

    duration = current_time - start_time

    logs.append({
        "stage": stage,
        "timestamp": current_time,
        "since_start_seconds": duration,
    })

    logger.info("[RCA] %-25s (+%.3fs)", stage, duration)