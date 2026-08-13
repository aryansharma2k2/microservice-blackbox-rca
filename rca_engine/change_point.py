"""
Layer 1: CUSUM + Bootstrap change-point detection.

Implements the full pipeline from the FChain spec:
  Step 1 — baseline statistics
  Step 2 — standardization
  Step 3 — two-sided CUSUM
  Step 4 — block bootstrap threshold calibration
  Step 5 — threshold crossing detection
  Step 6 — last-reset onset estimation
  Step 7 — confidence scores
"""

import numpy as np
from dataclasses import dataclass, field
import logging
import time
from rca_engine.logger import log_stage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class ChangePointResult:
    """All outputs from Layer 1 for one metric on one VM."""

    change_points: list[int] = field(default_factory=list)
    # Indices into time_series where abnormal changes START (onset estimates)

    directions: list[str] = field(default_factory=list)
    # "up" or "down" for each change point

    confidence_scores: dict[int, float] = field(default_factory=dict)
    # Maps change point index → bootstrap confidence (0.0 to 1.0)

    cusum_up: np.ndarray = field(default_factory=lambda: np.array([]))
    cusum_down: np.ndarray = field(default_factory=lambda: np.array([]))
    cusum_combined: np.ndarray = field(default_factory=lambda: np.array([]))
    # Full CUSUM score trajectories — useful for debugging and Layer 4 rollback

    bootstrap_threshold: float = 0.0
    # The empirically calibrated threshold h

    bootstrap_maxima: list[float] = field(default_factory=list)
    # All M bootstrap maximum scores — passed to Layer 7 confidence scoring

    mu_0: float = 0.0
    sigma_0: float = 1.0
    # Stored for use by Layer 2 (PRESS model needs same baseline stats)

    low_variance_flag: bool = False
    # True if sigma_0 was near zero — downstream layers should note this


# ---------------------------------------------------------------------------
# Step 1: Baseline statistics
# ---------------------------------------------------------------------------

def compute_baseline_stats(
    baseline_data: np.ndarray,
) -> tuple[float, float, bool]:
    """
    Compute mean and std from baseline (normal behavior) data.

    Returns
    -------
    mu_0 : float
    sigma_0 : float
    low_variance_flag : bool
        True if sigma was near zero and was clamped to 1.0
    """
    baseline_data = np.asarray(baseline_data, dtype=float).ravel()

    if len(baseline_data) < 30:
        # Not enough data for reliable statistics — proceed with warning
        logger.warning(
            f"[WARN] Baseline length {len(baseline_data)} < 30. "
            "Statistics may be unreliable."
        )

    mu_0 = float(np.mean(baseline_data))
    sigma_0 = float(np.std(baseline_data, ddof=1))

    low_variance_flag = False
    if sigma_0 < 1e-6:
        # Metric is essentially constant — any change will look dramatic
        sigma_0 = 1.0
        low_variance_flag = True

    return mu_0, sigma_0, low_variance_flag


# ---------------------------------------------------------------------------
# Step 2: Standardization
# ---------------------------------------------------------------------------

def standardize(
    series: np.ndarray,
    mu_0: float,
    sigma_0: float,
) -> np.ndarray:
    """
    z_t = (x_t - mu_0) / sigma_0

    Transforms raw metric values into dimensionless standard-deviation units.
    z_t = 0  → observation at baseline mean (normal)
    z_t = 2  → 2 standard deviations above baseline (suspicious)
    z_t = -2 → 2 standard deviations below baseline (suspicious)
    """
    series = np.asarray(series, dtype=float).ravel()
    return (series - mu_0) / sigma_0


# ---------------------------------------------------------------------------
# Step 3: Two-sided CUSUM
# ---------------------------------------------------------------------------

def run_cusum(
    z: np.ndarray,
    k: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run two-sided CUSUM on a standardized series z.

    Upward CUSUM:   g_up[t]   = max(0, g_up[t-1]   + z[t] - k)
    Downward CUSUM: g_down[t] = max(0, g_down[t-1] - z[t] - k)
    Combined:       g[t]      = max(g_up[t], g_down[t])

    Parameters
    ----------
    z : standardized time series (output of standardize())
    k : reference value, default 0.5
        → k=0.5 means: accumulate evidence only if z_t > 0.5
        → sensitive to shifts of ~1 standard deviation
        → increase k to reduce sensitivity to small shifts

    Returns
    -------
    g_up, g_down, g_combined : arrays of length len(z)
    """
    z = np.asarray(z, dtype=float).ravel()
    W = len(z)

    g_up   = np.zeros(W)
    g_down = np.zeros(W)

    for t in range(1, W):
        g_up[t]   = max(0.0, g_up[t-1]   + z[t] - k)
        g_down[t] = max(0.0, g_down[t-1] - z[t] - k)

    g_combined = np.maximum(g_up, g_down)

    return g_up, g_down, g_combined


# ---------------------------------------------------------------------------
# Step 4: Block bootstrap threshold calibration
# ---------------------------------------------------------------------------

def _generate_block_bootstrap_sample(
    baseline_data: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate one bootstrap sample of the same length as baseline_data
    by sampling contiguous blocks with replacement.

    Why blocks and not individual samples?
    System metrics sampled at 1-second intervals are temporally correlated
    — consecutive CPU values are not independent. IID resampling destroys
    this structure and underestimates the CUSUM score's natural variability,
    producing a threshold that is too low and causing false alarms.
    Block bootstrap preserves short-range temporal dependence.
    """
    n = len(baseline_data)
    num_blocks = int(np.ceil(n / block_size))

    # All possible starting positions for a block
    max_start = max(1, n - block_size + 1)
    starts = rng.integers(0, max_start, size=num_blocks)

    # Build the bootstrap sample by concatenating sampled blocks
    blocks = [baseline_data[s : s + block_size] for s in starts]
    sample = np.concatenate(blocks)

    # Trim to exactly n samples
    return sample[:n]


def bootstrap_threshold(
    baseline_data: np.ndarray,
    mu_0: float,
    sigma_0: float,
    k: float = 0.5,
    n_replicates: int = 1000,
    block_size: int = 10,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, list[float]]:
    """
    Empirically calibrate the CUSUM threshold using block bootstrap.

    Algorithm
    ---------
    Repeat n_replicates times:
        1. Generate a synthetic "normal" dataset by block-resampling baseline
        2. Standardize using the SAME mu_0, sigma_0 (not recomputed)
           — critical: we want to measure variability of the detector
             under the null, not variability of the data itself
        3. Run CUSUM on the synthetic dataset
        4. Record the maximum CUSUM score reached

    Set threshold h = confidence-th percentile of recorded maxima.

    Interpretation
    --------------
    h at 95th percentile means:
        Under genuinely normal behavior, the CUSUM score exceeds h
        only ~5% of the time.
        If the real stream exceeds h, something unusual is happening.

    Parameters
    ----------
    baseline_data  : array of normal behavior samples
    mu_0, sigma_0  : baseline statistics (from compute_baseline_stats)
    k              : same k used in run_cusum
    n_replicates   : number of bootstrap samples (default 1000)
    block_size     : contiguous block length for resampling (default 10)
    confidence     : percentile level for threshold (default 0.95)
    seed           : random seed for reproducibility

    Returns
    -------
    h              : float, the calibrated threshold
    maxima         : list of floats, all bootstrap maximum scores
                     (needed for confidence score computation in Step 7)
    """
    baseline_data = np.asarray(baseline_data, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    maxima = []

    for _ in range(n_replicates):
        # Step 1: Generate synthetic normal dataset
        synthetic = _generate_block_bootstrap_sample(
            baseline_data, block_size, rng
        )

        # Step 2: Standardize using SAME baseline statistics
        # Do NOT recompute mu/sigma per replicate
        z_synth = standardize(synthetic, mu_0, sigma_0)

        # Step 3: Run CUSUM
        _, _, g_combined = run_cusum(z_synth, k)

        # Step 4: Record maximum score
        maxima.append(float(np.max(g_combined)))

    # Threshold = high quantile of bootstrap maxima
    h = float(np.percentile(maxima, confidence * 100))
    if h == 0.0:
        h = 1e-5

    return h, maxima


# ---------------------------------------------------------------------------
# Step 5 + 6: Detect crossings and estimate onsets via last-reset heuristic
# ---------------------------------------------------------------------------

def _find_onset_via_last_reset(
    alarm_index: int,
    detector: np.ndarray,
) -> int:
    """
    Given that CUSUM crossed the threshold at alarm_index,
    walk backward to find the last time *detector* was exactly 0.

    This is the "last reset heuristic":
        The CUSUM score resets to 0 when accumulated evidence
        becomes negative — i.e., when the metric briefly returns
        to normal. The last reset before the alarm marks where
        the current run of abnormal evidence began.

    *detector* must be the one-sided score that actually fired —
    ``g_up`` for an upward change, ``g_down`` for a downward one.
    Passing the combined score instead silently breaks the heuristic:
    ``g_combined = max(g_up, g_down)`` is zero only when *both*
    detectors are simultaneously zero, so under a sustained drift in
    either direction it never resets and every onset collapses to 0.

    Returns onset_index = last_zero + 1
    (one step after the last reset = where accumulation restarted)
    """
    # Walk backward from alarm_index
    for j in range(alarm_index - 1, -1, -1):
        if detector[j] == 0.0:
            return j + 1  # One step after the last zero

    # If no zero found, the change started at or before index 0
    return 0


def detect_crossings_and_onsets(
    g_up: np.ndarray,
    g_down: np.ndarray,
    g_combined: np.ndarray,
    h: float,
    onset_estimator: str = "last_reset",
) -> tuple[list[int], list[str]]:
    """
    Find all threshold crossings in g_combined,
    group contiguous crossings into single events,
    and estimate the onset of each event using the last-reset heuristic.

    Parameters
    ----------
    g_up, g_down, g_combined : CUSUM score arrays from run_cusum()
    h : bootstrap-calibrated threshold

    Returns
    -------
    onsets     : list of int — onset index for each distinct change event
    directions : list of str — "up" or "down" for each event
    """
    W = len(g_combined)
    onsets = []
    directions = []

    t = 0
    while t < W:
        if g_combined[t] >= h:
            # Found a threshold crossing — start of a new event
            alarm_time = t

            # Determine direction from which detector triggered
            if g_up[t] >= g_down[t]:
                direction = "up"
            else:
                direction = "down"

            # Skip over the entire contiguous crossing run
            # (multiple consecutive indices above threshold = one event)
            while t < W and g_combined[t] >= h:
                t += 1

            # Estimate the true onset using the last-reset heuristic, walking
            # back through the detector that fired rather than the combined
            # score — see _find_onset_via_last_reset.
            if onset_estimator == "crossing":
                # Where the evidence became conclusive. A consistent
                # definition across metrics, which is what cross-metric onset
                # ordering needs. Biased later for small effects, since they
                # take longer to cross — the effect-size gate handles those.
                onset = alarm_time
            else:
                onset = _find_onset_via_last_reset(
                    alarm_time, g_up if direction == "up" else g_down
                )

            onsets.append(onset)
            directions.append(direction)
        else:
            t += 1

    return onsets, directions


# ---------------------------------------------------------------------------
# Step 7: Confidence scores
# ---------------------------------------------------------------------------

def compute_confidence_scores(
    onsets: list[int],
    g_combined: np.ndarray,
    bootstrap_maxima: list[float],
    peak_window: int = 10,
) -> dict[int, float]:
    """
    For each detected onset, compute a confidence score:

        confidence = proportion of bootstrap maxima that are LESS THAN
                     the observed peak CUSUM score near this onset

    Interpretation
    --------------
    confidence = 0.97 → 97% of normal bootstrap runs never reached
                         this CUSUM score. Very likely a real change.
    confidence > 0.95 → reliable
    confidence 0.90-0.95 → marginal, flag but keep
    confidence < 0.90 → weak evidence

    Parameters
    ----------
    onsets           : onset indices from detect_crossings_and_onsets()
    g_combined       : full CUSUM score array
    bootstrap_maxima : list of max scores from bootstrap_threshold()
    peak_window      : how many indices after onset to look for peak score
    """
    maxima_array = np.array(bootstrap_maxima)
    confidence_scores = {}

    for onset in onsets:
        # Get the peak CUSUM score in a small window after the onset
        end = min(onset + peak_window, len(g_combined))
        observed_peak = float(np.max(g_combined[onset:end]))

        # Proportion of bootstrap maxima below this observed peak
        confidence = float(np.mean(maxima_array < observed_peak))
        confidence_scores[onset] = confidence

    return confidence_scores


# ---------------------------------------------------------------------------
# Main entry point: full Layer 1 pipeline
# ---------------------------------------------------------------------------

def run_layer1(
    time_series: np.ndarray,
    baseline_data: np.ndarray,
    k: float = 0.5,
    n_bootstrap: int = 1000,
    block_size: int = 10,
    confidence_level: float = 0.95,
    seed: int | None = None,
    start_time: float | None = None,
    logs: list[dict] | None = None,
    onset_estimator: str = "last_reset",
) -> ChangePointResult:
    """
    Full Layer 1 pipeline for one metric on one VM.

    Run this independently for each of the 7 metrics per VM.
    Results feed into Layer 2 (PRESS prediction error filtering).

    Parameters
    ----------
    time_series      : look-back window of raw metric values.
    baseline_data    : longer window of known normal behavior, length ≥300
    k                : CUSUM reference value (default 0.5)
    n_bootstrap      : number of bootstrap replicates (default 1000)
    block_size       : block size for block bootstrap (default 10)
    confidence_level : bootstrap percentile for threshold (default 0.95)
    seed             : random seed for reproducibility

    Returns
    -------
    ChangePointResult with all outputs needed by downstream layers
    """
    # Log timing for Layer 1
    if start_time is not None and logs is not None:
        stage_start = time.time()
        log_stage("LAYER1_CUSUM", __file__, start_time, time.time(), logs)

    time_series   = np.asarray(time_series,  dtype=float).ravel()
    baseline_data = np.asarray(baseline_data, dtype=float).ravel()

    # Handle NaN/Inf in time series
    time_series = _sanitize_series(time_series)

    # ── Step 1: Baseline statistics ──────────────────────────────────────
    mu_0, sigma_0, low_var = compute_baseline_stats(baseline_data)

    # ── Step 2: Standardize ──────────────────────────────────────────────
    z = standardize(time_series, mu_0, sigma_0)

    # ── Step 3: Two-sided CUSUM ──────────────────────────────────────────
    g_up, g_down, g_combined = run_cusum(z, k)

    # ── Step 4: Bootstrap threshold ──────────────────────────────────────
    h, bootstrap_maxima = bootstrap_threshold(
        baseline_data,
        mu_0, sigma_0,
        k=k,
        n_replicates=n_bootstrap,
        block_size=block_size,
        confidence=confidence_level,
        seed=seed,
    )

    # ── Steps 5+6: Detect crossings and estimate onsets ──────────────────
    onsets, directions = detect_crossings_and_onsets(
        g_up, g_down, g_combined, h, onset_estimator=onset_estimator
    )

    # ── Step 7: Confidence scores ─────────────────────────────────────────
    confidence_scores = compute_confidence_scores(
        onsets, g_combined, bootstrap_maxima
    )

    # ── Edge case: change started before window ───────────────────────────
    # If onset == 0 and g_combined[0] is already high,
    # the fault predates the look-back window.
    # Flag this in the result for upstream handling.

    return ChangePointResult(
        change_points      = onsets,
        directions         = directions,
        confidence_scores  = confidence_scores,
        cusum_up           = g_up,
        cusum_down         = g_down,
        cusum_combined     = g_combined,
        bootstrap_threshold= h,
        bootstrap_maxima   = bootstrap_maxima,
        mu_0               = mu_0,
        sigma_0            = sigma_0,
        low_variance_flag  = low_var,
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _sanitize_series(series: np.ndarray) -> np.ndarray:
    """
    Replace NaN and Inf values with linear interpolation from neighbors.
    Handles edge cases where first or last values are NaN.
    """
    series = series.copy()
    bad = ~np.isfinite(series)

    if not np.any(bad):
        return series

    indices = np.arange(len(series))
    good = ~bad

    if not np.any(good):
        # Entire series is NaN — return zeros with warning
        logger.warning("Entire time series is NaN/Inf. Returning zeros.")
        return np.zeros_like(series)

    # Linear interpolation over bad indices
    series[bad] = np.interp(indices[bad], indices[good], series[good])
    return series