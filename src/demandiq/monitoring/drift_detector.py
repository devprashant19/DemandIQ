"""Data and performance drift detection heuristics."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_performance_drift(
    rolling_mape_series: pd.Series[Any], threshold: float = 8.0, streak: int = 3
) -> bool:
    """Detect if rolling MAPE has breached a threshold for a consecutive number of days.

    Args:
        rolling_mape_series: Time series of rolling MAPE percentages.
        threshold: The threshold beyond which performance is considered degraded.
        streak: Number of consecutive days the threshold must be breached to flag drift.

    Returns:
        bool: True if drift is detected, False otherwise.
    """
    if len(rolling_mape_series) < streak:
        return False

    recent_values = rolling_mape_series.tail(streak).to_numpy()
    drift = bool(np.all(recent_values > threshold))

    if drift:
        logger.warning(
            "Performance Drift Detected: MAPE exceeded %.1f%% for %d consecutive days. "
            "Recent values: %s",
            threshold,
            streak,
            recent_values,
        )
    return drift


def detect_data_drift(
    new_df: pd.DataFrame, baseline_df: pd.DataFrame, threshold: float = 0.2
) -> dict[str, float]:
    """Calculate Population Stability Index (PSI) to detect data distribution drift in orders.

    Args:
        new_df: The recent data to evaluate.
        baseline_df: The historical baseline data.
        threshold: The PSI threshold indicating significant drift.

    Returns:
        dict: A mapping from city to calculated PSI value.
    """
    psi_results = {}

    for city in new_df["city"].unique():
        new_city_df = new_df[new_df["city"] == city]
        base_city_df = baseline_df[baseline_df["city"] == city]

        if len(new_city_df) == 0 or len(base_city_df) == 0:
            continue

        # Define 10 quantiles based on baseline distribution
        bins = np.percentile(base_city_df["orders"].to_numpy(), np.linspace(0, 100, 11))

        # Ensure bins are unique (e.g., if lots of 0s)
        bins = np.unique(bins)
        if len(bins) < 2:
            bins = np.array([0, np.inf])
        else:
            bins[0] = -np.inf
            bins[-1] = np.inf

        base_counts, _ = np.histogram(base_city_df["orders"].to_numpy(), bins=bins)
        new_counts, _ = np.histogram(new_city_df["orders"].to_numpy(), bins=bins)

        base_pct = base_counts / max(len(base_city_df), 1)
        new_pct = new_counts / max(len(new_city_df), 1)

        # Add small epsilon to avoid division by zero or log(0)
        base_pct = np.maximum(base_pct, 1e-6)
        new_pct = np.maximum(new_pct, 1e-6)

        psi = np.sum((new_pct - base_pct) * np.log(new_pct / base_pct))
        psi_results[str(city)] = float(psi)

        if psi > threshold:
            logger.warning(
                "Data Drift Detected in %s: PSI = %.3f (Threshold: %.3f)", city, psi, threshold
            )

    return psi_results
