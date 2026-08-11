"""Data quality monitoring for freshness and schema drift."""

import datetime
from typing import Any

import pandas as pd


def check_data_health(df: pd.DataFrame, max_stale_days: int = 3) -> dict[str, Any]:
    """Check dataset for freshness and expected schema columns.

    Args:
        df: Input dataframe containing orders data.
        max_stale_days: Maximum allowed days between now and the most recent record.

    Returns:
        dict: A dictionary containing health status ('is_healthy', 'errors', 'warnings').
    """
    health: dict[str, Any] = {
        "is_healthy": True,
        "errors": [],
        "warnings": [],
    }

    # 0. Volume Check
    if len(df) == 0:
        health["is_healthy"] = False
        health["errors"].append("Dataset is completely empty.")
        return health

    # 1. Schema Drift Check
    expected_cols = {"date", "city", "orders"}
    missing = expected_cols - set(df.columns)
    if missing:
        health["is_healthy"] = False
        health["errors"].append(f"Missing critical columns: {missing}")
        return health  # Stop further checks if critical columns are missing

    # 2. Freshness Check
    max_date = df["date"].max()
    now = datetime.datetime.now()

    # Convert both to naive or aware if needed, assuming naive here based on typical load
    if hasattr(max_date, "tzinfo") and max_date.tzinfo is not None:
        max_date = max_date.replace(tzinfo=None)

    days_stale = (now - max_date).days

    if days_stale > max_stale_days:
        health["is_healthy"] = False
        health["errors"].append(
            f"Data freshness violation: Most recent record is {days_stale} days old (max allowed: {max_stale_days})."
        )
    elif days_stale > 0:
        health["warnings"].append(
            f"Data is {days_stale} days old, but within acceptable threshold."
        )

    return health
