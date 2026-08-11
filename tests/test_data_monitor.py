"""Unit tests for data health monitoring."""

import datetime

import pandas as pd

from demandiq.data.monitor import check_data_health


def test_check_data_health_healthy():
    """Verify healthy dataset passes checks."""
    now = datetime.datetime.now()
    df = pd.DataFrame(
        {
            "date": [now, now - datetime.timedelta(days=1)],
            "city": ["New York", "Chicago"],
            "orders": [100, 200],
        }
    )
    health = check_data_health(df, max_stale_days=3)
    assert health["is_healthy"] is True
    assert len(health["errors"]) == 0


def test_check_data_health_missing_cols():
    """Verify missing columns trigger error."""
    df = pd.DataFrame({"city": ["New York"], "orders": [100]})
    health = check_data_health(df, max_stale_days=3)
    assert health["is_healthy"] is False
    assert len(health["errors"]) == 1
    assert "Missing critical columns" in health["errors"][0]


def test_check_data_health_stale():
    """Verify stale dataset triggers error."""
    old_date = datetime.datetime.now() - datetime.timedelta(days=10)
    df = pd.DataFrame({"date": [old_date], "city": ["New York"], "orders": [100]})
    health = check_data_health(df, max_stale_days=3)
    assert health["is_healthy"] is False
    assert any("Data freshness violation" in err for err in health["errors"])


def test_check_data_health_empty():
    """Verify empty dataset triggers error."""
    df = pd.DataFrame(columns=["date", "city", "orders"])
    health = check_data_health(df)
    assert health["is_healthy"] is False
    assert any("Dataset is completely empty" in err for err in health["errors"])
