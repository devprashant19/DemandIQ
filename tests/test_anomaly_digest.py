"""Unit tests for anomaly alert digest generation."""

import datetime

import pandas as pd

from demandiq.reports.anomaly_digest import generate_markdown_digest


def test_generate_markdown_digest_empty():
    """Verify empty dataframe handling."""
    df = pd.DataFrame()
    digest = generate_markdown_digest(df, "New York")
    assert "All Clear" in digest
    assert "No operational anomalies detected" in digest


def test_generate_markdown_digest_with_events():
    """Verify markdown output for anomalies."""
    df = pd.DataFrame(
        {
            "date": [datetime.datetime.now(), datetime.datetime.now() - datetime.timedelta(days=1)],
            "anomaly_type": ["surge", "dip"],
            "anomaly_score": [0.8, 0.9],
            "orders": [1500, 500],
            "pred_orders": [1000, 1000],
            "residuals": [500, -500],
        }
    )

    digest = generate_markdown_digest(df, "Chicago")
    assert "🚨 Anomaly Alert Digest: Chicago" in digest
    assert "**2** total anomaly events" in digest
    assert "SURGE" in digest
    assert "DIP" in digest
    assert "Severity Score" in digest
