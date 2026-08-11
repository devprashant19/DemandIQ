"""Unit tests for model report card generation."""

import numpy as np
import pandas as pd

from demandiq.models.model_card import generate_model_report


class DummyForecaster:
    """Mock forecaster class for testing model report generation."""

    def __init__(self):
        """Initialize mock forecaster with fitted state."""
        self.is_fitted = True

    def predict(self, df):
        """Mock prediction returning 90% of actual orders."""
        return df["orders"].to_numpy() * 0.9  # Mock predictions

    def predict_intervals(self, df):
        """Mock prediction intervals."""
        preds = self.predict(df)
        return {"mean": preds, "p10": preds * 0.8, "p90": preds * 1.2}


class DummyDetector:
    """Mock anomaly detector class."""

    def predict(self, residuals):
        """Mock anomaly prediction based on residual threshold."""
        return np.where(np.abs(residuals) > 10, True, False)


def test_generate_model_report():
    """Verify report generation logic."""
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2024-01-01", periods=5),
            "city": ["New York", "New York", "Chicago", "Chicago", "Chicago"],
            "orders": [100, 110, 200, 210, 190],
        }
    )

    forecaster = DummyForecaster()
    detector = DummyDetector()

    report = generate_model_report(df, forecaster, detector)

    assert "global" in report
    assert "cities" in report
    assert "calibration" in report
    assert "anomaly" in report
    assert "metadata" in report

    assert report["calibration"]["target_coverage"] == 0.80
    assert "New York" in report["cities"]
    assert "Chicago" in report["cities"]
