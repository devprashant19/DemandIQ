"""Unit tests for the DriftDetector module."""

import numpy as np
import pandas as pd

from demandiq.monitoring.drift_detector import detect_data_drift, detect_performance_drift


class TestDetectPerformanceDrift:
    """Tests for the detect_performance_drift function."""

    def test_no_drift_below_threshold(self) -> None:
        """Should return False when MAPE is below threshold."""
        series = pd.Series([5.0, 5.0, 5.0])
        assert detect_performance_drift(series, threshold=8.0, streak=3) is False

    def test_drift_detected_above_threshold(self) -> None:
        """Should return True when MAPE exceeds threshold for streak days."""
        series = pd.Series([10.0, 11.0, 12.0])
        assert detect_performance_drift(series, threshold=8.0, streak=3) is True

    def test_no_drift_short_series(self) -> None:
        """Should return False when series is shorter than streak."""
        series = pd.Series([15.0, 15.0])
        assert detect_performance_drift(series, threshold=8.0, streak=3) is False

    def test_partial_streak_no_drift(self) -> None:
        """Should return False when only part of the tail exceeds threshold."""
        series = pd.Series([5.0, 10.0, 10.0, 5.0])
        assert detect_performance_drift(series, threshold=8.0, streak=3) is False

    def test_custom_streak(self) -> None:
        """Should respect custom streak parameter."""
        series = pd.Series([10.0, 10.0])
        assert detect_performance_drift(series, threshold=8.0, streak=2) is True


class TestDetectDataDrift:
    """Tests for the detect_data_drift function."""

    def test_no_drift_identical_distributions(self) -> None:
        """PSI should be near zero for identical distributions."""
        rng = np.random.default_rng(42)
        baseline = pd.DataFrame({"city": ["NYC"] * 100, "orders": rng.poisson(50, 100)})
        new_data = pd.DataFrame({"city": ["NYC"] * 100, "orders": rng.poisson(50, 100)})
        results = detect_data_drift(new_data, baseline)
        assert "NYC" in results
        assert results["NYC"] < 0.5

    def test_drift_detected_different_distributions(self) -> None:
        """PSI should be high for very different distributions."""
        rng = np.random.default_rng(42)
        baseline = pd.DataFrame({"city": ["NYC"] * 100, "orders": rng.poisson(10, 100)})
        new_data = pd.DataFrame({"city": ["NYC"] * 100, "orders": rng.poisson(200, 100)})
        results = detect_data_drift(new_data, baseline)
        assert "NYC" in results
        assert results["NYC"] > 0.2

    def test_missing_city_in_new_data(self) -> None:
        """Should skip cities not in new_df."""
        baseline = pd.DataFrame({"city": ["NYC"] * 50, "orders": [10] * 50})
        new_data = pd.DataFrame({"city": ["Chicago"] * 50, "orders": [10] * 50})
        results = detect_data_drift(new_data, baseline)
        assert "NYC" not in results

    def test_returns_dict(self) -> None:
        """Should return a dict with city keys."""
        baseline = pd.DataFrame({"city": ["NYC"] * 30, "orders": [10] * 30})
        new_data = pd.DataFrame({"city": ["NYC"] * 30, "orders": [10] * 30})
        results = detect_data_drift(new_data, baseline)
        assert isinstance(results, dict)
