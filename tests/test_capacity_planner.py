"""Unit tests for the CapacityPlanner module."""

import numpy as np
import pandas as pd
import pytest

from demandiq.recommendations.capacity_planner import CapacityPlanner


@pytest.fixture()
def sample_forecast() -> pd.DataFrame:
    """Create a sample forecast DataFrame."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5),
            "city": ["NYC"] * 5,
            "pred_orders": [100.0, 110.0, 90.0, 120.0, 105.0],
            "pred_p90": [130.0, 140.0, 120.0, 155.0, 135.0],
        }
    )


class TestCapacityPlanner:
    """Tests for the CapacityPlanner class."""

    def test_basic_recommendations(self, sample_forecast: pd.DataFrame) -> None:
        """Should produce all required output columns."""
        planner = CapacityPlanner(safety_stock_pct=10.0, default_unit_cost=2.0)
        result = planner.recommend(sample_forecast)
        assert "recommended_inventory" in result.columns
        assert "reorder_point" in result.columns
        assert "expected_cost" in result.columns
        assert "max_exposed_cost" in result.columns
        assert "risk_level" in result.columns

    def test_empty_dataframe_returns_empty(self) -> None:
        """Should return an empty DataFrame when input is empty."""
        planner = CapacityPlanner()
        result = planner.recommend(pd.DataFrame())
        assert result.empty

    def test_missing_columns_returns_copy(self, sample_forecast: pd.DataFrame) -> None:
        """Should return a copy when required columns are missing."""
        planner = CapacityPlanner()
        df = sample_forecast.drop(columns=["pred_p90"])
        result = planner.recommend(df)
        assert "recommended_inventory" not in result.columns

    def test_recommended_inventory_above_p90(self, sample_forecast: pd.DataFrame) -> None:
        """Recommended inventory must always be >= pred_p90."""
        planner = CapacityPlanner(safety_stock_pct=10.0)
        result = planner.recommend(sample_forecast)
        assert (result["recommended_inventory"] >= result["pred_p90"]).all()

    def test_custom_unit_cost(self, sample_forecast: pd.DataFrame) -> None:
        """Expected cost should scale linearly with unit cost."""
        planner = CapacityPlanner(default_unit_cost=1.0)
        result_1 = planner.recommend(sample_forecast)
        result_2 = planner.recommend(sample_forecast, unit_cost=5.0)
        # Costs should be 5x with unit_cost=5
        np.testing.assert_allclose(
            result_2["expected_cost"].to_numpy(),
            result_1["expected_cost"].to_numpy() * 5,
        )

    def test_risk_levels_valid(self, sample_forecast: pd.DataFrame) -> None:
        """Risk levels should only be Low, Medium, or High."""
        planner = CapacityPlanner()
        result = planner.recommend(sample_forecast)
        assert set(result["risk_level"].unique()).issubset({"Low", "Medium", "High"})
