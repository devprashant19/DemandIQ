"""Capacity and inventory recommendation engine based on demand forecasts."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CapacityPlanner:
    """Generates capacity/inventory recommendations given demand forecast metrics.

    Uses the forecast mean and upper bounds (p90) to recommend safety stock
    and reorder thresholds to mitigate out-of-stock risks.
    """

    def __init__(self, safety_stock_pct: float = 10.0, default_unit_cost: float = 1.0) -> None:
        """Initialize capacity planner with default risk parameters.

        Args:
            safety_stock_pct: Additional padding percentage on top of the p90 forecast.
            default_unit_cost: Default monetary cost per unit for expected cost estimates.
        """
        self.safety_stock_pct = safety_stock_pct
        self.default_unit_cost = default_unit_cost

    def recommend(
        self,
        forecast_df: pd.DataFrame,
        unit_cost: float | None = None,
    ) -> pd.DataFrame:
        """Generate recommendations for the provided forecast data.

        Args:
            forecast_df: DataFrame containing at least 'date', 'city', 'pred_orders', and 'pred_p90'.
            unit_cost: Optional override for unit cost.

        Returns:
            pd.DataFrame: DataFrame with capacity recommendations appended.
        """
        if forecast_df.empty:
            return pd.DataFrame()

        req_cols = {"date", "city", "pred_orders", "pred_p90"}
        if not req_cols.issubset(forecast_df.columns):
            logger.warning("Forecast DataFrame is missing required columns for capacity planning.")
            return forecast_df.copy()

        cost = unit_cost if unit_cost is not None else self.default_unit_cost

        df = forecast_df.copy()

        # Calculate recommended inventory as p90 + safety stock percentage
        # We use p90 to cover most variance, then add a strategic buffer
        buffer_multiplier = 1.0 + (self.safety_stock_pct / 100.0)
        df["recommended_inventory"] = np.ceil(df["pred_p90"] * buffer_multiplier).astype(int)

        # Reorder point is slightly below recommended but above mean to avoid disruption
        df["reorder_point"] = np.ceil((df["pred_orders"] + df["pred_p90"]) / 2.0).astype(int)

        # Expected baseline cost if meeting the mean forecast exactly
        df["expected_cost"] = df["pred_orders"] * cost

        # Max exposed cost based on recommended inventory limit
        df["max_exposed_cost"] = df["recommended_inventory"] * cost

        # Risk level classification based on the spread between p90 and mean
        # High spread means high uncertainty
        spread_ratio = np.where(
            df["pred_orders"] > 0, (df["pred_p90"] - df["pred_orders"]) / df["pred_orders"], 0.0
        )

        conditions = [
            spread_ratio < 0.15,
            (spread_ratio >= 0.15) & (spread_ratio < 0.30),
            spread_ratio >= 0.30,
        ]
        choices = ["Low", "Medium", "High"]
        df["risk_level"] = np.select(conditions, choices, default="Medium")

        return df
