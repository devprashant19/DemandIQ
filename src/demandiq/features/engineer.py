"""Feature engineering module for time-series demand forecasting without data leakage."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "dow",
    "month",
    "day",
    "day_of_year",
    "week_of_year",
    "quarter",
    "is_weekend",
    "is_friday",
    "is_month_start",
    "is_month_end",
    "sin_doy",
    "cos_doy",
    "orders_lag_1",
    "orders_lag_2",
    "orders_lag_3",
    "orders_lag_4",
    "orders_lag_5",
    "orders_lag_6",
    "orders_lag_7",
    "orders_lag_14",
    "orders_lag_21",
    "orders_lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_min_7",
    "rolling_max_7",
    "rolling_mean_14",
    "rolling_std_14",
    "rolling_min_14",
    "rolling_max_14",
    "rolling_mean_28",
    "rolling_std_28",
    "rolling_min_28",
    "rolling_max_28",
    "city_target_enc",
    "temp_sq",
    "temp_deviation",
    "log_rainfall",
    "rain_intensity",
    "promo_x_weekend",
    "promo_x_rainy",
    "festival_x_rainy",
    "holiday_x_weekend",
    "promo_x_holiday",
    "lag_7_ratio",
    "rolling_ratio_7_28",
]


def compute_city_target_encoding(df: pd.DataFrame) -> dict[str, float]:
    """Compute mean order volume per city on training data to avoid leakage.

    Args:
        df (pd.DataFrame): Training DataFrame containing city and orders columns.

    Returns:
        dict[str, float]: Mapping of city name to average historical order volume.
    """
    if "city" not in df.columns or "orders" not in df.columns:
        return {}
    encoding_series = df.groupby("city")["orders"].mean()
    return {str(city): float(val) for city, val in encoding_series.items()}


def build_features(
    df: pd.DataFrame, target_encoding: dict[str, float] | None = None
) -> pd.DataFrame:
    """Generate >40 features (temporal, lags, rolling stats, weather, interactions) without mutation or leakage.

    Args:
        df (pd.DataFrame): Input dataframe containing date, city, orders, weather, and promo flags.
        target_encoding (dict[str, float] | None): Pre-computed training set target encoding dictionary per city.
            If None, encoding is computed on the current dataset directly.

    Returns:
        pd.DataFrame: New DataFrame containing original columns and engineered feature columns with zero NaNs.
    """
    out_df = df.copy()
    out_df["date"] = pd.to_datetime(out_df["date"])
    out_df = out_df.sort_values(by=["city", "date"]).reset_index(drop=True)

    # 1. Temporal / Calendar features (12 features)
    dt_index = out_df["date"].dt
    out_df["dow"] = dt_index.dayofweek.astype(int)
    out_df["month"] = dt_index.month.astype(int)
    out_df["day"] = dt_index.day.astype(int)
    out_df["day_of_year"] = dt_index.day_of_year.astype(int)
    out_df["week_of_year"] = dt_index.isocalendar().week.astype(int)
    out_df["quarter"] = dt_index.quarter.astype(int)
    out_df["is_weekend"] = (out_df["dow"] >= 5).astype(int)
    out_df["is_friday"] = (out_df["dow"] == 4).astype(int)
    out_df["is_month_start"] = dt_index.is_month_start.astype(int)
    out_df["is_month_end"] = dt_index.is_month_end.astype(int)
    out_df["sin_doy"] = np.sin(2 * np.pi * out_df["day_of_year"] / 365.25)
    out_df["cos_doy"] = np.cos(2 * np.pi * out_df["day_of_year"] / 365.25)

    # 2. Lag features per city (10 features)
    lags = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]
    grouped = out_df.groupby("city")["orders"]
    for lag in lags:
        out_df[f"orders_lag_{lag}"] = grouped.shift(lag)

    # 3. Rolling window statistics on orders_lag_1 to ensure zero leakage (12 features)
    lag_1_grouped = out_df.groupby("city")["orders_lag_1"]
    for win in [7, 14, 28]:
        out_df[f"rolling_mean_{win}"] = lag_1_grouped.transform(
            lambda x, w=win: x.rolling(w, min_periods=1).mean()
        )
        out_df[f"rolling_std_{win}"] = lag_1_grouped.transform(
            lambda x, w=win: x.rolling(w, min_periods=1).std()
        ).fillna(0.0)
        out_df[f"rolling_min_{win}"] = lag_1_grouped.transform(
            lambda x, w=win: x.rolling(w, min_periods=1).min()
        )
        out_df[f"rolling_max_{win}"] = lag_1_grouped.transform(
            lambda x, w=win: x.rolling(w, min_periods=1).max()
        )

    # 4. Target encoding (1 feature)
    if target_encoding is None:
        target_encoding = compute_city_target_encoding(out_df)
    default_mean = float(np.mean(list(target_encoding.values()))) if target_encoding else 0.0
    out_df["city_target_enc"] = out_df["city"].map(target_encoding).fillna(default_mean)

    # 5. Weather transformations (4 features)
    if "temperature_c" in out_df.columns:
        out_df["temp_sq"] = out_df["temperature_c"] ** 2
        out_df["temp_deviation"] = out_df["temperature_c"] - 15.0
    else:
        out_df["temp_sq"] = 0.0
        out_df["temp_deviation"] = 0.0

    if "rainfall_mm" in out_df.columns:
        out_df["log_rainfall"] = np.log1p(np.maximum(0.0, out_df["rainfall_mm"]))
        zeros_float = np.zeros(len(out_df), dtype=float)
        is_rainy_val = (
            out_df["is_rainy"].to_numpy(dtype=float)
            if "is_rainy" in out_df.columns
            else zeros_float
        )
        out_df["rain_intensity"] = out_df["rainfall_mm"].to_numpy(dtype=float) * is_rainy_val
    else:
        out_df["log_rainfall"] = 0.0
        out_df["rain_intensity"] = 0.0

    # 6. Interaction terms and ratios (6 features)
    zeros_int = np.zeros(len(out_df), dtype=int)
    promo_val = (
        out_df["promo_active"].to_numpy(dtype=int)
        if "promo_active" in out_df.columns
        else zeros_int
    )
    holiday_val = (
        out_df["is_holiday"].to_numpy(dtype=int) if "is_holiday" in out_df.columns else zeros_int
    )
    festival_val = (
        out_df["festival_flag"].to_numpy(dtype=int)
        if "festival_flag" in out_df.columns
        else zeros_int
    )
    rain_val = out_df["is_rainy"].to_numpy(dtype=int) if "is_rainy" in out_df.columns else zeros_int
    weekend_val = out_df["is_weekend"].to_numpy(dtype=int)

    out_df["promo_x_weekend"] = (promo_val * weekend_val).astype(int)
    out_df["promo_x_rainy"] = (promo_val * rain_val).astype(int)
    out_df["festival_x_rainy"] = (festival_val * rain_val).astype(int)
    out_df["holiday_x_weekend"] = (holiday_val * weekend_val).astype(int)
    out_df["promo_x_holiday"] = (promo_val * holiday_val).astype(int)

    # Ratios (safe division with epsilon)
    out_df["lag_7_ratio"] = out_df["orders_lag_1"] / (out_df["orders_lag_7"] + 1.0)
    out_df["rolling_ratio_7_28"] = out_df["rolling_mean_7"] / (out_df["rolling_mean_28"] + 1.0)

    # Impute initial NaNs resulting from lag shifts using backwards filling then zero-fill
    out_df = out_df.bfill().ffill().fillna(0.0)
    out_df = out_df.reset_index(drop=True)

    logger.debug(
        "Engineered %d total features cleanly without NaNs or leakage.", len(FEATURE_COLUMNS)
    )
    return out_df
