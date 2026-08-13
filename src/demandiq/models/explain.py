"""SHAP explainability module wrapping TreeExplainer for feature importance attribution."""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from demandiq.features.engineer import FEATURE_COLUMNS, build_features
from demandiq.models.forecaster import DemandForecaster

logger = logging.getLogger(__name__)


def get_shap_values(
    model: DemandForecaster | lgb.LGBMRegressor, X: pd.DataFrame, city: str | None = None
) -> tuple[np.ndarray[Any, Any], float | np.ndarray[Any, Any]]:
    """Compute exact SHAP attribution values and expected baseline value for predictions.

    Args:
        model (DemandForecaster | lgb.LGBMRegressor): Fitted ensemble forecaster or underlying LightGBM model.
        X (pd.DataFrame): DataFrame containing input records to explain.
        city (str | None): Target city if passing a multi-city DemandForecaster. If None, derives from X.

    Returns:
        tuple[np.ndarray, float | np.ndarray]: Tuple containing array of SHAP attribution values and baseline expected value.

    Raises:
        RuntimeError: If forecaster has not been fitted or missing LightGBM component.
    """
    work_df = X.copy()
    missing = [c for c in FEATURE_COLUMNS if c not in work_df.columns]
    if missing:
        work_df = build_features(work_df)

    if isinstance(model, DemandForecaster):
        if not model.is_fitted:
            raise RuntimeError("DemandForecaster must be fitted before computing SHAP values.")
        target_city = city if city else str(work_df["city"].iloc[0])
        if target_city not in model._lgb_models:
            avail = list(model._lgb_models.keys())[0]
            lgb_mod = model._lgb_models[avail]
        else:
            lgb_mod = model._lgb_models[target_city]
    else:
        lgb_mod = model

    feature_matrix = work_df[FEATURE_COLUMNS].to_numpy()
    explainer = shap.TreeExplainer(lgb_mod)
    shap_vals = explainer.shap_values(feature_matrix)

    expected_val = explainer.expected_value
    if isinstance(expected_val, list | np.ndarray) and len(expected_val) == 1:
        expected_val = float(expected_val[0])

    return np.array(shap_vals), expected_val


def get_top_drivers(
    model: DemandForecaster | lgb.LGBMRegressor,
    X_row: pd.DataFrame | pd.Series[Any],
    n: int = 5,
    city: str | None = None,
) -> list[tuple[str, float]] | list[list[tuple[str, float]]]:
    """Extract top feature contributors sorted by absolute SHAP attribution for single row or batch.

    Args:
        model (DemandForecaster | lgb.LGBMRegressor): Trained model instance.
        X_row (pd.DataFrame | pd.Series): Single observation row or multi-row batch.
        n (int): Number of top contributing drivers to return.
        city (str | None): Optional city override for ensemble selection.

    Returns:
        list[tuple[str, float]] | list[list[tuple[str, float]]]: Sorted feature contribution pairs per observation.
    """
    if isinstance(X_row, pd.Series):
        df_batch = X_row.to_frame().T
        is_single_row = True
    else:
        df_batch = X_row.copy()  # type: ignore[assignment]
        is_single_row = len(df_batch) == 1

    shap_matrix, _ = get_shap_values(model, df_batch, city=city)

    batch_drivers: list[list[tuple[str, float]]] = []
    for row_idx in range(len(df_batch)):
        row_shaps = shap_matrix[row_idx]
        # Pair feature names with SHAP values
        feature_contributions = list(
            zip(FEATURE_COLUMNS, [float(val) for val in row_shaps], strict=False)
        )
        # Sort descending by absolute contribution magnitude
        sorted_drivers = sorted(feature_contributions, key=lambda x: abs(x[1]), reverse=True)
        batch_drivers.append(sorted_drivers[:n])

    if is_single_row:
        return batch_drivers[0]
    return batch_drivers


def get_weather_shap_contributions(
    model: DemandForecaster | lgb.LGBMRegressor, X: pd.DataFrame, city: str | None = None
) -> np.ndarray[Any, Any]:
    """Extract cumulative SHAP attribution for weather-related features across observations.

    Args:
        model (DemandForecaster | lgb.LGBMRegressor): Fitted model instance.
        X (pd.DataFrame): DataFrame containing input records.
        city (str | None): Optional target city.

    Returns:
        np.ndarray: Array of total weather SHAP attributions per row.
    """
    shap_vals, _ = get_shap_values(model, X, city)

    weather_cols = {
        "temp_sq",
        "temp_deviation",
        "log_rainfall",
        "rain_intensity",
        "promo_x_rainy",
        "festival_x_rainy",
    }
    weather_indices = [i for i, col in enumerate(FEATURE_COLUMNS) if col in weather_cols]

    if len(weather_indices) == 0:
        return np.zeros(len(X), dtype=float)

    weather_sum = np.sum(shap_vals[:, weather_indices], axis=1)
    return np.array(weather_sum, dtype=float)


def get_global_shap_summary(
    model: DemandForecaster | lgb.LGBMRegressor,
    X: pd.DataFrame,
    city: str | None = None,
    n_features: int = 15,
) -> pd.DataFrame:
    """Compute global SHAP feature importance as mean absolute SHAP across all observations.

    This provides a macro-level view of which features drive the model's predictions overall,
    complementing the per-observation local attribution from get_top_drivers.

    Args:
        model (DemandForecaster | lgb.LGBMRegressor): Fitted model instance.
        X (pd.DataFrame): DataFrame containing input records to summarize.
        city (str | None): Optional target city for ensemble selection.
        n_features (int): Number of top features to return (default 15).

    Returns:
        pd.DataFrame: DataFrame with columns ['feature', 'mean_abs_shap'] sorted descending.
    """
    shap_vals, _ = get_shap_values(model, X, city)

    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    summary_df = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "mean_abs_shap": [float(v) for v in mean_abs_shap]}
    )
    summary_df = summary_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return summary_df.head(n_features)
