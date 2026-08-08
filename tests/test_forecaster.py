"""Unit and integration tests for DemandForecaster class."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demandiq.features.engineer import build_features
from demandiq.models.forecaster import DemandForecaster


@pytest.fixture(scope="module")
def fe_orders_df(sample_orders_df: pd.DataFrame) -> pd.DataFrame:
    """Return feature-engineered synthetic dataframe for modeling tests."""
    return build_features(sample_orders_df)


def test_forecaster_fit_predict(fe_orders_df: pd.DataFrame) -> None:
    """Verify forecaster fits across multiple cities without error and outputs valid predictions."""
    model = DemandForecaster(lgb_weight=0.6, prophet_weight=0.4, random_seed=42)
    model.fit(fe_orders_df)

    assert model.is_fitted
    assert "New York" in model._lgb_models
    assert "Chicago" in model._lgb_models
    assert "New York" in model._prophet_models
    assert "Chicago" in model._prophet_models

    preds = model.predict(fe_orders_df)
    assert len(preds) == len(fe_orders_df)
    assert np.all(np.isfinite(preds)), "Predictions contain non-finite numbers (NaN/inf)."
    assert np.all(preds >= 0.0), "Predictions contain invalid negative values."


def test_forecaster_save_load_round_trip(fe_orders_df: pd.DataFrame, temp_models_dir: Path) -> None:
    """Assert model serialization save/load round-trip yields identical prediction arrays."""
    model_path = temp_models_dir / "test_ensemble.pkl"

    orig_model = DemandForecaster(random_seed=123)
    orig_model.fit(fe_orders_df)
    orig_preds = orig_model.predict(fe_orders_df)

    orig_model.save(model_path)
    assert model_path.exists()

    loaded_model = DemandForecaster.load(model_path)
    assert loaded_model.is_fitted
    loaded_preds = loaded_model.predict(fe_orders_df)

    np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-5)


def test_forecaster_unfitted_predict_raises(fe_orders_df: pd.DataFrame) -> None:
    """Assert invoking predict on unfitted model raises clear RuntimeError."""
    unfitted = DemandForecaster()
    with pytest.raises(RuntimeError, match="must be fitted or loaded before calling predict"):
        unfitted.predict(fe_orders_df)


def test_forecaster_predict_intervals(fe_orders_df: pd.DataFrame) -> None:
    """Verify predict_intervals outputs mean, p10, p50, p90 arrays with expected inequality bounds."""
    model = DemandForecaster(random_seed=42)
    model.fit(fe_orders_df)

    intervals = model.predict_intervals(fe_orders_df)
    assert "mean" in intervals
    assert "p10" in intervals
    assert "p50" in intervals
    assert "p90" in intervals

    assert len(intervals["p10"]) == len(fe_orders_df)
    assert np.all(
        intervals["p10"] <= intervals["mean"] + 1e-6
    ), "p10 should be less than or equal to mean"
    assert np.all(
        intervals["p90"] >= intervals["mean"] - 1e-6
    ), "p90 should be greater than or equal to mean"


def test_forecaster_forecast_future(fe_orders_df: pd.DataFrame) -> None:
    """Verify forecast_future generates N future days for each city."""
    model = DemandForecaster(random_seed=42)
    model.fit(fe_orders_df)

    # Take last 40 days for a couple of cities to speed up test
    cities = ["New York", "Chicago"]
    sub_df = fe_orders_df[fe_orders_df["city"].isin(cities)].copy()

    # We just need the last 40 days per city
    sub_df = sub_df.groupby("city").tail(40).reset_index(drop=True)

    horizon = 5
    future_df = model.forecast_future(horizon_days=horizon, last_known_df=sub_df)

    assert len(future_df) == len(cities) * horizon
    assert "is_future" in future_df.columns
    assert future_df["is_future"].all()
    assert future_df["orders"].notnull().all()
    assert future_df["pred_orders"].notnull().all()

    # Check that dates are correctly advanced
    ny_hist = sub_df[sub_df["city"] == "New York"]
    last_hist_date = ny_hist["date"].max()

    ny_future = future_df[future_df["city"] == "New York"]
    assert len(ny_future) == horizon
    assert ny_future["date"].min() == last_hist_date + pd.Timedelta(days=1)
    assert ny_future["date"].max() == last_hist_date + pd.Timedelta(days=horizon)
