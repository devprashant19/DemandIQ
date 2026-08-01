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
