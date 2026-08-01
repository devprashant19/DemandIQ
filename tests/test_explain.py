"""Unit tests for SHAP feature explainability wrappers."""

import numpy as np
import pandas as pd
import pytest
from demandiq.features.engineer import FEATURE_COLUMNS, build_features
from demandiq.models.explain import get_shap_values, get_top_drivers
from demandiq.models.forecaster import DemandForecaster


@pytest.fixture(scope="module")
def fitted_model_and_data(sample_orders_df: pd.DataFrame) -> tuple[DemandForecaster, pd.DataFrame]:
    """Train and return forecaster alongside engineered dataset for explainability assertions."""
    fe_df = build_features(sample_orders_df)
    model = DemandForecaster(random_seed=42).fit(fe_df)
    return model, fe_df


def test_shap_expected_value_sum_consistency(fitted_model_and_data: tuple[DemandForecaster, pd.DataFrame]) -> None:
    """Assert standard SHAP mathematical sanity check: expected_value + sum(shap_values) approximately equals model prediction."""
    model, fe_df = fitted_model_and_data
    # Pick a sample batch from one city to check underlying LightGBM explainer math
    sample_sub = fe_df[fe_df["city"] == "New York"].iloc[:5].copy()
    
    lgb_mod = model._lgb_models["New York"]
    shap_matrix, expected_val = get_shap_values(lgb_mod, sample_sub)
    
    lgb_preds = lgb_mod.predict(sample_sub[FEATURE_COLUMNS].to_numpy())
    shap_sums = shap_matrix.sum(axis=1) + expected_val
    
    np.testing.assert_allclose(shap_sums, lgb_preds, rtol=1e-3, atol=1e-3)


def test_get_top_drivers_single_row(fitted_model_and_data: tuple[DemandForecaster, pd.DataFrame]) -> None:
    """Verify get_top_drivers on single row Series returns exactly N sorted items by magnitude."""
    model, fe_df = fitted_model_and_data
    single_row = fe_df.iloc[0]
    
    n_drivers = 5
    drivers = get_top_drivers(model, single_row, n=n_drivers)
    
    assert isinstance(drivers, list)
    assert len(drivers) == n_drivers
    
    # Assert each item is a (feature_name, shap_contribution) tuple
    for item in drivers:
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], str)
        assert isinstance(item[1], float)
        
    # Assert strictly sorted by absolute contribution magnitude descending
    abs_vals = [abs(val) for _, val in drivers]
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_get_top_drivers_batch(fitted_model_and_data: tuple[DemandForecaster, pd.DataFrame]) -> None:
    """Verify get_top_drivers on multi-row batch returns a list of sorted feature lists."""
    model, fe_df = fitted_model_and_data
    batch_df = fe_df.iloc[:4].copy()
    
    drivers_batch = get_top_drivers(model, batch_df, n=3)
    assert isinstance(drivers_batch, list)
    assert len(drivers_batch) == 4
    for row_drivers in drivers_batch:
        assert len(row_drivers) == 3
