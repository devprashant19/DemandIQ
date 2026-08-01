"""Unit tests for feature engineering pipeline and transformations."""

import pandas as pd

from demandiq.features.engineer import FEATURE_COLUMNS, build_features, compute_city_target_encoding


def test_build_features_schema(sample_orders_df: pd.DataFrame) -> None:
    """Verify that build_features creates more than 40 features with correct column names."""
    feat_df = build_features(sample_orders_df)
    for col in FEATURE_COLUMNS:
        assert col in feat_df.columns, f"Expected feature column {col} missing."
    assert len(FEATURE_COLUMNS) >= 40


def test_no_nans_remaining(sample_orders_df: pd.DataFrame) -> None:
    """Assert zero NaNs remain in the dataset after the designed fill strategy."""
    feat_df = build_features(sample_orders_df)
    assert feat_df.isna().sum().sum() == 0, "Engineered dataframe contains NaN values."


def test_lag_features_shifted_correctly() -> None:
    """Assert that lag features correspond exactly to historically shifted row values."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    test_df = pd.DataFrame(
        {
            "date": dates,
            "city": "TestCity",
            "orders": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "temperature_c": 20.0,
            "rainfall_mm": 0.0,
            "is_rainy": 0,
            "is_holiday": 0,
            "festival_flag": 0,
            "promo_active": 0,
        }
    )

    feat_df = build_features(test_df)

    # At index 1 (date 2023-01-02, orders 20), lag_1 should be 10 (the previous day's orders)
    assert feat_df.loc[1, "orders_lag_1"] == 10.0
    # At index 3 (orders 40), lag_1 should be 30, lag_2 should be 20, lag_3 should be 10
    assert feat_df.loc[3, "orders_lag_1"] == 30.0
    assert feat_df.loc[3, "orders_lag_2"] == 20.0
    assert feat_df.loc[3, "orders_lag_3"] == 10.0


def test_target_encoding_no_leakage() -> None:
    """Assert target encoding on fold N does not use fold N's own target rows."""
    fold_train = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=2),
            "city": ["New York", "Chicago"],
            "orders": [100, 200],
            "temperature_c": [15, 15],
            "rainfall_mm": [0, 0],
        }
    )

    fold_test = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-03", periods=2),
            "city": ["New York", "Chicago"],
            "orders": [9999, 8888],  # Completely different order volumes in test fold
            "temperature_c": [15, 15],
            "rainfall_mm": [0, 0],
        }
    )

    train_enc = compute_city_target_encoding(fold_train)
    assert train_enc["New York"] == 100.0
    assert train_enc["Chicago"] == 200.0

    test_feats = build_features(fold_test, target_encoding=train_enc)

    ny_enc = test_feats.loc[test_feats["city"] == "New York", "city_target_enc"].iloc[0]
    ch_enc = test_feats.loc[test_feats["city"] == "Chicago", "city_target_enc"].iloc[0]

    assert ny_enc == 100.0, f"Target encoding leaked test fold values! Got {ny_enc} instead of 100."
    assert ch_enc == 200.0, f"Target encoding leaked test fold values! Got {ch_enc} instead of 200."


def test_pure_function_no_side_effects(sample_orders_df: pd.DataFrame) -> None:
    """Verify that calling build_features does not mutate the input dataframe."""
    orig_df = sample_orders_df.copy()
    _ = build_features(sample_orders_df)
    pd.testing.assert_frame_equal(orig_df, sample_orders_df)
