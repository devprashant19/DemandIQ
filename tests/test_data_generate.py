"""Unit tests for synthetic dataset generator."""

import pandas as pd
import pytest
from demandiq.data.generate_synthetic import generate_synthetic_data, save_synthetic_data


def test_generator_determinism() -> None:
    """Verify that identical random seed parameters yield identical dataset outputs."""
    df1, gt1 = generate_synthetic_data(n_days=50, seed=999, cities=["New York"])
    df2, gt2 = generate_synthetic_data(n_days=50, seed=999, cities=["New York"])
    
    pd.testing.assert_frame_equal(df1, df2)
    pd.testing.assert_frame_equal(gt1, gt2)


def test_generator_schema_and_dtypes(sample_orders_df: pd.DataFrame) -> None:
    """Verify generated dataframe possesses required output columns and valid data types."""
    expected_columns = {
        "date",
        "city",
        "orders",
        "temperature_c",
        "rainfall_mm",
        "is_rainy",
        "is_holiday",
        "festival_flag",
        "promo_active",
    }
    assert expected_columns.issubset(sample_orders_df.columns)
    assert pd.api.types.is_datetime64_any_dtype(sample_orders_df["date"])
    assert pd.api.types.is_integer_dtype(sample_orders_df["orders"])


def test_generator_domain_validity(sample_orders_df: pd.DataFrame) -> None:
    """Ensure orders and rainfall remain strictly non-negative across generated samples."""
    assert (sample_orders_df["orders"] >= 0).all()
    assert (sample_orders_df["rainfall_mm"] >= 0.0).all()
    assert sample_orders_df["is_rainy"].isin([0, 1]).all()
    assert sample_orders_df["is_holiday"].isin([0, 1]).all()


def test_generator_date_range() -> None:
    """Assert generated output strictly obeys requested date boundaries and length."""
    n_days = 60
    df, _ = generate_synthetic_data(n_days=n_days, seed=42, cities=["Austin"], end_date="2023-12-31")
    assert len(df) == n_days
    assert df["date"].max() == pd.Timestamp("2023-12-31")
    assert df["date"].min() == pd.Timestamp("2023-12-31") - pd.Timedelta(days=n_days - 1)


def test_save_synthetic_data(tmp_path: pd.Timestamp | Any) -> None:
    """Verify exporting files to CSV preserves integrity and separates ground truth labels."""
    orders_csv = tmp_path / "test_orders.csv"
    anom_csv = tmp_path / "test_anomalies.csv"
    
    save_synthetic_data(out_path=orders_csv, anomalies_out_path=anom_csv, n_years=1, seed=42)
    
    assert orders_csv.exists()
    assert anom_csv.exists()
    
    loaded_orders = pd.read_csv(orders_csv)
    loaded_anom = pd.read_csv(anom_csv)
    
    assert "is_anomaly" not in loaded_orders.columns
    assert "is_anomaly" in loaded_anom.columns
    assert len(loaded_orders) == len(loaded_anom)
