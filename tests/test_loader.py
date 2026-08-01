"""Unit tests for schema validator and file loader."""

from datetime import datetime, timedelta
import pandas as pd
import pytest
from demandiq.data.loader import DataValidationError, load_and_validate_orders, validate_orders_df


def test_loader_valid_data_passes(sample_orders_df: pd.DataFrame) -> None:
    """Verify that a schema-compliant DataFrame successfully passes validation unchanged."""
    validated = validate_orders_df(sample_orders_df.copy())
    assert len(validated) == len(sample_orders_df)
    assert not validated.empty


def test_loader_rejects_negative_orders(sample_orders_df: pd.DataFrame) -> None:
    """Assert DataValidationError is raised when negative order values are injected."""
    bad_df = sample_orders_df.copy()
    bad_df.loc[0, "orders"] = -50
    with pytest.raises(DataValidationError, match="Orders cannot be negative"):
        validate_orders_df(bad_df)


def test_loader_rejects_missing_column(sample_orders_df: pd.DataFrame) -> None:
    """Assert DataValidationError is raised when an essential schema column is omitted."""
    bad_df = sample_orders_df.drop(columns=["orders"])
    with pytest.raises(DataValidationError):
        validate_orders_df(bad_df)


def test_loader_rejects_wrong_dtype(sample_orders_df: pd.DataFrame) -> None:
    """Assert DataValidationError is raised when invalid non-coercible string types appear in numerical columns."""
    bad_df = sample_orders_df.copy()
    # Using object dtype column injection
    bad_df["orders"] = bad_df["orders"].astype(str)
    bad_df.loc[0, "orders"] = "invalid_number_string"
    with pytest.raises(DataValidationError):
        validate_orders_df(bad_df)


def test_loader_rejects_future_dates(sample_orders_df: pd.DataFrame) -> None:
    """Assert DataValidationError is raised when records contain future calendar timestamps."""
    bad_df = sample_orders_df.copy()
    future_dt = datetime.now() + timedelta(days=365)
    bad_df.loc[0, "date"] = future_dt
    with pytest.raises(DataValidationError, match="Future dates are not allowed"):
        validate_orders_df(bad_df)


def test_load_and_validate_orders_file_not_found() -> None:
    """Assert DataValidationError is raised when pointing to non-existent CSV path."""
    with pytest.raises(DataValidationError, match="does not exist"):
        load_and_validate_orders("non_existent_fake_path.csv")


def test_load_and_validate_orders_success(sample_orders_df: pd.DataFrame, tmp_path: pd.Timestamp | Any) -> None:
    """Verify successful CSV parsing and validation from disk path."""
    csv_path = tmp_path / "orders.csv"
    sample_orders_df.to_csv(csv_path, index=False)
    
    loaded = load_and_validate_orders(csv_path)
    assert len(loaded) == len(sample_orders_df)
    assert pd.api.types.is_datetime64_any_dtype(loaded["date"])
