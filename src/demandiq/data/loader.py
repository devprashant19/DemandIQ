"""Data loader and schema validator for raw order datasets."""

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import pandera as pa

from demandiq.config import settings

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Exception raised when dataset schema or data domain validation fails."""

    pass


def _get_orders_schema() -> pa.DataFrameSchema:
    """Construct and return the Pandera validation schema for order datasets.

    Returns:
        pa.DataFrameSchema: Configured validation schema.
    """
    today_dt = pd.to_datetime(date.today())
    return pa.DataFrameSchema(
        columns={
            "date": pa.Column(
                pa.DateTime,
                checks=[
                    pa.Check(lambda d: d <= today_dt, error="Future dates are not allowed."),
                ],
                nullable=False,
            ),
            "city": pa.Column(pa.String, nullable=False),
            "orders": pa.Column(
                pa.Int,
                checks=[pa.Check(lambda x: x >= 0, error="Orders cannot be negative.")],
                nullable=False,
            ),
            "temperature_c": pa.Column(pa.Float, nullable=False, coerce=True),
            "rainfall_mm": pa.Column(
                pa.Float, checks=[pa.Check.ge(0.0)], nullable=False, coerce=True
            ),
            "is_rainy": pa.Column(
                pa.Int, checks=[pa.Check.isin([0, 1])], nullable=False, coerce=True
            ),
            "is_holiday": pa.Column(
                pa.Int, checks=[pa.Check.isin([0, 1])], nullable=False, coerce=True
            ),
            "festival_flag": pa.Column(
                pa.Int, checks=[pa.Check.isin([0, 1])], nullable=False, coerce=True
            ),
            "promo_active": pa.Column(
                pa.Int, checks=[pa.Check.isin([0, 1])], nullable=False, coerce=True
            ),
        },
        strict=False,
        coerce=True,
    )


def validate_orders_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate DataFrame against standard order schema requirements.

    Args:
        df (pd.DataFrame): Input DataFrame to validate.

    Returns:
        pd.DataFrame: Validated and correctly coerced DataFrame.

    Raises:
        DataValidationError: If schema check, type coercion, or range check fails.
    """
    schema = _get_orders_schema()
    try:
        validated = schema.validate(df, lazy=True)
        return validated
    except Exception as exc:
        msg = f"Data validation failed for orders dataset: {exc}"
        logger.error(msg)
        raise DataValidationError(msg) from exc


def load_and_validate_orders(path: Path | str | None = None) -> pd.DataFrame:
    """Load raw orders dataset from disk and perform schema validation.

    Args:
        path (Path | str | None): File system path to orders CSV. Defaults to settings path.

    Returns:
        pd.DataFrame: Validated orders dataset.

    Raises:
        DataValidationError: If loading fails or data fails schema validation.
    """
    load_p = Path(path) if path is not None else settings.raw_orders_path
    if not load_p.exists():
        raise DataValidationError(f"Orders dataset file does not exist at: {load_p}")

    try:
        df = pd.read_csv(load_p)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="raise")
    except Exception as exc:
        raise DataValidationError(f"Failed to read CSV file from {load_p}: {exc}") from exc

    logger.info("Loaded dataset from %s (%d rows). Performing validation...", load_p, len(df))
    validated_df = validate_orders_df(df)
    logger.info("Validation completed successfully for %s.", load_p)
    return validated_df
