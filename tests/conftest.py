"""Shared Pytest fixtures for unit and integration testing across DemandIQ."""

from pathlib import Path
import pandas as pd
import pytest

from demandiq.data.generate_synthetic import generate_synthetic_data


@pytest.fixture(scope="session")
def sample_orders_df() -> pd.DataFrame:
    """Provide a small deterministic synthetic DataFrame across 2 cities (~200 rows).

    Returns:
        pd.DataFrame: Synthetic order volume dataset with 2 cities across 100 days (200 rows).
    """
    orders_df, _ = generate_synthetic_data(
        n_days=100,
        seed=123,
        cities=["New York", "Chicago"],
        end_date="2023-12-31",
    )
    return orders_df


@pytest.fixture(scope="session")
def sample_ground_truth_df() -> pd.DataFrame:
    """Provide the ground truth anomaly DataFrame corresponding to sample_orders_df.

    Returns:
        pd.DataFrame: Synthetic anomaly labels for the ~200 row test fixture.
    """
    _, ground_truth_df = generate_synthetic_data(
        n_days=100,
        seed=123,
        cities=["New York", "Chicago"],
        end_date="2023-12-31",
    )
    return ground_truth_df


@pytest.fixture
def temp_models_dir(tmp_path: Path) -> Path:
    """Create and return a temporary models directory for isolation during tests.

    Args:
        tmp_path (Path): Pytest built-in temporary directory fixture.

    Returns:
        Path: Path to isolated test models folder.
    """
    models_d = tmp_path / "models"
    models_d.mkdir(parents=True, exist_ok=True)
    return models_d
