"""End-to-end integration tests for DemandIQ complete operational workflow."""

import time
from pathlib import Path
import pandas as pd
import pytest
from demandiq.data.generate_synthetic import generate_synthetic_data
from demandiq.pipeline import run_pipeline


@pytest.mark.integration
def test_end_to_end_pipeline_integration(tmp_path: Path) -> None:
    """Verify full pipeline: generate synthetic data -> features -> train -> backtest -> anomalies in under 60 seconds."""
    start_time = time.time()

    raw_csv = tmp_path / "raw_orders.csv"
    feats_file = tmp_path / "processed_features.csv"
    model_file = tmp_path / "ensemble.pkl"
    det_file = tmp_path / "detector.pkl"

    # 1. Generate small synthetic dataset (~200 rows)
    df_raw, _ = generate_synthetic_data(n_days=100, seed=42, cities=["New York", "Chicago"])
    df_raw.to_csv(raw_csv, index=False)
    assert raw_csv.exists()

    # 2. Run end-to-end orchestration pipeline
    forecaster, detector, cv_metrics = run_pipeline(
        raw_path=raw_csv,
        features_out_path=feats_file,
        forecaster_path=model_file,
        detector_path=det_file,
        n_splits=3,
    )

    elapsed_time = time.time() - start_time

    # Assert models are fitted and artifacts successfully materialized on disk
    assert forecaster.is_fitted
    assert detector.is_fitted
    assert feats_file.exists(), "Engineered features file was not created by pipeline."
    assert model_file.exists(), "Trained forecaster model artifact was not saved by pipeline."
    assert det_file.exists(), "Trained anomaly detector artifact was not saved by pipeline."

    # Assert walk-forward validation completed with valid metrics
    assert "mape_model" in cv_metrics
    assert cv_metrics["mape_model"] < cv_metrics["mape_baseline"]

    # Assert whole operation finishes within 60-second execution budget
    assert elapsed_time < 60.0, f"Pipeline execution took {elapsed_time:.2f}s (exceeded 60s budget!)."
