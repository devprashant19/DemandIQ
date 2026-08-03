"""Unit and integration tests for anomaly detection ground truth evaluation."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from demandiq.anomaly.detector import HybridAnomalyDetector
from demandiq.models.evaluate_anomaly import evaluate_anomaly_detection


def test_evaluate_anomaly_detection_with_dataframe(
    sample_orders_df: pd.DataFrame, sample_ground_truth_df: pd.DataFrame
) -> None:
    """Verify evaluating against ground truth returns precision, recall, and F1 metrics."""
    # Create fake residuals where true anomalies have huge residuals
    work_df = sample_orders_df.copy()
    merged = pd.merge(work_df, sample_ground_truth_df, on=["date", "city"])

    # Generate residuals that align with ground truth anomalies
    residuals = np.where(merged["is_anomaly"] == 1, 600.0, 10.0)

    detector = HybridAnomalyDetector(contamination=0.02, z_threshold=2.5, strict_mode=False)
    detector.fit(residuals)

    results: dict[str, Any] = evaluate_anomaly_detection(
        detector=detector,
        df=work_df,
        residuals=residuals,
        ground_truth_df=sample_ground_truth_df,
    )

    assert "overall_precision" in results
    assert "overall_recall" in results
    assert "overall_f1" in results
    assert "city_metrics" in results
    assert results["overall_f1"] > 0.0
    assert "New York" in results["city_metrics"]
    assert "Chicago" in results["city_metrics"]


def test_evaluate_anomaly_detection_missing_file(
    sample_orders_df: pd.DataFrame, tmp_path: Path
) -> None:
    """Verify missing ground truth file gracefully returns empty dictionary."""
    detector = HybridAnomalyDetector()
    detector.fit(np.zeros(len(sample_orders_df)))

    results = evaluate_anomaly_detection(
        detector=detector,
        df=sample_orders_df,
        residuals=np.zeros(len(sample_orders_df)),
        ground_truth_path=tmp_path / "nonexistent.csv",
    )
    assert results == {}


def test_evaluate_anomaly_detection_raises_when_no_predictions(
    sample_orders_df: pd.DataFrame, sample_ground_truth_df: pd.DataFrame
) -> None:
    """Verify ValueError is raised when DataFrame lacks is_anomaly and no residuals are passed."""
    detector = HybridAnomalyDetector()
    detector.fit(np.zeros(len(sample_orders_df)))

    with pytest.raises(
        ValueError, match="DataFrame missing 'is_anomaly' and no residuals provided"
    ):
        evaluate_anomaly_detection(
            detector=detector,
            df=sample_orders_df,
            ground_truth_df=sample_ground_truth_df,
        )
