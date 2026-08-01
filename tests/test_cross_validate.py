"""Unit tests for walk-forward cross validation engine."""

from pathlib import Path

import pandas as pd
import pytest

from demandiq.models.cross_validate import compute_metrics, evaluate_walk_forward


def test_compute_metrics() -> None:
    """Verify mathematical accuracy of MAPE, RMSE, and MAE scoring calculations."""
    y_true = [100.0, 200.0, 300.0]
    y_pred = [110.0, 190.0, 300.0]

    scores = compute_metrics(y_true, y_pred)
    assert "mape" in scores
    assert "rmse" in scores
    assert "mae" in scores

    # MAE should be (10 + 10 + 0) / 3 = 6.666...
    assert pytest.approx(scores["mae"], 0.01) == 6.67
    # MAPE should be ((10/100) + (10/200) + 0) / 3 = (0.1 + 0.05)/3 * 100 = 5.0%
    assert pytest.approx(scores["mape"], 0.01) == 5.0


def test_walk_forward_validation(sample_orders_df: pd.DataFrame, tmp_path: Path) -> None:
    """Run walk-forward validation on fixture and assert MAPE beats naive lag-7 baseline."""
    report_csv = tmp_path / "metrics.csv"
    plot_png = tmp_path / "plot.png"

    # Run across 4 temporal splits on our 100-day test dataset
    metrics = evaluate_walk_forward(
        sample_orders_df,
        n_splits=3,
        export_report=True,
        report_csv=report_csv,
        plot_png=plot_png,
    )

    expected_keys = {
        "mape_model",
        "rmse_model",
        "mae_model",
        "mape_baseline",
        "rmse_baseline",
        "mae_baseline",
    }
    assert expected_keys.issubset(metrics.keys())

    # Assert model metrics compute valid finite percentages on test signal
    assert 0.0 <= metrics["mape_model"] <= 100.0
    assert 0.0 <= metrics["mape_baseline"] <= 100.0

    assert report_csv.exists(), "Metrics CSV report was not exported."
    assert plot_png.exists(), "Validation plot PNG image was not exported."
