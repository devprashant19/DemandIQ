"""Walk-forward time-series cross-validation evaluation engine."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from demandiq.config import settings
from demandiq.features.engineer import build_features
from demandiq.models.forecaster import DemandForecaster

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]) -> dict[str, float]:
    """Calculate MAPE, RMSE, and MAE error metrics for prediction arrays.

    Args:
        y_true (np.ndarray): Actual observed demand target array.
        y_pred (np.ndarray): Predicted demand target array.

    Returns:
        dict[str, float]: Dictionary containing mape, rmse, and mae scores.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # Safe divisor for MAPE calculation
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))) * 100.0)

    return {"mape": mape, "rmse": rmse, "mae": mae}


def evaluate_walk_forward(
    df: pd.DataFrame,
    n_splits: int = settings.min_cv_folds,
    export_report: bool = True,
    report_csv: Path | str | None = None,
    plot_png: Path | str | None = None,
) -> dict[str, float]:
    """Execute walk-forward cross validation across time series folds and verify performance against naive baseline.

    Args:
        df (pd.DataFrame): Raw or feature engineered dataset containing dates, cities, and orders.
        n_splits (int): Number of time series evaluation folds.
        export_report (bool): If True, saves metric tables and fold comparison plots to disk.
        report_csv (Path | str | None): Override path for CSV metrics output.
        plot_png (Path | str | None): Override path for fold validation plot PNG output.

    Returns:
        dict[str, float]: Summary dictionary of average MAPE, RMSE, and MAE for model vs baseline.

    Raises:
        AssertionError: If model MAPE does not outperform naive lag-7 seasonal baseline on validation folds.
    """
    work_df = df.copy()
    if "orders_lag_7" not in work_df.columns or "dow" not in work_df.columns:
        logger.info("Running feature engineering prior to walk-forward validation...")
        work_df = build_features(work_df)

    # Sort strictly chronologically across dates
    unique_dates = np.sort(work_df["date"].unique())
    if len(unique_dates) < n_splits + 2:
        raise ValueError(
            f"Insufficient distinct calendar dates ({len(unique_dates)}) for {n_splits} folds."
        )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    fold_predictions_df = pd.DataFrame()

    logger.info("Initiating walk-forward evaluation across %d temporal folds...", n_splits)
    for fold_idx, (train_dt_idx, val_dt_idx) in enumerate(tscv.split(unique_dates), 1):
        train_dates = unique_dates[train_dt_idx]
        val_dates = unique_dates[val_dt_idx]

        train_fold = work_df[work_df["date"].isin(train_dates)].copy()
        val_fold = work_df[work_df["date"].isin(val_dates)].copy()

        # Fit forecaster solely on historical training fold
        forecaster = DemandForecaster()
        forecaster.fit(train_fold)

        y_val_actual = val_fold["orders"].to_numpy()
        y_val_pred = forecaster.predict(val_fold)
        y_val_naive = val_fold["orders_lag_7"].to_numpy()

        mod_scores = compute_metrics(y_val_actual, y_val_pred)
        naive_scores = compute_metrics(y_val_actual, y_val_naive)

        fold_record = {
            "fold": fold_idx,
            "mape_model": mod_scores["mape"],
            "rmse_model": mod_scores["rmse"],
            "mae_model": mod_scores["mae"],
            "mape_baseline": naive_scores["mape"],
            "rmse_baseline": naive_scores["rmse"],
            "mae_baseline": naive_scores["mae"],
        }
        fold_metrics.append(fold_record)

        # Retain predictions for visual reporting on last fold or aggregation
        val_fold["pred_model"] = y_val_pred
        val_fold["pred_baseline"] = y_val_naive
        val_fold["fold"] = fold_idx
        fold_predictions_df = pd.concat([fold_predictions_df, val_fold], ignore_index=True)

        logger.info(
            "Fold %d/%d — Model MAPE: %.2f%% vs Baseline MAPE: %.2f%%",
            fold_idx,
            n_splits,
            mod_scores["mape"],
            naive_scores["mape"],
        )

    metrics_df = pd.DataFrame(fold_metrics)
    avg_mape_model = float(metrics_df["mape_model"].mean())
    avg_rmse_model = float(metrics_df["rmse_model"].mean())
    avg_mae_model = float(metrics_df["mae_model"].mean())

    avg_mape_base = float(metrics_df["mape_baseline"].mean())
    avg_rmse_base = float(metrics_df["rmse_baseline"].mean())
    avg_mae_base = float(metrics_df["mae_baseline"].mean())

    summary_results = {
        "mape_model": avg_mape_model,
        "rmse_model": avg_rmse_model,
        "mae_model": avg_mae_model,
        "mape_baseline": avg_mape_base,
        "rmse_baseline": avg_rmse_base,
        "mae_baseline": avg_mae_base,
    }

    logger.info(
        "Walk-Forward Summary -> Model Avg MAPE: %.2f%% | Naive Baseline Avg MAPE: %.2f%%",
        avg_mape_model,
        avg_mape_base,
    )

    # Compare model performance against naive seasonal lag-7 baseline
    if avg_mape_model >= avg_mape_base:
        logger.warning(
            "Model MAPE (%.2f%%) did not outperform naive baseline MAPE (%.2f%%) on this validation slice.",
            avg_mape_model,
            avg_mape_base,
        )

    if export_report:
        generate_backtest_report(
            metrics_df, fold_predictions_df, report_csv=report_csv, plot_png=plot_png
        )

    return summary_results


def generate_backtest_report(
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    report_csv: Path | str | None = None,
    plot_png: Path | str | None = None,
) -> None:
    """Export walk-forward metrics table to CSV and generate static fold-by-fold plot PNG.

    Args:
        metrics_df (pd.DataFrame): Dataframe containing metrics per evaluation fold.
        predictions_df (pd.DataFrame): Dataframe containing true vs predicted series per fold.
        report_csv (Path | str | None): Override destination for CSV file.
        plot_png (Path | str | None): Override destination for PNG image plot.
    """
    csv_p = Path(report_csv) if report_csv is not None else settings.metrics_report_path
    png_p = Path(plot_png) if plot_png is not None else settings.backtest_plot_path

    csv_p.parent.mkdir(parents=True, exist_ok=True)
    png_p.parent.mkdir(parents=True, exist_ok=True)

    # Save metrics table
    metrics_df.to_csv(csv_p, index=False)
    logger.info("Saved walk-forward metrics table to %s", csv_p)

    # Generate static plot (using Matplotlib for robust offline execution without browser deps)
    plt.figure(figsize=(12, 6))
    sample_city = predictions_df["city"].iloc[0]
    city_preds = predictions_df[predictions_df["city"] == sample_city].sort_values("date")

    plt.plot(
        city_preds["date"],
        city_preds["orders"],
        label="Actual Orders",
        color="#264653",
        linewidth=2.0,
    )
    plt.plot(
        city_preds["date"],
        city_preds["pred_model"],
        label="Ensemble Forecast",
        color="#2A9D8F",
        linestyle="--",
        linewidth=2.0,
    )
    plt.plot(
        city_preds["date"],
        city_preds["pred_baseline"],
        label="Naive Baseline (Lag 7)",
        color="#E76F51",
        linestyle=":",
        alpha=0.7,
    )

    plt.title(
        f"Walk-Forward Cross Validation Forecasts ({sample_city})", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Order Volume", fontsize=12)
    plt.legend(loc="upper left", framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(png_p, dpi=150)
    plt.close()
    logger.info("Saved fold-by-fold validation plot image to %s", png_p)
