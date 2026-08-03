"""Evaluation module for checking anomaly detection performance against ground-truth labels."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from demandiq.anomaly.detector import HybridAnomalyDetector
from demandiq.config import settings

logger = logging.getLogger(__name__)


def evaluate_anomaly_detection(
    detector: HybridAnomalyDetector,
    df: pd.DataFrame,
    residuals: np.ndarray[Any, Any] | list[float] | None = None,
    ground_truth_path: Path | str | None = None,
    ground_truth_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Evaluate anomaly detector predictions against true synthetic anomaly labels.

    Args:
        detector (HybridAnomalyDetector): Trained anomaly detector instance.
        df (pd.DataFrame): DataFrame containing date, city, and optional is_anomaly flag or residuals.
        residuals (np.ndarray | list[float] | None): Optional residuals array to predict anomaly flags if not in df.
        ground_truth_path (Path | str | None): Path to ground truth CSV file. Defaults to config setting.
        ground_truth_df (pd.DataFrame | None): Optional preloaded ground truth DataFrame (overrides path).

    Returns:
        dict[str, Any]: Evaluation summary containing overall and per-city precision, recall, and F1 scores.
    """
    if ground_truth_df is not None:
        gt_df = ground_truth_df.copy()
    else:
        gt_path = Path(ground_truth_path) if ground_truth_path else settings.ground_truth_anomalies_path
        if not gt_path.exists():
            logger.warning("Ground-truth anomaly file not found at %s. Skipping evaluation.", gt_path)
            return {}
        gt_df = pd.read_csv(gt_path)

    work_df = df.copy()
    work_df["date"] = pd.to_datetime(work_df["date"])
    gt_df["date"] = pd.to_datetime(gt_df["date"])

    # Ensure predictions are present
    if "is_anomaly" not in work_df.columns:
        if residuals is not None:
            work_df["is_anomaly"] = detector.predict(residuals)
        elif "orders" in work_df.columns and "pred_orders" in work_df.columns:
            res_val = work_df["orders"].to_numpy() - work_df["pred_orders"].to_numpy()
            work_df["is_anomaly"] = detector.predict(res_val)
        elif "residuals" in work_df.columns:
            work_df["is_anomaly"] = detector.predict(work_df["residuals"].to_numpy())
        else:
            raise ValueError(
                "DataFrame missing 'is_anomaly' and no residuals provided to calculate predictions."
            )

    # Merge predicted anomalies with true anomalies on date and city
    merged = pd.merge(
        work_df[["date", "city", "is_anomaly"]],
        gt_df[["date", "city", "is_anomaly"]],
        on=["date", "city"],
        suffixes=("_pred", "_true"),
    )

    if merged.empty:
        logger.warning(
            "No matching records found between evaluation dataframe and ground-truth labels."
        )
        return {}

    y_true = merged["is_anomaly_true"].astype(int).to_numpy()
    y_pred = merged["is_anomaly_pred"].astype(int).to_numpy()

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    logger.info(
        "Anomaly Detection Evaluation Summary -> Precision: %.2f | Recall: %.2f | F1: %.2f",
        precision,
        recall,
        f1,
    )

    city_metrics: dict[str, dict[str, float]] = {}
    for city in sorted(merged["city"].unique().tolist()):
        city_mask = merged["city"] == city
        cy_true = merged.loc[city_mask, "is_anomaly_true"].astype(int).to_numpy()
        cy_pred = merged.loc[city_mask, "is_anomaly_pred"].astype(int).to_numpy()

        c_prec = float(precision_score(cy_true, cy_pred, zero_division=0))
        c_rec = float(recall_score(cy_true, cy_pred, zero_division=0))
        c_f1 = float(f1_score(cy_true, cy_pred, zero_division=0))

        city_metrics[str(city)] = {"precision": c_prec, "recall": c_rec, "f1": c_f1}
        logger.info(
            "  City: %-12s -> Precision: %.2f | Recall: %.2f | F1: %.2f", city, c_prec, c_rec, c_f1
        )

    return {
        "overall_precision": precision,
        "overall_recall": recall,
        "overall_f1": f1,
        "city_metrics": city_metrics,
        "n_samples": len(merged),
        "n_true_anomalies": int(y_true.sum()),
        "n_predicted_anomalies": int(y_pred.sum()),
    }
