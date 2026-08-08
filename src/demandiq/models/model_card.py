"""Generate structured performance report cards for the DemandIQ models."""

import datetime
from typing import Any

import numpy as np
import pandas as pd

from demandiq.config import settings
from demandiq.models.cross_validate import compute_metrics


def generate_model_report(
    df: pd.DataFrame,
    forecaster: Any,
    detector: Any,
) -> dict[str, Any]:
    """Generate a comprehensive health report card for the deployed models.

    Args:
        df (pd.DataFrame): Validation dataframe containing actuals and features.
        forecaster: The trained DemandForecaster instance.
        detector: The trained HybridAnomalyDetector instance.

    Returns:
        dict[str, Any]: Structured dictionary containing model health metrics.
    """
    report: dict[str, Any] = {
        "global": {},
        "cities": {},
        "calibration": {},
        "anomaly": {},
        "metadata": {},
    }

    # 1. Evaluate Forecasting Accuracy
    preds = forecaster.predict(df)
    global_metrics = compute_metrics(df["orders"].to_numpy(), preds)
    report["global"] = global_metrics

    for city in df["city"].unique():
        city_df = df[df["city"] == city]
        city_preds = forecaster.predict(city_df)
        report["cities"][str(city)] = compute_metrics(city_df["orders"].to_numpy(), city_preds)

    # 2. Evaluate Interval Calibration (p10 to p90 coverage)
    # A perfectly calibrated model should have ~80% of actuals within p10-p90.
    try:
        intervals = forecaster.predict_intervals(df)
        p10 = intervals["p10"]
        p90 = intervals["p90"]
        actuals = df["orders"].to_numpy()

        # Avoid division by zero
        if len(actuals) > 0:
            within_bounds = np.logical_and(actuals >= p10, actuals <= p90)
            coverage = float(np.mean(within_bounds))
        else:
            coverage = 0.0

        report["calibration"]["p10_p90_coverage"] = coverage
        report["calibration"]["target_coverage"] = 0.80
    except Exception:
        report["calibration"]["p10_p90_coverage"] = None

    # 3. Anomaly Detector Rate
    residuals = df["orders"].to_numpy() - preds
    try:
        anomalies = detector.predict(residuals)
        anom_rate = float(np.mean(anomalies))
    except Exception:
        anom_rate = 0.0
    report["anomaly"]["flag_rate"] = anom_rate

    # 4. Model Metadata & Age
    try:
        # Check file modified time
        stat = settings.forecaster_model_path.stat()
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
        age = datetime.datetime.now() - mtime
        age_days = float(age.total_seconds() / 86400.0)
    except Exception:
        age_days = -1.0

    report["metadata"]["age_days"] = age_days
    report["metadata"]["is_fitted"] = getattr(forecaster, "is_fitted", False)

    return report
