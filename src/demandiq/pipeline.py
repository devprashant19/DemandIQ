"""End-to-end operational training and evaluation workflow orchestration script."""

import logging
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from demandiq.anomaly.detector import HybridAnomalyDetector
from demandiq.config import settings
from demandiq.data.loader import load_and_validate_orders
from demandiq.features.engineer import build_features
from demandiq.models.cross_validate import evaluate_walk_forward
from demandiq.models.forecaster import DemandForecaster

logger = logging.getLogger(__name__)


def run_pipeline(
    raw_path: Path | str | None = None,
    features_out_path: Path | str | None = None,
    forecaster_path: Path | str | None = None,
    detector_path: Path | str | None = None,
    n_splits: int = settings.min_cv_folds,
) -> tuple[DemandForecaster, HybridAnomalyDetector, dict[str, float]]:
    """Execute end-to-end pipeline: data load -> features -> backtest -> ensemble train -> anomaly fit -> save artifacts.

    Args:
        raw_path (Path | str | None): Path to input synthetic CSV dataset.
        features_out_path (Path | str | None): Target path to export engineered features Parquet/CSV.
        forecaster_path (Path | str | None): Target path to save trained forecaster model artifact.
        detector_path (Path | str | None): Target path to save trained anomaly detector artifact.
        n_splits (int): Number of walk-forward validation splits to run.

    Returns:
        tuple[DemandForecaster, HybridAnomalyDetector, dict[str, float]]: Trained models and validation summary metrics.
    """
    logger.info("=== Starting DemandIQ Pipeline Execution ===")

    # 1. Load and validate data against schema
    df_raw = load_and_validate_orders(path=raw_path)

    # 2. Engineer leak-free features
    logger.info("Generating >40 temporal, lag, rolling, and interaction features...")
    df_feats = build_features(df_raw)

    out_feats_p = Path(features_out_path) if features_out_path is not None else settings.processed_features_path
    out_feats_p.parent.mkdir(parents=True, exist_ok=True)
    if out_feats_p.suffix == ".parquet":
        try:
            df_feats.to_parquet(out_feats_p, index=False)
        except Exception:
            # Fallback to CSV if parquet engine unsupported in current container state
            fallback = out_feats_p.with_suffix(".csv")
            df_feats.to_csv(fallback, index=False)
            logger.info("Exported engineered feature dataset to CSV fallback: %s", fallback)
    else:
        df_feats.to_csv(out_feats_p, index=False)
        logger.info("Exported engineered feature dataset to %s", out_feats_p)

    # 3. Walk-Forward Cross Validation Backtesting
    logger.info("Executing walk-forward backtesting against seasonal baseline...")
    cv_metrics = evaluate_walk_forward(df_feats, n_splits=n_splits, export_report=True)

    # 4. Train complete ensemble model across entire dataset
    logger.info("Fitting complete DemandForecaster ensemble across all available historical records...")
    forecaster = DemandForecaster()
    forecaster.fit(df_feats)
    forecaster.save(path=forecaster_path)

    # 5. Fit Hybrid Anomaly Detector on observed residual error distribution
    logger.info("Calculating model residuals and fitting HybridAnomalyDetector...")
    full_preds = forecaster.predict(df_feats)
    residuals = df_feats["orders"].to_numpy() - full_preds

    detector = HybridAnomalyDetector()
    detector.fit(residuals)
    detector.save(path=detector_path)

    logger.info("=== Successfully Completed DemandIQ Pipeline Execution ===")
    return forecaster, detector, cv_metrics


def main() -> None:
    """CLI handler for executing full DemandIQ production model training pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_pipeline()


if __name__ == "__main__":
    main()
