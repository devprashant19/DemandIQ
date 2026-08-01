"""Hybrid anomaly detector combining IsolationForest and statistical Z-score/IQR rules."""

import logging
import pickle
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.ensemble import IsolationForest

from demandiq.config import settings

logger = logging.getLogger(__name__)


class HybridAnomalyDetector:
    """Hybrid anomaly detection class marrying IsolationForest unsupervised separation with residual Z-score limits.

    Design Decision:
        We employ a dual-signal combination architecture:
        - In 'union' mode (default, high sensitivity): An observation is flagged as anomalous if EITHER
          Isolation Forest diagnoses an outlier structure OR the absolute rolling residual Z-score exceeds the threshold.
          This prevents slow-drift seasonal misses from escaping statistical limits.
        - In 'strict' mode (high specificity): Both methods must concurrently flag the anomaly to trigger an alert.
    """

    def __init__(
        self,
        contamination: float = settings.anomaly_contamination,
        z_threshold: float = settings.zscore_threshold,
        strict_mode: bool = False,
        random_seed: int = settings.random_seed,
    ) -> None:
        """Initialize hybrid anomaly detector parameters and backend models.

        Args:
            contamination (float): IsolationForest expected anomaly proportion (~0.015).
            z_threshold (float): Z-score cutoff on model residuals (default 2.5).
            strict_mode (bool): If True, requires BOTH detectors to agree to flag an anomaly.
            random_seed (int): Reproducibility seed for IsolationForest tree construction.
        """
        self.contamination = contamination
        self.z_threshold = z_threshold
        self.strict_mode = strict_mode
        self.random_seed = random_seed
        self.iforest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_seed,
            n_jobs=1,
        )
        self.res_mean: float = 0.0
        self.res_std: float = 1.0
        self.is_fitted: bool = False

    def fit(self, residuals: np.ndarray | list[float]) -> "HybridAnomalyDetector":
        """Fit IsolationForest on residual distributions and establish mean/std baseline parameters.

        Args:
            residuals (np.ndarray | list[float]): Historical model errors (actual - predicted).

        Returns:
            HybridAnomalyDetector: Fitted self instance.
        """
        res_arr = np.array(residuals, dtype=float).reshape(-1, 1)
        self.res_mean = float(np.mean(res_arr))
        self.res_std = float(np.maximum(np.std(res_arr), 1e-6))

        self.iforest.fit(res_arr)
        self.is_fitted = True
        logger.info(
            "Fitted HybridAnomalyDetector (mean residual=%.2f, std=%.2f, strict_mode=%s)",
            self.res_mean,
            self.res_std,
            self.strict_mode,
        )
        return self

    def predict(self, residuals: np.ndarray | list[float]) -> np.ndarray:
        """Predict boolean anomaly status across input residual sequence.

        Args:
            residuals (np.ndarray | list[float]): New residual series (actual - predicted).

        Returns:
            np.ndarray: Boolean array where True indicates an anomalous demand spike or dip.

        Raises:
            RuntimeError: If detector has not been fitted prior to invoking predict.
        """
        if not self.is_fitted:
            raise RuntimeError("HybridAnomalyDetector must be fitted or loaded prior to predict.")

        res_arr = np.array(residuals, dtype=float).reshape(-1, 1)

        # 1. Isolation Forest signal (-1 indicates anomaly, 1 indicates normal)
        if_preds = self.iforest.predict(res_arr)
        if_flag = if_preds == -1

        # 2. Statistical Z-score calculation
        z_scores = np.abs((res_arr.flatten() - self.res_mean) / self.res_std)
        z_flag = z_scores >= self.z_threshold

        # Combine signals according to operational policy
        if self.strict_mode:
            anomalies = np.logical_and(if_flag, z_flag)
        else:
            anomalies = np.logical_or(if_flag, z_flag)

        return np.array(anomalies, dtype=bool)

    def score(self, residuals: np.ndarray | list[float]) -> np.ndarray:
        """Compute anomaly numeric severity scores combining normalized Z-scores and anomaly decision function.

        Args:
            residuals (np.ndarray | list[float]): Input residuals array.

        Returns:
            np.ndarray: One-dimensional severity scores (higher positive scores indicate stronger anomalies).

        Raises:
            RuntimeError: If detector has not been fitted prior to invoking score.
        """
        if not self.is_fitted:
            raise RuntimeError("HybridAnomalyDetector must be fitted or loaded prior to score.")

        res_arr = np.array(residuals, dtype=float).reshape(-1, 1)
        z_scores = np.abs((res_arr.flatten() - self.res_mean) / self.res_std)
        
        # Negative decision function indicates stronger anomaly
        if_scores = -1.0 * self.iforest.decision_function(res_arr)
        
        # Composite score weighting both statistical extremeness and isolation isolation depth
        composite_score = 0.6 * z_scores + 0.4 * np.maximum(0.0, if_scores * 10.0)
        return np.array(composite_score, dtype=float)

    def save(self, path: Path | str | None = None) -> None:
        """Serialize trained anomaly detector to disk.

        Args:
            path (Path | str | None): Destination path. Defaults to config setting.
        """
        save_p = Path(path) if path is not None else settings.anomaly_detector_path
        save_p.parent.mkdir(parents=True, exist_ok=True)
        with open(save_p, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved trained HybridAnomalyDetector artifact to %s", save_p)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "HybridAnomalyDetector":
        """Load trained anomaly detector from filesystem artifact.

        Args:
            path (Path | str | None): Path to pickle artifact. Defaults to config setting.

        Returns:
            HybridAnomalyDetector: Deserialized detector ready for inference.

        Raises:
            FileNotFoundError: If model file does not exist on disk.
        """
        load_p = Path(path) if path is not None else settings.anomaly_detector_path
        if not load_p.exists():
            raise FileNotFoundError(f"Anomaly detector artifact not found at: {load_p}")
        with open(load_p, "rb") as f:
            model = pickle.load(f)
        logger.info("Successfully loaded HybridAnomalyDetector from %s", load_p)
        return model
