"""Unit tests for hybrid anomaly detector class."""

from pathlib import Path

import numpy as np

from demandiq.anomaly.detector import HybridAnomalyDetector


def test_detector_fit_and_flag_outliers(temp_models_dir: Path) -> None:
    """Inject extreme residual outliers into normal distribution and assert detector triggers alerts."""
    rng = np.random.default_rng(12345)
    normal_residuals = rng.normal(loc=0.0, scale=20.0, size=500)

    detector = HybridAnomalyDetector(contamination=0.02, z_threshold=2.5, strict_mode=False)
    detector.fit(normal_residuals)

    assert detector.is_fitted

    # Check that normal residuals yield low false-positive rate
    normal_test = rng.normal(loc=0.0, scale=15.0, size=50)
    flags_normal = detector.predict(normal_test)
    assert (
        flags_normal.mean() < 0.10
    ), f"Excessive false positive rate ({flags_normal.mean()*100:.1f}%) on normal residuals."

    # Inject known massive residual anomalies
    extreme_outliers = np.array([500.0, -450.0, 700.0, -600.0])
    flags_outliers = detector.predict(extreme_outliers)
    assert np.all(flags_outliers), "Detector failed to flag synthetic massive residual outliers!"

    # Check severity scores are higher for outliers than normal errors
    scores_normal = detector.score(normal_test)
    scores_outliers = detector.score(extreme_outliers)
    assert scores_outliers.mean() > scores_normal.mean()


def test_detector_strict_mode() -> None:
    """Verify that strict mode requires simultaneous agreement between IsolationForest and Z-score."""
    detector = HybridAnomalyDetector(strict_mode=True, z_threshold=10.0)  # Unreachable Z-threshold
    normal_res = np.random.normal(0, 10, 100)
    detector.fit(normal_res)

    # Even an anomaly that might trip iforest shouldn't trigger in strict mode if Z-score under threshold
    res = detector.predict([50.0])
    assert not res[0]


def test_detector_save_load(temp_models_dir: Path) -> None:
    """Verify save and load round-trip preserves anomaly boundary scoring."""
    det_path = temp_models_dir / "anom_test.pkl"
    det = HybridAnomalyDetector()
    det.fit(np.random.normal(0, 5, 200))
    det.save(det_path)

    loaded = HybridAnomalyDetector.load(det_path)
    assert loaded.is_fitted
    assert loaded.res_mean == det.res_mean
    assert loaded.res_std == det.res_std


def test_detector_classify() -> None:
    """Verify classify distinguishes properly between normal, surge, and dip residuals."""
    rng = np.random.default_rng(999)
    normal_res = rng.normal(0, 10, 200)

    detector = HybridAnomalyDetector(z_threshold=2.5, strict_mode=False)
    detector.fit(normal_res)

    test_res = np.array([5.0, 500.0, -450.0, -2.0])
    classes = detector.classify(test_res)

    assert classes[0] == "normal"
    assert classes[1] == "surge"
    assert classes[2] == "dip"
    assert classes[3] == "normal"
