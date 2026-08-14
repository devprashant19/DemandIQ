"""Configuration settings and path constants for DemandIQ."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determine repository project root directory
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = MODELS_DIR / "reports"


class Settings(BaseSettings):
    """Application runtime configuration settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", protected_namespaces=()
    )

    # General Settings
    random_seed: int = Field(default=42, description="Default random seed for reproducibility")

    # Paths
    raw_orders_path: Path = Field(
        default=RAW_DATA_DIR / "orders.csv",
        description="Path to the synthetic raw order data",
    )
    ground_truth_anomalies_path: Path = Field(
        default=RAW_DATA_DIR / "ground_truth_anomalies.csv",
        description="Hidden evaluation file containing true anomaly labels",
    )
    processed_features_path: Path = Field(
        default=PROCESSED_DATA_DIR / "features.parquet",
        description="Path to engineered feature dataset",
    )
    forecaster_model_path: Path = Field(
        default=MODELS_DIR / "forecaster_ensemble.pkl",
        description="Path to save/load trained ensemble model",
    )
    anomaly_detector_path: Path = Field(
        default=MODELS_DIR / "anomaly_detector.pkl",
        description="Path to save/load trained anomaly detector",
    )
    metrics_report_path: Path = Field(
        default=REPORTS_DIR / "backtest_metrics.csv",
        description="CSV path for walk-forward evaluation metrics",
    )
    backtest_plot_path: Path = Field(
        default=REPORTS_DIR / "fold_predictions.png",
        description="PNG path for fold-by-fold validation plots",
    )
    model_registry_dir: Path = Field(
        default=MODELS_DIR / "registry",
        description="Path to save versioned models",
    )

    # Modeling defaults
    lgb_weight: float = Field(default=0.6, description="Weight for LightGBM model in ensemble")
    prophet_weight: float = Field(default=0.4, description="Weight for Prophet model in ensemble")
    min_cv_folds: int = Field(
        default=5, description="Number of TimeSeriesSplit walk-forward validation folds"
    )
    quantile_alphas: list[float] = Field(
        default=[0.1, 0.5, 0.9], description="Quantile alphas for uncertainty prediction intervals"
    )
    anomaly_contamination: float = Field(
        default=0.015, description="Expected proportion of anomalies in IsolationForest"
    )
    zscore_threshold: float = Field(
        default=2.5, description="Z-score threshold on residuals for anomaly detection"
    )

    # Notifications
    alert_enabled: bool = Field(
        default=False, description="Enable push notifications for anomalies"
    )
    smtp_host: str | None = Field(default=None, description="SMTP host for email alerts")
    smtp_port: int = Field(default=587, description="SMTP port")
    smtp_user: str | None = Field(default=None, description="SMTP username")
    smtp_password: str | None = Field(default=None, description="SMTP password")
    smtp_to: str | None = Field(default=None, description="Recipient email address")
    slack_webhook_url: str | None = Field(default=None, description="Slack webhook URL for alerts")


def get_settings() -> Settings:
    """Instantiate and return runtime application settings.

    Returns:
        Settings: Configured application settings instance.
    """
    # Ensure standard directories exist when settings are loaded
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return Settings()


settings = get_settings()
