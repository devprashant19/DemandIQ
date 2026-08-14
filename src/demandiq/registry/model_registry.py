"""Model registry for version control and A/B comparison of DemandIQ forecasters."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from demandiq.config import settings
from demandiq.models.cross_validate import compute_metrics
from demandiq.models.forecaster import DemandForecaster

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manages saved DemandForecaster versions, allowing for rollback and A/B comparisons."""

    def __init__(self, registry_dir: Path | str | None = None) -> None:
        """Initialize the model registry.

        Args:
            registry_dir: Path to the registry root directory.
        """
        self.registry_dir = (
            Path(registry_dir) if registry_dir is not None else settings.model_registry_dir
        )
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.registry_dir / "manifest.json"
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"versions": {}}
        try:
            with open(self.manifest_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load registry manifest: %s", e)
            return {"versions": {}}

    def _save_manifest(self) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2)

    def register(self, model: DemandForecaster, version_tag: str, description: str = "") -> None:
        """Save a forecaster model to the registry with a version tag.

        Args:
            model: The trained DemandForecaster to save.
            version_tag: Unique string tag for the version (e.g., 'v1.0.0-lgb0.6').
            description: Optional description of the model changes.
        """
        if not getattr(model, "is_fitted", False):
            raise ValueError("Cannot register an unfitted model.")

        version_dir = self.registry_dir / version_tag
        version_dir.mkdir(parents=True, exist_ok=True)

        model_path = version_dir / "forecaster.pkl"
        model.save(path=model_path)

        self._manifest["versions"][version_tag] = {
            "created_at": datetime.now().isoformat(),
            "description": description,
            "lgb_weight": getattr(model, "lgb_weight", None),
            "prophet_weight": getattr(model, "prophet_weight", None),
        }
        self._save_manifest()
        logger.info("Registered model version '%s' successfully.", version_tag)

    def load_version(self, version_tag: str) -> DemandForecaster:
        """Load a specific version from the registry.

        Args:
            version_tag: The tag to load.

        Returns:
            DemandForecaster: The loaded model.
        """
        if version_tag not in self._manifest["versions"]:
            raise ValueError(f"Version '{version_tag}' not found in registry.")

        model_path = self.registry_dir / version_tag / "forecaster.pkl"
        return DemandForecaster.load(model_path)

    def list_versions(self) -> list[dict[str, Any]]:
        """List all registered model versions.

        Returns:
            list[dict]: List of version metadata dictionaries, sorted by created_at.
        """
        versions = []
        for tag, meta in self._manifest["versions"].items():
            entry = {"tag": tag}
            entry.update(meta)
            versions.append(entry)
        return sorted(versions, key=lambda x: x["created_at"], reverse=True)

    def compare(self, tag_a: str, tag_b: str, df: pd.DataFrame) -> dict[str, Any]:
        """Perform head-to-head A/B comparison of two model versions on a dataset.

        Args:
            tag_a: Champion version tag.
            tag_b: Challenger version tag.
            df: Validation dataset with 'orders' and features.

        Returns:
            dict: Comparison metrics for both models.
        """
        model_a = self.load_version(tag_a)
        model_b = self.load_version(tag_b)

        preds_a = model_a.predict(df)
        preds_b = model_b.predict(df)

        actuals = df["orders"].to_numpy()

        metrics_a = compute_metrics(actuals, preds_a)
        metrics_b = compute_metrics(actuals, preds_b)

        return {
            tag_a: metrics_a,
            tag_b: metrics_b,
            "improvement": {
                "mape_diff": metrics_a["mape"] - metrics_b["mape"],
                "rmse_diff": metrics_a["rmse"] - metrics_b["rmse"],
            },
        }
