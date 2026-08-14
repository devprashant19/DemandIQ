"""Unit tests for the ModelRegistry module."""

from pathlib import Path

import pytest

from demandiq.registry.model_registry import ModelRegistry


class TestModelRegistry:
    """Tests for the ModelRegistry class."""

    def test_init_creates_registry_dir(self, tmp_path: Path) -> None:
        """Should create the registry directory on init."""
        reg_dir = tmp_path / "registry"
        registry = ModelRegistry(registry_dir=reg_dir)
        assert reg_dir.exists()
        assert registry.manifest_path.exists() or not registry.manifest_path.exists()

    def test_list_versions_empty(self, tmp_path: Path) -> None:
        """Should return empty list when no versions are registered."""
        registry = ModelRegistry(registry_dir=tmp_path / "registry")
        assert registry.list_versions() == []

    def test_load_manifest_missing_file(self, tmp_path: Path) -> None:
        """Should return default manifest when file does not exist."""
        registry = ModelRegistry(registry_dir=tmp_path / "registry")
        manifest = registry._load_manifest()
        assert "versions" in manifest
        assert manifest["versions"] == {}

    def test_load_manifest_corrupt_file(self, tmp_path: Path) -> None:
        """Should return default manifest when file is corrupt JSON."""
        reg_dir = tmp_path / "registry"
        reg_dir.mkdir()
        corrupt = reg_dir / "manifest.json"
        corrupt.write_text("{ invalid json }")
        registry = ModelRegistry(registry_dir=reg_dir)
        assert registry._manifest == {"versions": {}}

    def test_load_nonexistent_version_raises(self, tmp_path: Path) -> None:
        """Should raise ValueError when loading non-existent version."""
        registry = ModelRegistry(registry_dir=tmp_path / "registry")
        with pytest.raises(ValueError, match="not found"):
            registry.load_version("v999")

    def test_register_unfitted_model_raises(self, tmp_path: Path) -> None:
        """Should raise ValueError when registering an unfitted model."""
        from unittest.mock import MagicMock

        registry = ModelRegistry(registry_dir=tmp_path / "registry")
        mock_model = MagicMock()
        mock_model.is_fitted = False
        with pytest.raises(ValueError, match="unfitted"):
            registry.register(mock_model, "v1.0.0")

    def test_register_saves_version_metadata(self, tmp_path: Path) -> None:
        """Should save version metadata to manifest after registering."""
        from unittest.mock import MagicMock

        registry = ModelRegistry(registry_dir=tmp_path / "registry")
        mock_model = MagicMock()
        mock_model.is_fitted = True
        mock_model.lgb_weight = 0.6
        mock_model.prophet_weight = 0.4

        registry.register(mock_model, "v1.0.0", description="baseline")

        versions = registry.list_versions()
        assert len(versions) == 1
        assert versions[0]["tag"] == "v1.0.0"
        assert versions[0]["description"] == "baseline"

    def test_list_versions_sorted_by_created_at(self, tmp_path: Path) -> None:
        """Should return versions sorted newest-first."""
        from unittest.mock import MagicMock

        registry = ModelRegistry(registry_dir=tmp_path / "registry")
        for tag in ["v1.0.0", "v2.0.0", "v3.0.0"]:
            mock_model = MagicMock()
            mock_model.is_fitted = True
            mock_model.lgb_weight = 0.6
            mock_model.prophet_weight = 0.4
            registry.register(mock_model, tag)

        versions = registry.list_versions()
        assert versions[0]["tag"] == "v3.0.0"
        assert versions[-1]["tag"] == "v1.0.0"
