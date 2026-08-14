"""Unit tests for the FastAPI application."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from demandiq.api.app import app, models


@pytest.fixture
def client() -> TestClient:
    """Test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_models() -> None:
    """Clear models dict before and after each test."""
    models.clear()
    yield
    models.clear()


def test_health_check_no_models(client: TestClient) -> None:
    """Should return degraded when models are not loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["is_fitted"] is False


def test_health_check_with_models(client: TestClient) -> None:
    """Should return ok when models are loaded."""
    mock_forecaster = MagicMock()
    mock_forecaster.is_fitted = True
    models["forecaster"] = mock_forecaster

    mock_df = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")]})
    models["latest_df"] = mock_df

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["is_fitted"] is True


def test_metrics_no_models_raises_503(client: TestClient) -> None:
    """Should raise 503 if models not loaded."""
    response = client.get("/metrics")
    assert response.status_code == 503


@patch("demandiq.api.app.compute_metrics")
def test_metrics_success(mock_compute: MagicMock, client: TestClient) -> None:
    """Should return metrics."""
    mock_forecaster = MagicMock()
    mock_forecaster.predict.return_value = pd.Series([10.0])
    models["forecaster"] = mock_forecaster

    mock_df = pd.DataFrame(
        {"date": [pd.Timestamp("2024-01-01")], "city": ["NYC"], "orders": [10.0]}
    )
    models["latest_df"] = mock_df

    mock_compute.return_value = {"mape": 5.0}

    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "global" in data
    assert "cities" in data
    assert data["global"] == {"mape": 5.0}
    assert data["cities"]["NYC"] == {"mape": 5.0}


def test_predict_no_models_raises_503(client: TestClient) -> None:
    """Should raise 503 if forecaster not loaded."""
    response = client.post("/predict", json={"instances": []})
    assert response.status_code == 503


def test_predict_success(client: TestClient) -> None:
    """Should return predictions."""
    mock_forecaster = MagicMock()
    mock_forecaster.predict_intervals.return_value = {
        "mean": pd.Series([100.0]),
        "p10": pd.Series([90.0]),
        "p90": pd.Series([110.0]),
    }
    models["forecaster"] = mock_forecaster

    payload = {
        "instances": [
            {
                "date": "2024-01-01",
                "city": "NYC",
                "temperature_c": 15.0,
                "rainfall_mm": 0.0,
                "is_rainy": False,
                "promo_active": 0,
                "is_holiday": 0,
            }
        ]
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["pred_mean"] == 100.0


def test_forecast_no_models_raises_503(client: TestClient) -> None:
    """Should raise 503 if models not loaded."""
    response = client.post("/forecast", json={"horizon_days": 10})
    assert response.status_code == 503


def test_forecast_success(client: TestClient) -> None:
    """Should return forecast."""
    mock_forecaster = MagicMock()
    mock_future_df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "city": ["NYC"],
            "pred_orders": [100.0],
            "pred_p10": [90.0],
            "pred_p90": [110.0],
        }
    )
    mock_forecaster.forecast_future.return_value = mock_future_df
    models["forecaster"] = mock_forecaster

    models["latest_df"] = pd.DataFrame({"city": ["NYC"]})

    response = client.post("/forecast", json={"horizon_days": 10, "use_live_weather": False})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "NYC" in data["city_forecasts"]
    assert len(data["city_forecasts"]["NYC"]) == 1
