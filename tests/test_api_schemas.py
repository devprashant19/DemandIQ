"""Unit tests for the API schemas."""

from datetime import date

import pytest
from pydantic import ValidationError

from demandiq.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
    PredictionRow,
)


def test_forecast_request_valid() -> None:
    """Should accept valid horizons."""
    req = ForecastRequest(horizon_days=10, use_live_weather=True)
    assert req.horizon_days == 10
    assert req.use_live_weather is True


def test_forecast_request_invalid_horizon() -> None:
    """Should reject horizons outside 1-90."""
    with pytest.raises(ValidationError):
        ForecastRequest(horizon_days=0)
    with pytest.raises(ValidationError):
        ForecastRequest(horizon_days=91)


def test_prediction_row_defaults() -> None:
    """Should apply correct default values."""
    row = PredictionRow(date=date(2024, 1, 1), city="NYC")
    assert row.temperature_c == 15.0
    assert row.is_rainy is False
    assert row.promo_active == 0


def test_predict_batch_request() -> None:
    """Should validate a batch request."""
    row = PredictionRow(date=date(2024, 1, 1), city="NYC")
    req = PredictionRequest(instances=[row])
    assert len(req.instances) == 1
    assert req.instances[0].city == "NYC"


def test_predict_batch_response() -> None:
    """Should validate a batch response."""
    res = PredictionResponse(
        status="success",
        predictions=[
            PredictionResult(
                date="2024-01-01",
                city="NYC",
                pred_mean=100.0,
                pred_p10=90.0,
                pred_p90=110.0,
            )
        ],
    )
    assert len(res.predictions) == 1
    assert res.predictions[0].pred_mean == 100.0


def test_forecast_response() -> None:
    """Should validate a forecast response."""
    res = ForecastResponse(status="ok", city_forecasts={"NYC": [{"date": "2024-01-01"}]})
    assert res.status == "ok"
    assert "NYC" in res.city_forecasts
