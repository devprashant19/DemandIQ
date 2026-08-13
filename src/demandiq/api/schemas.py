"""Pydantic schemas for the DemandIQ FastAPI application."""

from typing import Any
from datetime import date
from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Request schema for future forecasting."""
    horizon_days: int = Field(default=14, ge=1, le=90, description="Number of days to forecast into the future")
    use_live_weather: bool = Field(default=False, description="Whether to fetch and use live weather data via Open-Meteo")

class ForecastResponse(BaseModel):
    """Response schema for future forecasting."""
    status: str
    city_forecasts: dict[str, list[dict[str, Any]]]

class PredictionRow(BaseModel):
    """Schema for a single row of prediction input."""
    date: date
    city: str
    orders_lag_1: float | None = None
    orders_lag_7: float | None = None
    # Weather
    temperature_c: float = 15.0
    rainfall_mm: float = 0.0
    is_rainy: bool = False
    # Promo/Holiday
    promo_active: int = 0
    is_holiday: int = 0
    festival_flag: int = 0

class PredictionRequest(BaseModel):
    """Request schema for point predictions."""
    instances: list[PredictionRow]

class PredictionResult(BaseModel):
    """Schema for a single prediction result."""
    date: str
    city: str
    pred_mean: float
    pred_p10: float
    pred_p90: float

class PredictionResponse(BaseModel):
    """Response schema for point predictions."""
    status: str
    predictions: list[PredictionResult]

class HealthResponse(BaseModel):
    """Response schema for the health endpoint."""
    status: str
    model_age_days: float
    is_fitted: bool
    data_freshness_days: int
