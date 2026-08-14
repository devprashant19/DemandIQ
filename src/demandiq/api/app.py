"""FastAPI application for DemandIQ inference and monitoring."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from demandiq.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)
from demandiq.config import settings
from demandiq.data.loader import load_and_validate_orders
from demandiq.features.engineer import build_features
from demandiq.models.cross_validate import compute_metrics
from demandiq.models.forecaster import DemandForecaster

logger = logging.getLogger(__name__)

# Global model state
models: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model artifacts during API startup."""
    try:
        forecaster = DemandForecaster.load(settings.forecaster_model_path)
        models["forecaster"] = forecaster
        logger.info("Successfully loaded forecaster into API memory.")

        # Load the latest dataset to check freshness
        raw_df = load_and_validate_orders(settings.raw_orders_path)
        models["latest_df"] = build_features(raw_df)
        logger.info("Successfully loaded reference dataset into API memory.")
    except Exception as e:
        logger.error("Failed to load models during API startup: %s", e)
    yield
    # Cleanup on shutdown
    models.clear()


app = FastAPI(
    title="DemandIQ REST API",
    description="Production ML Forecasting & Anomaly Engine API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Check API and Model health."""
    is_fitted = False
    age_days = -1.0
    data_stale = -1

    forecaster = models.get("forecaster")
    if forecaster:
        is_fitted = forecaster.is_fitted
        try:
            stat = settings.forecaster_model_path.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime)
            age = datetime.now() - mtime
            age_days = float(age.total_seconds() / 86400.0)
        except Exception:
            pass

    latest_df = models.get("latest_df")
    if latest_df is not None and not latest_df.empty:
        max_date = pd.to_datetime(latest_df["date"].max())
        data_stale = (datetime.now() - max_date).days

    return HealthResponse(
        status="ok" if is_fitted else "degraded",
        model_age_days=age_days,
        is_fitted=is_fitted,
        data_freshness_days=data_stale,
    )


@app.get("/metrics")
def get_metrics() -> dict[str, Any]:
    """Retrieve current performance metrics across all cities."""
    forecaster = models.get("forecaster")
    df = models.get("latest_df")

    if not forecaster or df is None or df.empty:
        raise HTTPException(status_code=503, detail="Models or data not loaded")

    try:
        preds = forecaster.predict(df)
        global_metrics = compute_metrics(df["orders"].to_numpy(), preds)

        city_metrics = {}
        for city in df["city"].unique():
            cdf = df[df["city"] == city]
            c_preds = forecaster.predict(cdf)
            city_metrics[str(city)] = compute_metrics(cdf["orders"].to_numpy(), c_preds)

        return {"global": global_metrics, "cities": city_metrics}
    except Exception as e:
        logger.error("Metrics computation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict", response_model=PredictionResponse)
def predict_points(request: PredictionRequest) -> PredictionResponse:
    """Generate point predictions with confidence intervals for given feature rows."""
    forecaster = models.get("forecaster")
    if not forecaster:
        raise HTTPException(status_code=503, detail="Forecaster not loaded")

    try:
        # Convert request to DataFrame
        rows = [i.model_dump() for i in request.instances]
        req_df = pd.DataFrame(rows)
        req_df["date"] = pd.to_datetime(req_df["date"])

        # Predict
        intervals = forecaster.predict_intervals(req_df)

        results = []
        for idx, row in req_df.iterrows():
            results.append(
                PredictionResult(
                    date=row["date"].strftime("%Y-%m-%d"),
                    city=row["city"],
                    pred_mean=float(intervals["mean"][idx]),
                    pred_p10=float(intervals["p10"][idx]),
                    pred_p90=float(intervals["p90"][idx]),
                )
            )

        return PredictionResponse(status="success", predictions=results)
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/forecast", response_model=ForecastResponse)
def forecast_future(request: ForecastRequest) -> ForecastResponse:
    """Generate a multi-day future demand forecast."""
    forecaster = models.get("forecaster")
    df = models.get("latest_df")

    if not forecaster or df is None or df.empty:
        raise HTTPException(status_code=503, detail="Models or data not loaded")

    try:
        weather_df = None
        if request.use_live_weather:
            # Import dynamically to avoid circular dependencies if any
            from demandiq.data.weather_fetcher import fetch_forecast_weather

            cities = df["city"].unique()
            w_dfs = []
            for city in cities:
                w_df = fetch_forecast_weather(str(city), request.horizon_days)
                if not w_df.empty:
                    w_dfs.append(w_df)
            if w_dfs:
                weather_df = pd.concat(w_dfs, ignore_index=True)

        future_df = forecaster.forecast_future(
            horizon_days=request.horizon_days, last_known_df=df, weather_df=weather_df
        )

        # Format response
        city_forecasts = {}
        for city in future_df["city"].unique():
            cdf = future_df[future_df["city"] == city]

            records = []
            for _, row in cdf.iterrows():
                records.append(
                    {
                        "date": row["date"].strftime("%Y-%m-%d"),
                        "pred_mean": float(row.get("pred_orders", 0.0)),
                        "pred_p10": float(row.get("pred_p10", 0.0)),
                        "pred_p90": float(row.get("pred_p90", 0.0)),
                    }
                )
            city_forecasts[str(city)] = records

        return ForecastResponse(status="success", city_forecasts=city_forecasts)
    except Exception as e:
        logger.error("Future forecasting failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
