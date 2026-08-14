"""Live weather data fetcher using Open-Meteo for future demand forecasting."""

import json
import logging
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Cache directory to avoid rate limiting and excessive requests
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "data" / "cache"


def _get_city_coordinates(city: str) -> tuple[float, float]:
    """Return hardcoded coordinates for the 5 supported logistics markets.

    Args:
        city: The city name.

    Returns:
        tuple: (latitude, longitude)
    """
    coords = {
        "New York": (40.7128, -74.0060),
        "Chicago": (41.8781, -87.6298),
        "Los Angeles": (34.0522, -118.2437),
        "Austin": (30.2672, -97.7431),
        "Miami": (25.7617, -80.1918),
    }
    return coords.get(city, coords["New York"])


def fetch_forecast_weather(city: str, horizon_days: int) -> pd.DataFrame:
    """Fetch live 14-day weather forecast for a city using Open-Meteo.

    If the requested horizon is longer than 14 days, it will backfill the remaining
    days with the historical average of the fetched data.

    Args:
        city: City name.
        horizon_days: Number of days to forecast into the future.

    Returns:
        pd.DataFrame: DataFrame with 'date', 'city', 'temperature_c', 'rainfall_mm', 'is_rainy'.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    today_str = date.today().isoformat()
    cache_file = _CACHE_DIR / f"weather_{city.replace(' ', '_')}_{today_str}.json"

    lat, lon = _get_city_coordinates(city)

    data = None
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                data = json.load(f)
            logger.info("Loaded weather forecast for %s from cache.", city)
        except Exception as e:
            logger.warning("Failed to load weather cache for %s: %s", city, e)

    if data is None:
        # Open-Meteo Free API (No key required)
        # 16-day forecast max
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&daily=temperature_2m_mean,precipitation_sum&timezone=auto&forecast_days=16"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "DemandIQ-ML"})
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    with open(cache_file, "w") as f:
                        json.dump(data, f)
                    logger.info("Fetched live weather forecast for %s.", city)
                else:
                    logger.error("Weather API returned status %s for %s", response.status, city)
        except Exception as e:
            logger.error("Exception fetching weather for %s: %s", city, e)

    # Process data into DataFrame
    if data and "daily" in data:
        daily = data["daily"]
        dates = daily.get("time", [])
        temps = daily.get("temperature_2m_mean", [])
        precip = daily.get("precipitation_sum", [])

        rows = []
        for d, t, p in zip(dates, temps, precip, strict=False):
            rows.append(
                {
                    "date": pd.to_datetime(d),
                    "city": city,
                    "temperature_c": float(t) if t is not None else 15.0,
                    "rainfall_mm": float(p) if p is not None else 0.0,
                    "is_rainy": p is not None and float(p) > 2.0,
                }
            )
        df = pd.DataFrame(rows)
    else:
        # Fallback if API fails completely
        df = pd.DataFrame(columns=["date", "city", "temperature_c", "rainfall_mm", "is_rainy"])

    # Extrapolate if horizon is longer than 16 days
    if not df.empty and horizon_days > len(df):
        last_date = df["date"].max()
        avg_temp = df["temperature_c"].mean()
        avg_rain = df["rainfall_mm"].mean()

        extra_days = horizon_days - len(df)
        extra_rows = []
        for i in range(1, extra_days + 1):
            extra_rows.append(
                {
                    "date": last_date + pd.Timedelta(days=i),
                    "city": city,
                    "temperature_c": avg_temp,
                    "rainfall_mm": avg_rain,
                    "is_rainy": avg_rain > 2.0,
                }
            )
        df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

    return df
