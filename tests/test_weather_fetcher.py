"""Unit tests for the weather fetcher module."""

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

from demandiq.data.weather_fetcher import _get_city_coordinates, fetch_forecast_weather


def test_get_city_coordinates() -> None:
    """Should return correct coordinates for known cities."""
    lat, lon = _get_city_coordinates("New York")
    assert lat == 40.7128
    assert lon == -74.0060

    lat, lon = _get_city_coordinates("Chicago")
    assert lat == 41.8781

    # Default fallback for unknown city
    lat, lon = _get_city_coordinates("Unknown City")
    assert lat == 40.7128
    assert lon == -74.0060


def test_fetch_forecast_weather_cached(tmp_path: Path) -> None:
    """Should load from cache if today cache file exists."""
    today_str = date.today().isoformat()
    # The function builds: _CACHE_DIR / f"weather_{city.replace(' ', '_')}_{today_str}.json"
    cache_file = tmp_path / f"weather_NYC_{today_str}.json"

    with open(cache_file, "w") as f:
        json.dump(
            {
                "daily": {
                    "time": ["2024-01-01"],
                    "temperature_2m_mean": [20.0],
                    "precipitation_sum": [5.0],
                }
            },
            f,
        )

    with patch("demandiq.data.weather_fetcher._CACHE_DIR", tmp_path):
        df = fetch_forecast_weather("NYC", horizon_days=1)

    assert len(df) == 1
    assert df["temperature_c"].iloc[0] == 20.0
    assert df["rainfall_mm"].iloc[0] == 5.0
    assert df["is_rainy"].iloc[0]  # 5.0 > 2.0


def test_fetch_forecast_weather_api_success(tmp_path: Path) -> None:
    """Should fetch from API when no cache exists, parse and return DataFrame."""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = json.dumps(
        {
            "daily": {
                "time": ["2024-01-01", "2024-01-02"],
                "temperature_2m_mean": [15.0, 10.0],
                "precipitation_sum": [0.0, 3.0],
            }
        }
    ).encode("utf-8")

    with (
        patch("demandiq.data.weather_fetcher._CACHE_DIR", tmp_path),
        patch("urllib.request.urlopen") as mock_urlopen,
    ):
        mock_urlopen.return_value.__enter__ = MagicMock(return_value=mock_response)
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)
        df = fetch_forecast_weather("NYC", horizon_days=2)

    assert len(df) == 2
    assert df["temperature_c"].iloc[0] == 15.0
    assert df["rainfall_mm"].iloc[1] == 3.0
    assert df["is_rainy"].iloc[1]  # 3.0 > 2.0 � use truthiness, not `is True` (numpy bool_)
    assert not df["is_rainy"].iloc[0]  # 0.0 not rainy


def test_fetch_forecast_weather_extrapolation(tmp_path: Path) -> None:
    """Should extrapolate with averages when horizon > fetched days."""
    today_str = date.today().isoformat()
    cache_file = tmp_path / f"weather_NYC_{today_str}.json"

    with open(cache_file, "w") as f:
        json.dump(
            {
                "daily": {
                    "time": ["2024-01-01", "2024-01-02"],
                    "temperature_2m_mean": [10.0, 20.0],  # mean = 15.0
                    "precipitation_sum": [0.0, 4.0],  # mean = 2.0
                }
            },
            f,
        )

    with patch("demandiq.data.weather_fetcher._CACHE_DIR", tmp_path):
        df = fetch_forecast_weather("NYC", horizon_days=4)

    assert len(df) == 4
    assert df["temperature_c"].iloc[2] == 15.0
    assert df["temperature_c"].iloc[3] == 15.0
    assert df["rainfall_mm"].iloc[2] == 2.0


def test_fetch_forecast_weather_api_failure_returns_empty(tmp_path: Path) -> None:
    """Should return empty DataFrame if API call raises an exception."""
    with (
        patch("demandiq.data.weather_fetcher._CACHE_DIR", tmp_path),
        patch("urllib.request.urlopen", side_effect=Exception("Network error")),
    ):
        df = fetch_forecast_weather("NYC", horizon_days=5)

    assert df.empty


def test_fetch_forecast_weather_none_values_handled(tmp_path: Path) -> None:
    """Should use fallback values (15.0 / 0.0) when API returns None for a day."""
    today_str = date.today().isoformat()
    cache_file = tmp_path / f"weather_NYC_{today_str}.json"

    with open(cache_file, "w") as f:
        json.dump(
            {
                "daily": {
                    "time": ["2024-01-01", "2024-01-02"],
                    "temperature_2m_mean": [None, 25.0],
                    "precipitation_sum": [None, 0.0],
                }
            },
            f,
        )

    with patch("demandiq.data.weather_fetcher._CACHE_DIR", tmp_path):
        df = fetch_forecast_weather("NYC", horizon_days=2)

    assert len(df) == 2
    assert df["temperature_c"].iloc[0] == 15.0  # fallback
    assert df["rainfall_mm"].iloc[0] == 0.0  # fallback
