"""Synthetic demand dataset generator for food-delivery business modeled across multiple cities."""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from demandiq.config import settings

logger = logging.getLogger(__name__)

DEFAULT_CITIES: dict[str, float] = {
    "New York": 5000.0,
    "Chicago": 3500.0,
    "Los Angeles": 4500.0,
    "Austin": 2000.0,
    "Miami": 2800.0,
}


def generate_synthetic_data(
    n_years: int = 3,
    n_days: int | None = None,
    seed: int = 42,
    cities: list[str] | dict[str, float] | None = None,
    end_date: str = "2023-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate deterministic synthetic daily food-delivery order volume across cities.

    Args:
        n_years (int): Number of historical years to simulate if n_days is None.
        n_days (int | None): Exact number of historical days to generate per city.
        seed (int): Random seed for NumPy reproducibility.
        cities (list[str] | dict[str, float] | None): Cities to include with optional base volumes.
        end_date (str): End date string in YYYY-MM-DD format.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple containing (orders_df, ground_truth_anomalies_df).
    """
    rng = np.random.default_rng(seed)

    if cities is None:
        city_bases = DEFAULT_CITIES
    elif isinstance(cities, list):
        city_bases = {city: DEFAULT_CITIES.get(city, 3000.0) for city in cities}
    else:
        city_bases = cities

    total_days = n_days if n_days is not None else int(n_years * 365)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=total_days - 1)
    date_range = pd.date_range(start=start_dt, end=end_dt, freq="D")

    records = []
    anomaly_records = []

    for city, base_vol in city_bases.items():
        # Temperature modeling: sinusoidal variations per season
        temp_offset = 20.0 if city in ["Miami", "Los Angeles", "Austin"] else 12.0
        temp_amp = 8.0 if city in ["Miami", "Los Angeles"] else 15.0
        day_of_year = date_range.day_of_year.to_numpy()
        temps = (
            temp_offset
            + temp_amp * np.sin(2 * np.pi * (day_of_year - 105) / 365.25)
            + rng.normal(0, 3.0, size=total_days)
        )

        # Rainfall modeling
        rain_prob = 0.3 if city in ["Miami", "New York"] else 0.15
        is_raining = rng.random(size=total_days) < rain_prob
        rainfall = np.where(is_raining, rng.exponential(scale=8.0, size=total_days), 0.0)
        is_rainy_flag = (rainfall > 2.0).astype(int)

        # Calendar and promo features
        dow = date_range.dayofweek.to_numpy()
        is_weekend = (dow >= 5).astype(int)
        is_friday = (dow == 4).astype(int)

        # Deterministic simple holiday check (Jan 1, Jul 4, Dec 25, Nov Thanksgiving approx)
        month = date_range.month.to_numpy()
        day = date_range.day.to_numpy()
        is_holiday = (
            ((month == 1) & (day == 1))
            | ((month == 7) & (day == 4))
            | ((month == 12) & (day == 25))
            | ((month == 11) & (day >= 22) & (day <= 28) & (dow == 3))
        ).astype(int)

        promo_active = (rng.random(size=total_days) < 0.18).astype(int)
        festival_flag = ((is_weekend == 1) & (rng.random(size=total_days) < 0.20)).astype(int)

        # Baseline seasonality & linear growth trend
        time_index = np.arange(total_days)
        growth_trend = 1.0 + 0.25 * (time_index / 1095.0)  # ~25% growth over 3 years
        weekly_seasonality = 1.0 + 0.25 * is_weekend + 0.15 * is_friday
        yearly_seasonality = 1.0 + 0.10 * np.cos(2 * np.pi * (day_of_year - 20) / 365.25)

        # Combine demand drivers
        expected_demand = (
            base_vol
            * growth_trend
            * weekly_seasonality
            * yearly_seasonality
            * (1.0 + 0.18 * promo_active)
            * (1.0 + 0.12 * festival_flag)
            * (1.0 + 0.10 * is_rainy_flag)
            * (1.0 - 0.20 * is_holiday)
        )

        # Add Gaussian noise
        noise_std = np.maximum(0.05 * expected_demand, 50.0)
        simulated_orders = rng.normal(loc=expected_demand, scale=noise_std)

        # Anomaly injection (~1.5% of days)
        anomaly_mask = rng.random(size=total_days) < 0.015
        is_spike = rng.random(size=total_days) > 0.5
        spike_multiplier = rng.uniform(2.1, 3.2, size=total_days)
        drop_multiplier = rng.uniform(0.15, 0.4, size=total_days)

        final_orders = np.where(
            anomaly_mask,
            np.where(
                is_spike, simulated_orders * spike_multiplier, simulated_orders * drop_multiplier
            ),
            simulated_orders,
        )

        # Ensure strict positivity and integer formatting
        final_orders_int = np.maximum(np.round(final_orders), 10).astype(int)

        for i in range(total_days):
            dt_str = date_range[i]
            records.append(
                {
                    "date": dt_str,
                    "city": city,
                    "orders": int(final_orders_int[i]),
                    "temperature_c": float(np.round(temps[i], 2)),
                    "rainfall_mm": float(np.round(rainfall[i], 2)),
                    "is_rainy": int(is_rainy_flag[i]),
                    "is_holiday": int(is_holiday[i]),
                    "festival_flag": int(festival_flag[i]),
                    "promo_active": int(promo_active[i]),
                }
            )
            anomaly_records.append(
                {
                    "date": dt_str,
                    "city": city,
                    "is_anomaly": int(anomaly_mask[i]),
                }
            )

    orders_df = pd.DataFrame(records)
    orders_df["date"] = pd.to_datetime(orders_df["date"])
    orders_df = orders_df.sort_values(by=["date", "city"]).reset_index(drop=True)

    ground_truth_df = pd.DataFrame(anomaly_records)
    ground_truth_df["date"] = pd.to_datetime(ground_truth_df["date"])
    ground_truth_df = ground_truth_df.sort_values(by=["date", "city"]).reset_index(drop=True)

    return orders_df, ground_truth_df


def save_synthetic_data(
    out_path: Path | str,
    anomalies_out_path: Path | str | None = None,
    n_years: int = 3,
    seed: int = 42,
) -> None:
    """Generate and export synthetic orders dataset and hidden ground-truth anomalies to CSV.

    Args:
        out_path (Path | str): Output path for orders dataset.
        anomalies_out_path (Path | str | None): Output path for ground-truth anomalies.
        n_years (int): Number of years of historical data to generate.
        seed (int): Random seed for determinism.
    """
    orders_df, ground_truth_df = generate_synthetic_data(n_years=n_years, seed=seed)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    orders_df.to_csv(out_p, index=False)
    logger.info(
        "Successfully exported synthetic orders data (%d rows) to %s", len(orders_df), out_p
    )

    anom_p = (
        Path(anomalies_out_path) if anomalies_out_path else settings.ground_truth_anomalies_path
    )
    anom_p.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_df.to_csv(anom_p, index=False)
    logger.info("Successfully exported true anomaly labels to %s", anom_p)


def main() -> None:  # pragma: no cover
    """CLI execution handler for running synthetic dataset generation."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Generate synthetic food-delivery orders data.")
    parser.add_argument(
        "--out", type=str, default=str(settings.raw_orders_path), help="Output path"
    )
    parser.add_argument(
        "--anomalies-out",
        type=str,
        default=str(settings.ground_truth_anomalies_path),
        help="Anomalies out",
    )
    parser.add_argument("--years", type=int, default=3, help="Number of historical years")
    parser.add_argument(
        "--seed", type=int, default=settings.random_seed, help="Random reproducibility seed"
    )

    args = parser.parse_args()
    save_synthetic_data(
        out_path=args.out, anomalies_out_path=args.anomalies_out, n_years=args.years, seed=args.seed
    )


if __name__ == "__main__":  # pragma: no cover
    main()
