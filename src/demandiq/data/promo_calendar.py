"""Promotional calendar manager for injecting future promotional campaigns into demand forecasts."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_CALENDAR_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "data"
    / "promos"
    / "calendar.json"
)


class PromoCalendar:
    """Manages promotional campaign schedules per city for forward-looking demand forecasting.

    Allows operators to inject known future promotions so that forecast_future() can
    accurately model promotional uplift rather than defaulting promo_active=0.

    Example::

        cal = PromoCalendar()
        cal.add_promo("New York", "2026-09-01", "2026-09-07", intensity=1.5)
        promo_df = cal.to_dataframe(pd.date_range("2026-09-01", periods=30))
    """

    def __init__(self) -> None:
        """Initialize an empty promotional calendar."""
        self._entries: list[dict[str, Any]] = []

    def add_promo(
        self,
        city: str,
        start_date: str | date,
        end_date: str | date,
        intensity: float = 1.0,
        label: str = "",
    ) -> None:
        """Add a promotional period for a given city.

        Args:
            city (str): Target city name (must match data city values).
            start_date (str | date): Promo start date (inclusive), ISO format or date object.
            end_date (str | date): Promo end date (inclusive), ISO format or date object.
            intensity (float): Multiplier representing promo strength (1.0 = standard, >1 = stronger).
            label (str): Optional descriptive label for this campaign (e.g., "Summer Sale").
        """
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)

        if end_date < start_date:
            raise ValueError(f"end_date ({end_date}) must be >= start_date ({start_date}).")
        if intensity <= 0:
            raise ValueError(f"intensity must be positive, got {intensity}.")

        entry = {
            "city": str(city),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "intensity": float(intensity),
            "label": str(label),
        }
        self._entries.append(entry)
        logger.info(
            "Added promo for %s: %s → %s (intensity=%.2f, label='%s')",
            city,
            start_date,
            end_date,
            intensity,
            label,
        )

    def remove_promo(self, city: str, start_date: str | date) -> int:
        """Remove all promos matching a given city and start date.

        Args:
            city (str): City name to match.
            start_date (str | date): Start date to match.

        Returns:
            int: Number of entries removed.
        """
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)
        before = len(self._entries)
        self._entries = [
            e
            for e in self._entries
            if not (e["city"] == city and e["start_date"] == start_date.isoformat())
        ]
        removed = before - len(self._entries)
        logger.info("Removed %d promo entries for %s on %s.", removed, city, start_date)
        return removed

    def list_promos(self, city: str | None = None) -> list[dict[str, Any]]:
        """Return all promotional entries, optionally filtered by city.

        Args:
            city (str | None): If specified, only return promos for that city.

        Returns:
            list[dict]: List of promo entry dicts sorted by start_date.
        """
        entries = self._entries if city is None else [e for e in self._entries if e["city"] == city]
        return sorted(entries, key=lambda x: x["start_date"])

    def to_dataframe(
        self,
        date_range: pd.DatetimeIndex | None = None,
        cities: list[str] | None = None,
    ) -> pd.DataFrame:
        """Convert promo calendar to a long-format DataFrame aligned to a date range.

        Args:
            date_range (pd.DatetimeIndex | None): Date range to generate rows for. If None,
                derives from the union of all promo periods.
            cities (list[str] | None): List of cities to include. If None, infers from entries.

        Returns:
            pd.DataFrame: DataFrame with columns ['date', 'city', 'promo_active', 'promo_intensity'].
                promo_active is 1 on promotional days and 0 otherwise.
        """
        if not self._entries and date_range is None:
            return pd.DataFrame(columns=["date", "city", "promo_active", "promo_intensity"])

        # Build date range from entries if not supplied
        if date_range is None:
            all_starts = [date.fromisoformat(e["start_date"]) for e in self._entries]
            all_ends = [date.fromisoformat(e["end_date"]) for e in self._entries]
            date_range = pd.date_range(min(all_starts), max(all_ends))

        # Infer cities
        if cities is None:
            cities = sorted({e["city"] for e in self._entries})

        rows = []
        for city in cities:
            city_entries = [e for e in self._entries if e["city"] == city]
            for dt in date_range:
                d = dt.date()
                promo_active = 0
                promo_intensity = 0.0
                for entry in city_entries:
                    s = date.fromisoformat(entry["start_date"])
                    end = date.fromisoformat(entry["end_date"])
                    if s <= d <= end:
                        promo_active = 1
                        promo_intensity = max(promo_intensity, entry["intensity"])
                rows.append(
                    {
                        "date": pd.Timestamp(dt),
                        "city": city,
                        "promo_active": promo_active,
                        "promo_intensity": promo_intensity,
                    }
                )

        return pd.DataFrame(rows)

    def save(self, path: Path | str | None = None) -> None:
        """Persist calendar to a JSON file.

        Args:
            path (Path | str | None): Destination file path. Defaults to data/promos/calendar.json.
        """
        save_p = Path(path) if path is not None else _DEFAULT_CALENDAR_PATH
        save_p.parent.mkdir(parents=True, exist_ok=True)
        with open(save_p, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, indent=2)
        logger.info("Saved PromoCalendar with %d entries to %s.", len(self._entries), save_p)

    @classmethod
    def load(cls, path: Path | str | None = None) -> PromoCalendar:
        """Load a PromoCalendar from a JSON file.

        Args:
            path (Path | str | None): Source file path. Defaults to data/promos/calendar.json.

        Returns:
            PromoCalendar: Populated calendar instance.

        Raises:
            FileNotFoundError: If the calendar file does not exist.
        """
        load_p = Path(path) if path is not None else _DEFAULT_CALENDAR_PATH
        if not load_p.exists():
            logger.info("No existing calendar found at %s. Returning empty calendar.", load_p)
            return cls()
        with open(load_p, encoding="utf-8") as f:
            entries = json.load(f)
        cal = cls()
        cal._entries = entries
        logger.info("Loaded PromoCalendar with %d entries from %s.", len(entries), load_p)
        return cal

    def apply_to_future_df(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """Overlay promo flags onto a future forecast DataFrame in-place.

        Merges the promo calendar onto a DataFrame containing 'date' and 'city' columns,
        updating the 'promo_active' and optionally 'promo_intensity' columns.

        Args:
            future_df (pd.DataFrame): Future forecast DataFrame with 'date' and 'city' columns.

        Returns:
            pd.DataFrame: Updated DataFrame with promo_active and promo_intensity columns set.
        """
        if future_df.empty or not self._entries:
            return future_df

        cities = future_df["city"].unique().tolist()
        date_range = pd.DatetimeIndex(future_df["date"].unique())
        promo_df = self.to_dataframe(date_range=date_range, cities=cities)

        if promo_df.empty:
            return future_df

        out = future_df.copy()
        promo_df["date"] = pd.to_datetime(promo_df["date"])
        out["date"] = pd.to_datetime(out["date"])

        merged = out.merge(
            promo_df[["date", "city", "promo_active", "promo_intensity"]],
            on=["date", "city"],
            how="left",
            suffixes=("", "_cal"),
        )
        if "promo_active_cal" in merged.columns:
            cal_mask = merged["promo_active_cal"].notna()
            merged.loc[cal_mask, "promo_active"] = merged.loc[cal_mask, "promo_active_cal"].astype(
                int
            )
            merged = merged.drop(columns=["promo_active_cal"])
        if "promo_intensity" in merged.columns:
            merged["promo_intensity"] = merged["promo_intensity"].fillna(0.0)

        return merged.reset_index(drop=True)

    def __len__(self) -> int:
        """Return the number of promos."""
        return len(self._entries)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PromoCalendar(entries={len(self._entries)})"
