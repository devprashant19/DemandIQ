"""Unit tests for the PromoCalendar module."""

from datetime import date
from pathlib import Path

import pandas as pd

from demandiq.data.promo_calendar import PromoCalendar


class TestPromoCalendar:
    """Tests for the PromoCalendar class."""

    def test_add_promo_string_dates(self) -> None:
        """Should accept string dates and store as an entry."""
        cal = PromoCalendar()
        cal.add_promo("NYC", "2026-09-01", "2026-09-07", intensity=1.5, label="Summer Sale")
        assert len(cal) == 1

    def test_add_promo_date_objects(self) -> None:
        """Should accept date objects and store as an entry."""
        cal = PromoCalendar()
        cal.add_promo("NYC", date(2026, 9, 1), date(2026, 9, 7), intensity=1.2)
        assert len(cal) == 1

    def test_add_multiple_promos(self) -> None:
        """Should accumulate multiple promo entries."""
        cal = PromoCalendar()
        cal.add_promo("NYC", "2026-09-01", "2026-09-07")
        cal.add_promo("Chicago", "2026-10-01", "2026-10-14")
        assert len(cal) == 2

    def test_repr(self) -> None:
        """Should return a meaningful string representation."""
        cal = PromoCalendar()
        cal.add_promo("NYC", "2026-09-01", "2026-09-07")
        assert "1" in repr(cal)

    def test_to_dataframe_no_promos(self) -> None:
        """Should return DataFrame with promo_active=0 when no promos are active."""
        cal = PromoCalendar()
        dates = pd.date_range("2026-01-01", periods=7)
        result = cal.to_dataframe(dates, cities=["NYC"])
        assert "promo_active" in result.columns
        assert (result["promo_active"] == 0).all()

    def test_to_dataframe_with_matching_promo(self) -> None:
        """Should mark promo_active=1 during promo periods."""
        cal = PromoCalendar()
        cal.add_promo("NYC", "2026-09-01", "2026-09-03", intensity=1.5)
        dates = pd.date_range("2026-09-01", periods=5)
        result = cal.to_dataframe(dates, cities=["NYC"])
        assert result.loc[result["date"].dt.date == date(2026, 9, 1), "promo_active"].values[0] == 1
        assert result.loc[result["date"].dt.date == date(2026, 9, 5), "promo_active"].values[0] == 0

    def test_to_dataframe_wrong_city(self) -> None:
        """Should not flag promo_active for wrong city."""
        cal = PromoCalendar()
        cal.add_promo("Chicago", "2026-09-01", "2026-09-07")
        dates = pd.date_range("2026-09-01", periods=5)
        result = cal.to_dataframe(dates, cities=["NYC"])
        assert (result["promo_active"] == 0).all()

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Should persist and reload entries from JSON."""
        cal = PromoCalendar()
        cal.add_promo("NYC", "2026-09-01", "2026-09-07", intensity=1.5)
        save_path = tmp_path / "calendar.json"
        cal.save(path=save_path)
        assert save_path.exists()

        loaded = PromoCalendar.load(path=save_path)
        assert len(loaded) == 1

    def test_load_nonexistent_file_returns_empty(self, tmp_path: Path) -> None:
        """Should return empty calendar when file does not exist."""
        cal = PromoCalendar.load(path=tmp_path / "nonexistent.json")
        assert len(cal) == 0
