"""LightGBM and Prophet hybrid ensemble forecasting engine."""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from demandiq.config import settings
from demandiq.features.engineer import FEATURE_COLUMNS, build_features

logger = logging.getLogger(__name__)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from prophet import Prophet


class DemandForecaster:
    """Ensemble demand forecaster combining LightGBM and Prophet per city."""

    def __init__(
        self,
        lgb_weight: float = settings.lgb_weight,
        prophet_weight: float = settings.prophet_weight,
        quantile_alphas: list[float] | None = None,
        random_seed: int = settings.random_seed,
    ) -> None:
        """Initialize ensemble demand forecaster with weights and seed.

        Args:
            lgb_weight (float): Relative weighting assigned to LightGBM predictions.
            prophet_weight (float): Relative weighting assigned to Prophet predictions.
            quantile_alphas (list[float] | None): Target quantiles for uncertainty intervals.
            random_seed (int): Reproducibility random seed for tree estimations.
        """
        self.lgb_weight = lgb_weight
        self.prophet_weight = prophet_weight
        self.quantile_alphas = (
            quantile_alphas if quantile_alphas is not None else settings.quantile_alphas
        )
        self.random_seed = random_seed
        self._lgb_models: dict[str, lgb.LGBMRegressor] = {}
        self._lgb_quantile_models: dict[str, dict[float, lgb.LGBMRegressor]] = {}
        self._prophet_models: dict[str, Prophet] = {}
        self._feature_cols: list[str] = FEATURE_COLUMNS.copy()
        self.is_fitted: bool = False

    def fit(
        self, X: pd.DataFrame, y: pd.Series[Any] | np.ndarray[Any, Any] | None = None
    ) -> DemandForecaster:
        """Fit individual LightGBM and Prophet models per city without code duplication.

        Args:
            X (pd.DataFrame): Training dataframe containing date, city, target orders, and features.
            y (pd.Series | np.ndarray | None): Optional explicit target orders series. If None,
                extracts 'orders' column from X.

        Returns:
            DemandForecaster: Fitted instance of self.

        Raises:
            ValueError: If required columns or targets are missing from training dataset.
        """
        train_df = X.copy()
        if "date" not in train_df.columns or "city" not in train_df.columns:
            raise ValueError("Training DataFrame must contain 'date' and 'city' columns.")

        if y is not None:
            train_df["orders"] = np.array(y)
        elif "orders" not in train_df.columns:
            raise ValueError("Target column 'orders' must be present in X if y is None.")

        # Ensure feature engineering has been executed
        missing_features = [c for c in self._feature_cols if c not in train_df.columns]
        if missing_features:
            logger.info(
                "Missing %d engineered features during fit. Generating features...",
                len(missing_features),
            )
            train_df = build_features(train_df)

        cities = train_df["city"].unique()
        logger.info(
            "Fitting DemandForecaster ensemble across %d cities: %s", len(cities), list(cities)
        )

        for city in [str(c) for c in cities]:
            city_df = train_df[train_df["city"] == city].sort_values("date").reset_index(drop=True)
            city_y = city_df["orders"].to_numpy()
            city_X = city_df[self._feature_cols].to_numpy()

            # 1. Fit LightGBM Regressor
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                random_state=self.random_seed,
                verbose=-1,
                n_jobs=1,
            )
            lgb_model.fit(city_X, city_y)
            self._lgb_models[city] = lgb_model

            # 1b. Fit Quantile LightGBM Regressors for prediction intervals
            self._lgb_quantile_models[city] = {}
            for alpha in self.quantile_alphas:
                q_model = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=alpha,
                    n_estimators=100,
                    learning_rate=0.05,
                    random_state=self.random_seed,
                    verbose=-1,
                    n_jobs=1,
                )
                q_model.fit(city_X, city_y)
                self._lgb_quantile_models[city][alpha] = q_model

            # 2. Fit Prophet Model (fully offline, cmdstanpy backend)
            prophet_df = pd.DataFrame({"ds": pd.to_datetime(city_df["date"]), "y": city_y})
            # Include custom offline regressors if available
            regressors = ["is_holiday", "promo_active", "is_rainy"]
            prophet_model = Prophet(
                weekly_seasonality=True,
                yearly_seasonality=True,
                daily_seasonality=False,
            )
            for reg in regressors:
                if reg in city_df.columns:
                    prophet_model.add_regressor(reg)
                    prophet_df[reg] = city_df[reg].to_numpy()

            # Suppress chatty fitting output
            prophet_model.fit(prophet_df)
            self._prophet_models[city] = prophet_model

        self.is_fitted = True
        logger.info("Successfully completed ensemble training across all cities.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Predict daily order volumes for input dataset using weighted ensemble average.

        Args:
            X (pd.DataFrame): Input dataframe containing date, city, and required feature columns.

        Returns:
            np.ndarray: One-dimensional array of predicted non-negative order volumes.

        Raises:
            RuntimeError: If forecaster has not been fitted prior to invoking predict.
        """
        if not self.is_fitted:
            raise RuntimeError("DemandForecaster must be fitted or loaded before calling predict.")

        pred_df = X.copy()
        missing_features = [c for c in self._feature_cols if c not in pred_df.columns]
        if missing_features:
            pred_df = build_features(pred_df)

        pred_df["orig_index"] = np.arange(len(pred_df))
        predictions = np.zeros(len(pred_df), dtype=float)

        for city in pred_df["city"].unique():
            city_mask = pred_df["city"] == city
            city_sub = pred_df[city_mask].copy()
            indices = city_sub["orig_index"].to_numpy()

            if city not in self._lgb_models or city not in self._prophet_models:
                logger.warning(
                    "City '%s' was not seen during training. Falling back to overall default model.",
                    city,
                )
                avail_cities = list(self._lgb_models.keys())
                lgb_mod = self._lgb_models[avail_cities[0]]
                prophet_mod = self._prophet_models[avail_cities[0]]
            else:
                lgb_mod = self._lgb_models[str(city)]
                prophet_mod = self._prophet_models[str(city)]

            # LightGBM inference
            city_X = city_sub[self._feature_cols].to_numpy()
            lgb_preds = lgb_mod.predict(city_X)

            # Prophet inference
            prophet_sub = pd.DataFrame({"ds": pd.to_datetime(city_sub["date"])})
            for reg in ["is_holiday", "promo_active", "is_rainy"]:
                if reg in city_sub.columns:
                    prophet_sub[reg] = city_sub[reg].to_numpy()

            prophet_out = prophet_mod.predict(prophet_sub)
            prophet_preds = prophet_out["yhat"].to_numpy()

            # Weighted combination
            combined = self.lgb_weight * lgb_preds + self.prophet_weight * prophet_preds
            predictions[indices] = combined

        # Ensure finite and strictly non-negative outputs
        safe_preds = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
        final_preds = np.maximum(0.0, safe_preds)
        return final_preds

    def predict_intervals(self, X: pd.DataFrame) -> dict[str, np.ndarray[Any, Any]]:
        """Predict expected demand point forecasts along with lower and upper confidence intervals.

        Args:
            X (pd.DataFrame): Input dataframe containing date, city, and required feature columns.

        Returns:
            dict[str, np.ndarray]: Dictionary with keys 'mean', 'p10', 'p50', and 'p90' containing prediction arrays.

        Raises:
            RuntimeError: If forecaster has not been fitted prior to invoking predict_intervals.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "DemandForecaster must be fitted or loaded before calling predict_intervals."
            )

        pred_df = X.copy()
        missing_features = [c for c in self._feature_cols if c not in pred_df.columns]
        if missing_features:
            pred_df = build_features(pred_df)

        pred_df["orig_index"] = np.arange(len(pred_df))
        n_rows = len(pred_df)
        res_mean = np.zeros(n_rows, dtype=float)
        res_p10 = np.zeros(n_rows, dtype=float)
        res_p50 = np.zeros(n_rows, dtype=float)
        res_p90 = np.zeros(n_rows, dtype=float)

        for city in pred_df["city"].unique():
            city_mask = pred_df["city"] == city
            city_sub = pred_df[city_mask].copy()
            indices = city_sub["orig_index"].to_numpy()

            if city not in self._lgb_models or city not in self._prophet_models:
                avail_cities = list(self._lgb_models.keys())
                lgb_mod = self._lgb_models[avail_cities[0]]
                q_mods = getattr(self, "_lgb_quantile_models", {}).get(avail_cities[0], {})
                prophet_mod = self._prophet_models[avail_cities[0]]
            else:
                lgb_mod = self._lgb_models[str(city)]
                q_mods = getattr(self, "_lgb_quantile_models", {}).get(str(city), {})
                prophet_mod = self._prophet_models[str(city)]

            city_X = city_sub[self._feature_cols].to_numpy()
            lgb_preds = lgb_mod.predict(city_X)

            lgb_p10 = q_mods[0.1].predict(city_X) if 0.1 in q_mods else lgb_preds
            lgb_p50 = q_mods[0.5].predict(city_X) if 0.5 in q_mods else lgb_preds
            lgb_p90 = q_mods[0.9].predict(city_X) if 0.9 in q_mods else lgb_preds

            # Prophet inference
            prophet_sub = pd.DataFrame({"ds": pd.to_datetime(city_sub["date"])})
            for reg in ["is_holiday", "promo_active", "is_rainy"]:
                if reg in city_sub.columns:
                    prophet_sub[reg] = city_sub[reg].to_numpy()

            prophet_out = prophet_mod.predict(prophet_sub)
            p_mean = prophet_out["yhat"].to_numpy()
            p_lower = prophet_out.get("yhat_lower", prophet_out["yhat"]).to_numpy()
            p_upper = prophet_out.get("yhat_upper", prophet_out["yhat"]).to_numpy()

            # Blend LightGBM quantiles with Prophet confidence bounds
            comb_mean = self.lgb_weight * lgb_preds + self.prophet_weight * p_mean
            comb_p10 = self.lgb_weight * lgb_p10 + self.prophet_weight * p_lower
            comb_p50 = self.lgb_weight * lgb_p50 + self.prophet_weight * p_mean
            comb_p90 = self.lgb_weight * lgb_p90 + self.prophet_weight * p_upper

            res_mean[indices] = comb_mean
            res_p10[indices] = np.minimum(comb_p10, comb_mean)
            res_p50[indices] = comb_p50
            res_p90[indices] = np.maximum(comb_p90, comb_mean)

        def clean_arr(arr: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            safe = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return np.maximum(0.0, safe)

        return {
            "mean": clean_arr(res_mean),
            "p10": clean_arr(res_p10),
            "p50": clean_arr(res_p50),
            "p90": clean_arr(res_p90),
        }

    def forecast_future(  # noqa: C901
        self, horizon_days: int, last_known_df: pd.DataFrame, weather_df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Generate future demand forecast iteratively rolling forward day by day.

        Args:
            horizon_days (int): Number of days to forecast into the future.
            last_known_df (pd.DataFrame): Historical data including at least 30 days of recent orders.
            weather_df (pd.DataFrame | None): Optional live weather data with 'date', 'city',
                'temperature_c', 'rainfall_mm', 'is_rainy' columns.

        Returns:
            pd.DataFrame: Forecasted dataframe containing future dates and predictions.

        Raises:
            RuntimeError: If forecaster has not been fitted prior to invoking forecast_future.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "DemandForecaster must be fitted or loaded before calling forecast_future."
            )

        future_dfs = []
        for city in last_known_df["city"].unique():
            city_hist = last_known_df[last_known_df["city"] == city].sort_values("date").copy()
            if len(city_hist) < 30:
                logger.warning(
                    f"City {city} has <30 days history. Rolling features may be unstable."
                )

            carry_cols = [
                col
                for col in [
                    "temperature_c",
                    "rainfall_mm",
                    "is_rainy",
                    "promo_active",
                    "is_holiday",
                ]
                if col in city_hist.columns
            ]

            # Pre-filter weather data for this city if available
            city_weather = None
            if weather_df is not None and not weather_df.empty:
                city_weather = weather_df[weather_df["city"] == city].copy()
                if not city_weather.empty:
                    city_weather["date"] = pd.to_datetime(city_weather["date"]).dt.date

            last_date = city_hist["date"].max()

            for step in range(1, horizon_days + 1):
                next_date = last_date + pd.Timedelta(days=step)
                new_row = city_hist.iloc[[-1]].copy()
                new_row["date"] = next_date
                new_row["orders"] = np.nan
                new_row["is_anomaly"] = False
                new_row["anomaly_score"] = 0.0

                for col in carry_cols:
                    if col in ["promo_active", "is_holiday"]:
                        new_row[col] = 0
                    elif col in ["is_rainy"]:
                        new_row[col] = False
                    else:
                        new_row[col] = city_hist[col].mean()

                # Override with live weather if available for this specific date
                if city_weather is not None:
                    match = city_weather[city_weather["date"] == next_date.date()]
                    if not match.empty:
                        for w_col in ["temperature_c", "rainfall_mm", "is_rainy"]:
                            if w_col in match.columns:
                                new_row[w_col] = match.iloc[0][w_col]

                new_row["is_future"] = True
                city_hist = pd.concat([city_hist, new_row], ignore_index=True)

                hist_tail = city_hist.tail(40).copy()
                feats = build_features(hist_tail)
                last_feat_row = feats.iloc[[-1]]

                intervals = self.predict_intervals(last_feat_row)

                idx_last = city_hist.index[-1]
                city_hist.loc[idx_last, "orders"] = intervals["mean"][0]
                city_hist.loc[idx_last, "pred_orders"] = intervals["mean"][0]
                city_hist.loc[idx_last, "pred_p10"] = intervals["p10"][0]
                city_hist.loc[idx_last, "pred_p50"] = intervals["p50"][0]
                city_hist.loc[idx_last, "pred_p90"] = intervals["p90"][0]

            future_dfs.append(city_hist[city_hist.get("is_future") == True])  # noqa: E712

        if not future_dfs:
            return pd.DataFrame()

        final_df = pd.concat(future_dfs, ignore_index=True)
        return final_df

    def save(self, path: Path | str | None = None) -> None:
        """Serialize and save trained ensemble models to filesystem.

        Args:
            path (Path | str | None): Destination model path. Defaults to settings path.
        """
        save_p = Path(path) if path is not None else settings.forecaster_model_path
        save_p.parent.mkdir(parents=True, exist_ok=True)
        with open(save_p, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved trained DemandForecaster artifact to %s", save_p)

    @classmethod
    def load(cls, path: Path | str | None = None) -> DemandForecaster:
        """Load serialized DemandForecaster ensemble model from disk.

        Args:
            path (Path | str | None): Path to model file. Defaults to settings path.

        Returns:
            DemandForecaster: Loaded and executable forecaster instance.

        Raises:
            FileNotFoundError: If target model file path does not exist.
        """
        load_p = Path(path) if path is not None else settings.forecaster_model_path
        if not load_p.exists():
            raise FileNotFoundError(f"No trained forecaster model found at: {load_p}")
        with open(load_p, "rb") as f:
            model = pickle.load(f)
        logger.info("Successfully loaded DemandForecaster from %s", load_p)
        return model
