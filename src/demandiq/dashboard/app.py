# ruff: noqa: C408, C901, B007
"""Interactive Streamlit dashboard app for demand forecasting and anomaly diagnostics."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose

from demandiq.anomaly.detector import HybridAnomalyDetector
from demandiq.config import settings
from demandiq.data.loader import load_and_validate_orders
from demandiq.data.monitor import check_data_health
from demandiq.features.engineer import build_features
from demandiq.models.cross_validate import compute_metrics, compute_rolling_accuracy
from demandiq.models.explain import get_top_drivers, get_weather_shap_contributions, get_global_shap_summary
from demandiq.models.forecaster import DemandForecaster
from demandiq.models.model_card import generate_model_report
from demandiq.reports.anomaly_digest import generate_markdown_digest

import plotly.express as px
from demandiq.data.promo_calendar import PromoCalendar
from demandiq.monitoring.drift_detector import detect_performance_drift
from demandiq.monitoring.scheduler import schedule_retrain
from demandiq.notifications.email_alert import send_email_alert
from demandiq.notifications.slack_alert import send_slack_alert
from demandiq.recommendations.capacity_planner import CapacityPlanner
from demandiq.registry.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# Configure webpage styling and metadata
st.set_page_config(
    page_title="DemandIQ | Production ML Forecasting & Anomaly Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject modern glassmorphic rich dark theme CSS aesthetics
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    color: #E2E8F0;
    background-color: #0B0F19;
}

/* Glassmorphism card container styling */
.glass-card {
    background: rgba(19, 26, 43, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.glass-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-2px);
}

/* Gradient Header Typography */
.gradient-title {
    background: linear-gradient(135deg, #6366F1 0%, #A855F7 50%, #EC4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.sub-title {
    color: #94A3B8;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Customized Metric Box */
.metric-box {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-val {
    font-size: 2.2rem;
    font-weight: 700;
    color: #38BDF8;
    margin-top: 8px;
}
.metric-label {
    font-size: 0.9rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 1px;
}
</style>
"""


def apply_custom_theme() -> None:
    """Inject custom responsive vanilla CSS into Streamlit document DOM."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading pre-trained DemandIQ models and datasets...")
def load_production_artifacts() -> tuple[DemandForecaster, HybridAnomalyDetector, pd.DataFrame]:
    """Load pre-trained models and engineered order dataset from filesystem.

    Returns:
        tuple[DemandForecaster, HybridAnomalyDetector, pd.DataFrame]: Tuple containing
            forecaster model, anomaly detector model, and engineered dataset.

    Raises:
        FileNotFoundError: If any essential artifact is missing from disk.
    """
    if not settings.forecaster_model_path.exists() or not settings.anomaly_detector_path.exists():
        raise FileNotFoundError("Model artifacts absent on local storage.")

    forecaster = DemandForecaster.load(settings.forecaster_model_path)
    detector = HybridAnomalyDetector.load(settings.anomaly_detector_path)

    if settings.processed_features_path.exists():
        if settings.processed_features_path.suffix == ".parquet":
            try:
                df = pd.read_parquet(settings.processed_features_path)
            except Exception:
                df = pd.read_csv(settings.processed_features_path.with_suffix(".csv"))
        else:
            df = pd.read_csv(settings.processed_features_path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
    else:
        raw_df = load_and_validate_orders(settings.raw_orders_path)
        df = build_features(raw_df)

    return forecaster, detector, df


@st.cache_data(show_spinner="Computing cross-city benchmarks and error matrices...")
def get_benchmark_data(
    df: pd.DataFrame, _forecaster: DemandForecaster, _detector: HybridAnomalyDetector
) -> pd.DataFrame:
    """Compute benchmark predictions and anomalies across the full historical dataset."""
    bench_df = df.copy()
    try:
        bench_df["pred_orders"] = _forecaster.predict(bench_df)
    except Exception:
        bench_df["pred_orders"] = bench_df.get("orders_lag_7", bench_df["orders"])
    bench_df["residuals"] = bench_df["orders"] - bench_df["pred_orders"]
    bench_df["is_anomaly"] = _detector.predict(bench_df["residuals"].to_numpy())
    bench_df["year_month"] = bench_df["date"].dt.to_period("M").astype(str)
    return bench_df


def main() -> None:  # pragma: no cover
    """Render interactive Streamlit application components and visual plots."""
    apply_custom_theme()

    st.markdown(
        '<div class="gradient-title">⚡ DemandIQ Analytics Platform</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-title">Real-Time LightGBM + Prophet Ensemble Forecasting & Hybrid Anomaly Diagnostics</div>',
        unsafe_allow_html=True,
    )

    # Graceful error check for missing model training artifacts
    try:
        forecaster, detector, df = load_production_artifacts()
    except Exception as exc:
        logger.warning("Failed to load artifacts in dashboard: %s", exc)
        st.error(
            "⚠️ **Pre-trained DemandIQ models and datasets not found!**\n\n"
            "Please run `make train` (or `python -m demandiq.pipeline`) in your terminal first "
            "to generate the synthetic dataset and train the machine learning ensemble artifacts."
        )
        st.stop()

    # Data Quality & Freshness Guardrail
    health = check_data_health(df, max_stale_days=7)
    if not health["is_healthy"]:
        st.error("🚨 **CRITICAL DATA QUALITY ALERT** 🚨")
        for err in health.get("errors", []):
            st.error(f"- {err}")
        st.warning(
            "Dashboard blocked due to stale or corrupted data. Please run the data pipeline to refresh records."
        )
        st.stop()

    for warn in health.get("warnings", []):
        st.sidebar.warning(f"⚠️ **Data Stale Warning:** {warn}")

    # --- Sidebar Filtering Controls ---
    st.sidebar.title("🎛️ Dashboard Controls")
    st.sidebar.markdown("Filter demand metrics by metropolitan region and timeline intervals.")

    cities = sorted(df["city"].unique().tolist())
    selected_city = str(st.sidebar.selectbox("Select City Location", options=cities, index=0))

    city_df = df[df["city"] == selected_city].sort_values("date").reset_index(drop=True)
    min_date = city_df["date"].min().date()
    max_date = city_df["date"].max().date()

    date_range = st.sidebar.date_input(
        "Select Historical Analysis Window",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        sub_df = city_df[(city_df["date"] >= start_dt) & (city_df["date"] <= end_dt)].copy()
    else:
        sub_df = city_df.copy()

    if sub_df.empty:
        st.warning("No historical demand order observations match the chosen timeframe filter.")
        st.stop()

    # Ensemble Weight Tuning
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Ensemble Weight Tuning")
    new_lgb_wt = st.sidebar.slider(
        "LightGBM Weight (%)",
        min_value=0,
        max_value=100,
        value=int(getattr(forecaster, "lgb_weight", 0.6) * 100),
        step=5,
        help="Higher weight favors the non-linear LightGBM model. Prophet weight is automatically calculated as (100% - LGB Weight).",
    )
    if hasattr(forecaster, "lgb_weight"):
        forecaster.lgb_weight = new_lgb_wt / 100.0
        forecaster.prophet_weight = 1.0 - forecaster.lgb_weight

    # Model retraining execution button in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Model Management")
    if st.sidebar.button("🔄 Retrain Ensemble Models", use_container_width=True):
        with st.sidebar.status("Rebuilding models & features...", expanded=True) as status:
            st.write("Generating synthetic data and engineering leak-free features...")
            from demandiq.pipeline import run_pipeline

            run_pipeline()
            st.cache_resource.clear()
            st.cache_data.clear()
            status.update(label="Retrain Complete!", state="complete", expanded=False)
        st.sidebar.success("Models retrained and reloaded into memory!")
        st.rerun()

    # Feature 10: Anomaly Sensitivity Tuner
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Anomaly Sensitivity")
    new_z_thresh = st.sidebar.slider("Z-Score Threshold", min_value=1.5, max_value=4.0, value=float(detector.z_threshold), step=0.1)
    new_strict_mode = st.sidebar.toggle("Strict Mode", value=detector.strict_mode, help="Require both Isolation Forest and Z-Score to agree.")
    detector.z_threshold = new_z_thresh
    detector.strict_mode = new_strict_mode

    # Feature 6: Promo Calendar
    st.sidebar.markdown("---")
    with st.sidebar.expander("📅 Promo Calendar", expanded=False):
        st.write("Inject upcoming promos into the future forecast.")
        pc = PromoCalendar.load()
        p_city = st.selectbox("City", options=cities, key="promo_city")
        p_start = st.date_input("Start Date", key="promo_start")
        p_end = st.date_input("End Date", key="promo_end")
        p_intensity = st.slider("Intensity", 1.0, 3.0, 1.2, 0.1)
        p_label = st.text_input("Label (e.g., Summer Sale)")
        if st.button("Add Promo"):
            if p_end >= p_start:
                pc.add_promo(p_city, p_start, p_end, p_intensity, p_label)
                pc.save()
                st.success("Promo added!")
            else:
                st.error("End date must be >= start date.")

    # Feature 1: Alert Settings
    st.sidebar.markdown("---")
    with st.sidebar.expander("🔔 Alert Settings", expanded=False):
        settings.alert_enabled = st.toggle("Enable Push Alerts", value=settings.alert_enabled)
        st.text_input("Slack Webhook URL", value=settings.slack_webhook_url if settings.slack_webhook_url else "", type="password", key="slack_url_input")
        settings.slack_webhook_url = st.session_state.slack_url_input

    # Feature 3: Auto-Retrain Settings
    st.sidebar.markdown("---")
    with st.sidebar.expander("⚙️ Auto-Retrain Settings", expanded=False):
        drift_thresh = st.slider("Drift Threshold (MAPE %)", 5.0, 15.0, 8.0, 0.5)
        cron_expr = st.text_input("Cron Schedule", value="0 2 * * 0")
        if st.button("Schedule Retraining"):
            schedule_retrain(cron_expr)
            st.success("Scheduled!")

    # Run predictions and detect residual anomalies for filtered view
    with st.spinner("Executing real-time ensemble inference..."):
        try:
            intervals = forecaster.predict_intervals(sub_df)
            sub_df["pred_orders"] = intervals["mean"]
            sub_df["pred_p10"] = intervals["p10"]
            sub_df["pred_p90"] = intervals["p90"]
        except AttributeError:
            sub_df["pred_orders"] = forecaster.predict(sub_df)
            sub_df["pred_p10"] = sub_df["pred_orders"] * 0.9
            sub_df["pred_p90"] = sub_df["pred_orders"] * 1.1

        sub_df["residuals"] = sub_df["orders"] - sub_df["pred_orders"]
        sub_df["is_anomaly"] = detector.predict(sub_df["residuals"].to_numpy())
        sub_df["anomaly_score"] = detector.score(sub_df["residuals"].to_numpy())
        try:
            sub_df["anomaly_type"] = detector.classify(sub_df["residuals"].to_numpy())
        except AttributeError:
            sub_df["anomaly_type"] = np.where(sub_df["is_anomaly"], "anomaly", "normal")

    # --- Main Application Tabs ---
    tab_deep_dive, tab_benchmarks, tab_future, tab_capacity, tab_model_card, tab_registry = st.tabs(
        ["📈 City Deep Dive", "🗺️ City Benchmarks", "📅 Future Forecast", "📦 Capacity Planning", "🏅 Model Report Card", "🔬 Model Registry"]
    )

    with tab_deep_dive:
        # --- Section 1: KPI & Metrics Panel ---
        st.markdown("### 📊 Performance Indicators & Error Metrics")
        scores = compute_metrics(sub_df["orders"].to_numpy(), sub_df["pred_orders"].to_numpy())
        anomaly_count = int(sub_df["is_anomaly"].sum())
        total_volume = int(sub_df["orders"].sum())

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="metric-box"><div class="metric-label">Ensemble MAPE</div>'
                f'<div class="metric-val">{scores["mape"]:.2f}%</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-box"><div class="metric-label">Ensemble RMSE</div>'
                f'<div class="metric-val">{scores["rmse"]:,.1f}</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="metric-box"><div class="metric-label">Detected Anomalies</div>'
                f'<div class="metric-val" style="color: #F43F5E;">{anomaly_count}</div></div>',
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f'<div class="metric-box"><div class="metric-label">Total Order Volume</div>'
                f'<div class="metric-val" style="color: #10B981;">{total_volume:,}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)

        # --- Section 2: Actual vs. Predicted Demand & Anomaly Line Chart ---
        st.markdown("### 📈 Demand Trajectory & Anomaly Highlights")
        fig = go.Figure()

        # Confidence interval band (p10 to p90)
        if "pred_p10" in sub_df.columns and "pred_p90" in sub_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=list(sub_df["date"]) + list(sub_df["date"])[::-1],
                    y=list(sub_df["pred_p90"]) + list(sub_df["pred_p10"])[::-1],
                    fill="toself",
                    fillcolor="rgba(168, 85, 247, 0.18)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    name="80% Prediction Interval (P10-P90)",
                )
            )

        # Actual orders trace
        fig.add_trace(
            go.Scatter(
                x=sub_df["date"],
                y=sub_df["orders"],
                mode="lines",
                name="Actual Orders",
                line=dict(color="#38BDF8", width=2.5),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Actual Orders: %{y:,}<extra></extra>",
            )
        )

        # Predicted orders trace
        fig.add_trace(
            go.Scatter(
                x=sub_df["date"],
                y=sub_df["pred_orders"],
                mode="lines",
                name="Ensemble Forecast (0.6 LGBM / 0.4 Prophet)",
                line=dict(color="#A855F7", width=2, dash="dash"),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Predicted Orders: %{y:,.1f}<extra></extra>",
            )
        )

        # Anomaly scatter marker highlight
        anom_rows = sub_df[sub_df["is_anomaly"]]
        if not anom_rows.empty:
            fig.add_trace(
                go.Scatter(
                    x=anom_rows["date"],
                    y=anom_rows["orders"],
                    mode="markers",
                    name="Flagged Anomaly (Isolation Forest + Z-Score)",
                    marker=dict(
                        color="#F43F5E",
                        size=12,
                        symbol="diamond-open",
                        line=dict(width=2, color="#F43F5E"),
                    ),
                    hovertemplate="<b>ANOMALY TRIGGERED</b><br>Date: %{x|%Y-%m-%d}<br>Orders: %{y:,}<extra></extra>",
                )
            )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.5)",
            font=dict(color="#E2E8F0"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Daily Orders", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        # CSV Export Button
        export_cols = [
            "date",
            "city",
            "orders",
            "pred_orders",
            "pred_p10",
            "pred_p90",
            "residuals",
            "is_anomaly",
            "anomaly_score",
            "anomaly_type",
        ]
        avail_cols = [c for c in export_cols if c in sub_df.columns]
        csv_data = sub_df[avail_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Forecast & Anomaly Results (CSV)",
            data=csv_data,
            file_name=f"demandiq_{selected_city.lower().replace(' ', '_')}_results.csv",
            mime="text/csv",
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # --- Section 2.5: Rolling Forecast Accuracy Tracker ---
        st.markdown("### 🎯 Rolling Forecast Accuracy (MAPE)")
        st.markdown("Monitor model drift and performance degradation over a sliding time window.")
        roll_window = st.radio(
            "Select Rolling Window", options=[7, 14, 30, 60], index=2, horizontal=True
        )
        rolling_df = compute_rolling_accuracy(
            sub_df, window_days=int(roll_window) if roll_window is not None else 30
        )
        if not rolling_df.empty:
            roll_fig = go.Figure(
                go.Scatter(
                    x=rolling_df["date"],
                    y=rolling_df["rolling_mape"],
                    mode="lines",
                    name=f"{roll_window}-Day Rolling MAPE",
                    line=dict(color="#EC4899", width=2.5),
                    hovertemplate="Date: %{x|%Y-%m-%d}<br>Rolling MAPE: %{y:.2f}%<extra></extra>",
                )
            )
            roll_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.5)",
                font=dict(color="#E2E8F0"),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(
                    title="Rolling MAPE (%)", showgrid=True, gridcolor="rgba(255,255,255,0.05)"
                ),
                margin=dict(l=20, r=20, t=30, b=20),
                hovermode="x unified",
                height=300,
            )
            st.plotly_chart(roll_fig, use_container_width=True)
        else:
            st.warning("Insufficient data to compute rolling accuracy.")
        st.markdown("<br>", unsafe_allow_html=True)

        # --- Section 3: Anomaly Detail Drill-Down Panel ---
        st.markdown("### 🚨 Anomaly Diagnostics & Root-Cause Inspector")
        st.markdown(
            "Examine detected surges and dips with operational severity scores and SHAP attribution drivers."
        )
        anom_df = sub_df[sub_df["is_anomaly"]].sort_values("date", ascending=False)
        if anom_df.empty:
            st.success(
                "✅ No demand anomalies or operational surges detected within the selected timeframe."
            )
        else:
            for idx, row in anom_df.iterrows():
                a_type = str(row.get("anomaly_type", "anomaly")).upper()
                icon = "📈" if "SURGE" in a_type else "📉"
                dt_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
                exp_label = (
                    f"{icon} {a_type} — {dt_str} | Actual: {row['orders']:,} "
                    f"(vs Forecast: {row['pred_orders']:,.0f}) | Severity Score: {row.get('anomaly_score', 0.0):.1f}"
                )
                with st.expander(exp_label):
                    ca, cb, cc = st.columns(3)
                    ca.metric("Observed Demand", f"{row['orders']:,}")
                    cb.metric(
                        "Ensemble Forecast",
                        f"{row['pred_orders']:,.0f}",
                        delta=f"{row['residuals']:+,.0f}",
                    )
                    cc.metric("Anomaly Severity", f"{row.get('anomaly_score', 0.0):.2f}")

                    st.markdown("#### 🔍 Primary SHAP Attribution Drivers")
                    row_drivers = get_top_drivers(forecaster, row, n=3, city=selected_city)
                    if isinstance(row_drivers, list) and all(
                        isinstance(i, tuple) for i in row_drivers
                    ):
                        for d_name, d_val in row_drivers:
                            sign = "+" if d_val > 0 else ""
                            st.markdown(
                                f"- **{d_name}**: `{sign}{d_val:.1f}` orders impact vs baseline"
                            )

            st.markdown("#### 📄 Export Alert Digest")
            digest_md = generate_markdown_digest(anom_df, selected_city)
            st.download_button(
                label="⬇️ Download Anomaly Alert Digest (Markdown)",
                data=digest_md.encode("utf-8"),
                file_name=f"demandiq_{selected_city.lower().replace(' ', '_')}_alerts.md",
                mime="text/markdown",
            )
        st.markdown("<br>", unsafe_allow_html=True)

        # --- Section 4: SHAP Feature Explanations ---
        st.markdown("### 🧠 SHAP Tree Attribution Drivers")
        st.markdown(
            "Inspect explicit feature importance drivers explaining the model prediction for a specific calendar observation."
        )

        date_options = sub_df["date"].dt.strftime("%Y-%m-%d").tolist()
        selected_date_str = str(
            st.selectbox(
                "Choose Specific Observation Date",
                options=date_options,
                index=len(date_options) - 1,
            )
        )

        target_row = sub_df[sub_df["date"] == pd.to_datetime(selected_date_str)].iloc[0]

        with st.spinner("Extracting SHAP attribution tree contributions..."):
            drivers = get_top_drivers(forecaster, target_row, n=8, city=selected_city)

        if isinstance(drivers, list) and all(isinstance(i, tuple) for i in drivers):
            d_names = [pair[0] for pair in drivers]
            d_vals = [pair[1] for pair in drivers]

            colors = ["#10B981" if v > 0 else "#F43F5E" for v in d_vals]

            shap_fig = go.Figure(
                go.Bar(
                    x=d_vals,
                    y=d_names,
                    orientation="h",
                    marker=dict(color=colors),
                    hovertemplate="Feature: %{y}<br>SHAP Contribution: %{x:+.2f} orders<extra></extra>",
                )
            )
            shap_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.5)",
                font=dict(color="#E2E8F0"),
                xaxis=dict(
                    title="SHAP Order Attribution Magnitude (vs. City Baseline)",
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.05)",
                ),
                yaxis=dict(autorange="reversed"),
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(shap_fig, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # --- Section 4.5: Weather Impact Attribution Panel ---
        st.markdown("### ☁️ Weather Impact Attribution")
        st.markdown(
            "Quantify the demand elasticity of weather (temperature and rainfall) using aggregated SHAP attributions."
        )

        with st.spinner("Calculating weather SHAP elasticity..."):
            weather_shap = get_weather_shap_contributions(forecaster, sub_df, selected_city)

            weather_fig = go.Figure()
            weather_fig.add_trace(
                go.Scatter(
                    x=sub_df["date"],
                    y=sub_df["orders"],
                    mode="lines",
                    name="Actual Orders",
                    line=dict(color="#38BDF8", width=2),
                )
            )
            weather_fig.add_trace(
                go.Bar(
                    x=sub_df["date"],
                    y=weather_shap,
                    name="Weather Order Impact",
                    marker=dict(color=["#10B981" if v > 0 else "#F43F5E" for v in weather_shap]),
                    yaxis="y2",
                    opacity=0.6,
                )
            )
            weather_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.5)",
                font=dict(color="#E2E8F0"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="Daily Orders", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                yaxis2=dict(
                    title="Weather Impact (Orders)",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
                margin=dict(l=20, r=20, t=50, b=20),
                hovermode="x unified",
                height=350,
            )
            st.plotly_chart(weather_fig, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- Section 4.8: Global SHAP Summary ---
        st.markdown("### 🌐 Global Feature Importance")
        st.markdown("Macro-level view of which features drive the model's predictions overall for this timeframe.")
        with st.spinner("Computing global feature importance..."):
            try:
                global_shap = get_global_shap_summary(forecaster, sub_df, city=selected_city, n_features=10)
                global_fig = px.bar(
                    global_shap, 
                    x="mean_abs_shap", 
                    y="feature", 
                    orientation="h",
                    title="Average Absolute SHAP Impact by Feature",
                    labels={"mean_abs_shap": "Mean Absolute SHAP Value", "feature": "Feature"}
                )
                global_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15, 23, 42, 0.5)",
                    font=dict(color="#E2E8F0"),
                    yaxis={'categoryorder':'total ascending'},
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(global_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not compute global SHAP summary: {e}")
        st.markdown("<br>", unsafe_allow_html=True)

        # --- Section 5: Structural Trend & Seasonality Decomposition (STL) ---
        st.markdown("### 🌊 Structural Trend & Seasonality Decomposition (STL)")
        st.markdown(
            "Deconstruct daily order time-series into macro underlying growth trend, weekly seasonal oscillation, and unexplained residual noise."
        )
        if len(sub_df) < 14:
            st.info(
                "⚠️ Please select an analysis window of at least 14 days to compute seasonal decomposition."
            )
        else:
            with st.spinner("Decomposing structural time-series via STL..."):
                try:
                    decomp = seasonal_decompose(
                        sub_df["orders"].to_numpy(), model="additive", period=7
                    )

                    dec_fig = go.Figure()
                    dec_fig.add_trace(
                        go.Scatter(
                            x=sub_df["date"],
                            y=decomp.trend,
                            name="Macro Trend",
                            line=dict(color="#10B981", width=2.5),
                        )
                    )
                    dec_fig.add_trace(
                        go.Scatter(
                            x=sub_df["date"],
                            y=decomp.seasonal,
                            name="Weekly Seasonality",
                            line=dict(color="#38BDF8", width=2),
                        )
                    )
                    dec_fig.add_trace(
                        go.Scatter(
                            x=sub_df["date"],
                            y=decomp.resid,
                            name="Residual Noise",
                            line=dict(color="#F43F5E", width=1.5, dash="dot"),
                        )
                    )

                    dec_fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15, 23, 42, 0.5)",
                        font=dict(color="#E2E8F0"),
                        legend=dict(
                            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                        ),
                        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                        yaxis=dict(
                            title="Order Component Volume",
                            showgrid=True,
                            gridcolor="rgba(255,255,255,0.05)",
                        ),
                        hovermode="x unified",
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    st.plotly_chart(dec_fig, use_container_width=True)
                except Exception as exc:
                    st.warning(f"Could not calculate decomposition on this time slice: {exc}")

    with tab_benchmarks:
        st.markdown("### 🗺️ Metropolitan Benchmark Comparisons")
        st.markdown(
            "Compare forecast accuracy, total demand volume, and anomaly surge frequency across all regional logistics markets."
        )

        bench_df = get_benchmark_data(df, forecaster, detector)

        # 1. Summary Table by City
        summary_rows: list[dict[str, Any]] = []
        for c in sorted(bench_df["city"].unique()):
            cdf = bench_df[bench_df["city"] == c]
            c_scores = compute_metrics(cdf["orders"].to_numpy(), cdf["pred_orders"].to_numpy())
            anom_rate = (cdf["is_anomaly"].sum() / len(cdf)) * 100.0
            summary_rows.append(
                {
                    "City Market": c,
                    "Avg Daily Orders": int(cdf["orders"].mean()),
                    "Total Volume": int(cdf["orders"].sum()),
                    "Ensemble MAPE (%)": round(c_scores["mape"], 2),
                    "Ensemble RMSE": round(c_scores["rmse"], 1),
                    "Anomaly Frequency (%)": round(anom_rate, 2),
                }
            )
        sum_df = pd.DataFrame(summary_rows)
        st.dataframe(sum_df, use_container_width=True, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("#### 🗺️ Geospatial Demand & Anomaly Heat Map")
        city_coords = {
            "New York": {"lat": 40.7128, "lon": -74.0060},
            "Chicago": {"lat": 41.8781, "lon": -87.6298},
            "Los Angeles": {"lat": 34.0522, "lon": -118.2437},
            "Austin": {"lat": 30.2672, "lon": -97.7431},
            "Miami": {"lat": 25.7617, "lon": -80.1918}
        }
        geo_df = sum_df.copy()
        geo_df["lat"] = geo_df["City Market"].map(lambda c: city_coords.get(c, {}).get("lat", 0))
        geo_df["lon"] = geo_df["City Market"].map(lambda c: city_coords.get(c, {}).get("lon", 0))
        geo_df["marker_size"] = geo_df["Avg Daily Orders"] / geo_df["Avg Daily Orders"].max() * 40
        
        geo_fig = px.scatter_geo(
            geo_df, lat="lat", lon="lon", hover_name="City Market",
            size="marker_size", color="Ensemble MAPE (%)",
            hover_data={"lat": False, "lon": False, "Avg Daily Orders": True, "Anomaly Frequency (%)": True},
            color_continuous_scale="Viridis", projection="albers usa",
            title="Market Overview: Volume & Accuracy"
        )
        geo_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.5)",
            font=dict(color="#E2E8F0"),
            geo=dict(bgcolor="rgba(0,0,0,0)", lakecolor="rgba(15, 23, 42, 0.5)"),
            margin=dict(l=0, r=0, t=40, b=0),
            height=400
        )
        st.plotly_chart(geo_fig, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown("#### 📊 MAPE vs Anomaly Frequency by City")
            scat_fig = go.Figure(
                go.Scatter(
                    x=sum_df["Ensemble MAPE (%)"],
                    y=sum_df["Anomaly Frequency (%)"],
                    mode="markers+text",
                    text=sum_df["City Market"],
                    textposition="top center",
                    marker=dict(
                        size=16,
                        color=["#6366F1", "#A855F7", "#EC4899", "#38BDF8", "#10B981"],
                        line=dict(width=2, color="#FFFFFF"),
                    ),
                )
            )
            scat_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.5)",
                font=dict(color="#E2E8F0"),
                xaxis=dict(
                    title="Ensemble MAPE (%)", showgrid=True, gridcolor="rgba(255,255,255,0.05)"
                ),
                yaxis=dict(
                    title="Anomaly Frequency (%)",
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.05)",
                ),
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(scat_fig, use_container_width=True)

        with bc2:
            st.markdown("#### 📦 Average Daily Order Volume by Market")
            bar_fig = go.Figure(
                go.Bar(
                    x=sum_df["City Market"],
                    y=sum_df["Avg Daily Orders"],
                    marker=dict(color=["#6366F1", "#A855F7", "#EC4899", "#38BDF8", "#10B981"]),
                )
            )
            bar_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.5)",
                font=dict(color="#E2E8F0"),
                xaxis=dict(title="City Market", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(
                    title="Avg Daily Orders", showgrid=True, gridcolor="rgba(255,255,255,0.05)"
                ),
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        # Heatmap of MAPE by city x month
        st.markdown("#### 🗓️ Monthly MAPE Accuracy Matrix (%)")
        hm_data: list[list[float | None]] = []
        months = sorted(bench_df["year_month"].unique())[-12:]
        for c in sorted(bench_df["city"].unique()):
            hm_row: list[float | None] = []
            for m in months:
                sub = bench_df[(bench_df["city"] == c) & (bench_df["year_month"] == m)]
                if len(sub) > 0:
                    mape_val = compute_metrics(
                        sub["orders"].to_numpy(), sub["pred_orders"].to_numpy()
                    )["mape"]
                    hm_row.append(round(float(mape_val), 2))
                else:
                    hm_row.append(None)
            hm_data.append(hm_row)

        hm_fig = go.Figure(
            go.Heatmap(
                z=hm_data,
                x=months,
                y=sorted(bench_df["city"].unique()),
                colorscale="Viridis",
                reversescale=True,
                colorbar=dict(title="MAPE %"),
            )
        )
        hm_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.5)",
            font=dict(color="#E2E8F0"),
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(hm_fig, use_container_width=True)

    with tab_future:
        st.markdown("### 📅 Future Horizon Forecasting")
        st.markdown(
            "Generate forward-looking demand volume predictions with confidence bands for upcoming operational cycles."
        )

        horizon = st.slider("Forecast Horizon (Days)", min_value=7, max_value=90, value=30, step=1)

        with st.spinner(f"Rolling forward {horizon} days into the future..."):
            try:
                future_df = forecaster.forecast_future(horizon_days=horizon, last_known_df=sub_df)

                if future_df.empty:
                    st.warning("Could not generate future forecast.")
                else:
                    st.success(
                        f"Successfully generated a {horizon}-day ahead forecast for {selected_city}!"
                    )

                    fut_city_df = future_df[future_df["city"] == selected_city].sort_values("date")

                    fut_fig = go.Figure()

                    # Future Confidence interval band (p10 to p90)
                    fut_fig.add_trace(
                        go.Scatter(
                            x=list(fut_city_df["date"]) + list(fut_city_df["date"])[::-1],
                            y=list(fut_city_df["pred_p90"]) + list(fut_city_df["pred_p10"])[::-1],
                            fill="toself",
                            fillcolor="rgba(168, 85, 247, 0.18)",
                            line=dict(color="rgba(255,255,255,0)"),
                            hoverinfo="skip",
                            name="80% Future Prediction Interval (P10-P90)",
                        )
                    )

                    # Predicted orders trace
                    fut_fig.add_trace(
                        go.Scatter(
                            x=fut_city_df["date"],
                            y=fut_city_df["pred_orders"],
                            mode="lines+markers",
                            name="Future Ensemble Forecast",
                            line=dict(color="#A855F7", width=2, dash="dash"),
                            hovertemplate="Date: %{x|%Y-%m-%d}<br>Predicted Orders: %{y:,.1f}<extra></extra>",
                        )
                    )

                    fut_fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15, 23, 42, 0.5)",
                        font=dict(color="#E2E8F0"),
                        legend=dict(
                            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                        ),
                        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                        yaxis=dict(
                            title="Daily Orders", showgrid=True, gridcolor="rgba(255,255,255,0.05)"
                        ),
                        margin=dict(l=20, r=20, t=50, b=20),
                        hovermode="x unified",
                        height=450,
                    )
                    st.plotly_chart(fut_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to generate future forecast: {e}")

    with tab_model_card:
        st.markdown("### 🏅 Global Model Report Card")
        st.markdown(
            "Comprehensive evaluation of model health, calibration, and anomaly diagnostic performance."
        )

        with st.spinner("Compiling model report card..."):
            report = generate_model_report(df, forecaster, detector)

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("Global MAPE", f"{report['global'].get('mape', 0.0):.2f}%")
            with mc2:
                st.metric("Global RMSE", f"{report['global'].get('rmse', 0.0):.1f}")
            with mc3:
                age_str = (
                    f"{report['metadata'].get('age_days', 0.0):.1f} days"
                    if report["metadata"].get("age_days", -1) >= 0
                    else "Unknown"
                )
                st.metric("Model Age", age_str)

            st.markdown("#### Interval Calibration (P10-P90 Coverage)")
            cov = report["calibration"].get("p10_p90_coverage", 0.0)
            if cov is not None:
                st.progress(cov)
                st.caption(f"Actual coverage: **{cov*100:.1f}%** (Target: 80%)")

            st.markdown("#### Anomaly Flagging Rate")
            rate = report["anomaly"].get("flag_rate", 0.0)
            if rate is not None:
                st.progress(rate if rate <= 1.0 else 1.0)
                st.caption(f"Flag rate: **{rate*100:.2f}%** (Expected ~1.5%)")

    with tab_capacity:
        st.markdown("### 📦 Capacity & Inventory Recommendation Engine")
        st.markdown("Translate future demand forecasts into actionable inventory and safety stock recommendations.")
        
        horizon = st.slider("Planning Horizon (Days)", min_value=7, max_value=90, value=14, step=1, key="cap_horizon")
        use_weather = st.checkbox("Include Live Weather API Forecasts", value=True, key="cap_weather")
        safety_pct = st.slider("Safety Stock Target (%)", 0.0, 50.0, 10.0, 1.0)
        unit_cost = st.number_input("Estimated Unit Cost ($)", value=10.0, step=1.0)
        
        if st.button("Generate Capacity Plan"):
            with st.spinner("Running forecasting and capacity planning models..."):
                try:
                    w_df = None
                    if use_weather:
                        from demandiq.data.weather_fetcher import fetch_forecast_weather
                        w_dfs = []
                        for c in df["city"].unique():
                            cwd = fetch_forecast_weather(str(c), horizon)
                            if not cwd.empty:
                                w_dfs.append(cwd)
                        if w_dfs:
                            w_df = pd.concat(w_dfs, ignore_index=True)
                            
                    future = forecaster.forecast_future(horizon, df, weather_df=w_df)
                    if not future.empty:
                        planner = CapacityPlanner(safety_stock_pct=safety_pct, default_unit_cost=unit_cost)
                        recs = planner.recommend(future)
                        
                        show_cols = ["date", "city", "pred_orders", "pred_p90", "recommended_inventory", "reorder_point", "risk_level", "max_exposed_cost"]
                        disp_df = recs[show_cols].copy()
                        disp_df["date"] = disp_df["date"].dt.strftime("%Y-%m-%d")
                        disp_df["pred_orders"] = disp_df["pred_orders"].round(1)
                        disp_df["pred_p90"] = disp_df["pred_p90"].round(1)
                        
                        st.dataframe(disp_df, use_container_width=True, hide_index=True)
                        
                        csv = disp_df.to_csv(index=False).encode('utf-8')
                        st.download_button("⬇️ Download Capacity Plan (CSV)", csv, "capacity_plan.csv", "text/csv")
                    else:
                        st.error("Failed to generate future forecast for capacity planning.")
                except Exception as e:
                    st.error(f"Error during capacity planning: {e}")

    with tab_registry:
        st.markdown("### 🔬 Model Registry & A/B Testing")
        st.markdown("Compare versions of the forecasting ensemble head-to-head.")
        
        reg = ModelRegistry()
        versions = reg.list_versions()
        
        if not versions:
            st.info("No models registered yet.")
            st.markdown("Register current model:")
            new_tag = st.text_input("Version Tag (e.g., v1.1.0-promo)")
            desc = st.text_area("Description")
            if st.button("Register Model"):
                reg.register(forecaster, new_tag, desc)
                st.success(f"Registered {new_tag}!")
                st.rerun()
        else:
            st.dataframe(pd.DataFrame(versions), use_container_width=True)
            
            st.markdown("#### A/B Model Comparison")
            c1, c2 = st.columns(2)
            with c1:
                tag_a = st.selectbox("Champion Model", [v["tag"] for v in versions], index=0)
            with c2:
                tag_b = st.selectbox("Challenger Model", [v["tag"] for v in versions], index=min(1, len(versions)-1))
                
            if st.button("Compare Models"):
                with st.spinner("Running head-to-head backtest..."):
                    try:
                        comp = reg.compare(tag_a, tag_b, df)
                        st.write(f"**Improvement of {tag_a} over {tag_b}:**")
                        st.json(comp["improvement"])
                        
                        m_a = comp[tag_a]
                        m_b = comp[tag_b]
                        
                        comp_df = pd.DataFrame([
                            {"Model": tag_a, "MAPE": m_a["mape"], "RMSE": m_a["rmse"], "MAE": m_a["mae"]},
                            {"Model": tag_b, "MAPE": m_b["mape"], "RMSE": m_b["rmse"], "MAE": m_b["mae"]}
                        ])
                        st.dataframe(comp_df, hide_index=True)
                    except Exception as e:
                        st.error(f"Comparison failed: {e}")

    st.markdown("---")
    st.caption(
        "⚡ Built for high-throughput reliability by the DemandIQ ML Engineering Team. Portfolio-grade AI software."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
