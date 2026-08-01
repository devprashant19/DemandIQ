"""Interactive Streamlit dashboard app for demand forecasting and anomaly diagnostics."""

import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandiq.anomaly.detector import HybridAnomalyDetector
from demandiq.config import settings
from demandiq.data.loader import load_and_validate_orders
from demandiq.features.engineer import build_features
from demandiq.models.cross_validate import compute_metrics
from demandiq.models.explain import get_top_drivers
from demandiq.models.forecaster import DemandForecaster

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


def main() -> None:
    """Render interactive Streamlit application components and visual plots."""
    apply_custom_theme()

    st.markdown('<div class="gradient-title">⚡ DemandIQ Analytics Platform</div>', unsafe_allow_html=True)
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

    # Run predictions and detect residual anomalies for filtered view
    with st.spinner("Executing real-time ensemble inference..."):
        sub_preds = forecaster.predict(sub_df)
        sub_df["pred_orders"] = sub_preds
        sub_df["residuals"] = sub_df["orders"] - sub_df["pred_orders"]
        sub_df["is_anomaly"] = detector.predict(sub_df["residuals"].to_numpy())

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
                marker=dict(color="#F43F5E", size=12, symbol="diamond-open", line=dict(width=2, color="#F43F5E")),
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
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Section 3: SHAP Feature Explanations ---
    st.markdown("### 🧠 SHAP Tree Attribution Drivers")
    st.markdown(
        "Inspect explicit feature importance drivers explaining the model prediction for a specific calendar observation."
    )

    date_options = sub_df["date"].dt.strftime("%Y-%m-%d").tolist()
    selected_date_str = str(st.selectbox("Choose Specific Observation Date", options=date_options, index=len(date_options) - 1))
    
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
            xaxis=dict(title="SHAP Order Attribution Magnitude (vs. City Baseline)", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(autorange="reversed"),
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(shap_fig, use_container_width=True)
    
    st.markdown("---")
    st.caption("⚡ Built for high-throughput reliability by the DemandIQ ML Engineering Team. Portfolio-grade AI software.")


if __name__ == "__main__":
    main()
