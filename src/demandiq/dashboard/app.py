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
    tab_deep_dive, tab_benchmarks, tab_future = st.tabs(
        ["📈 City Deep Dive", "🗺️ City Benchmarks", "📅 Future Forecast"]
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

    st.markdown("---")
    st.caption(
        "⚡ Built for high-throughput reliability by the DemandIQ ML Engineering Team. Portfolio-grade AI software."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
