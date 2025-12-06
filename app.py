import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import plotly.graph_objects as go

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="NYC Taxi – Anomaly Intelligence Dashboard",
    layout="wide",
    page_icon="🚕",
)

# ------------------------------------------------------------
# FULL ENTERPRISE UI + FIXED SCREEN + BANNER
# ------------------------------------------------------------
st.markdown("""
<style>

    /* REMOVE ALL SCROLLING */
    html, body {
        overflow: hidden !important;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #052836 !important;
        overflow: hidden !important;
        height: 100vh !important;
    }

    /* MAIN CONTAINER (NO SCROLL) */
    .block-container {
        padding: 0 !important;
        margin: 0 !important;
        height: 100vh !important;
        display: flex;
        flex-direction: column;
        overflow: hidden !important;
    }

    /* TOP HERO IMAGE */
    .hero-banner {
        width: 100%;
        height: 180px;
        background-image: url('https://images.unsplash.com/photo-1494976388531-d1058494cdd8');
        background-size: cover;
        background-position: center;
        border-bottom: 4px solid #0A3D4A;
        position: relative;
    }

    /* HERO OVERLAY TEXT */
    .hero-text {
        position: absolute;
        bottom: 15px;
        left: 40px;
        color: white;
        font-size: 38px;
        font-weight: 700;
        text-shadow: 0px 0px 10px #000;
    }

    .hero-small {
        font-size: 18px;
        opacity: 0.85;
    }

    /* NYC TAXI ICON */
    .taxi-icon {
        position: absolute;
        bottom: 10px;
        right: 40px;
        width: 140px;
        filter: drop-shadow(0px 0px 10px black);
    }

    /* SIDE ICON BAR */
    .sidebar-icons {
        position: fixed;
        top: 180px;
        left: 0;
        width: 70px;
        height: calc(100vh - 180px);
        background: #083645;
        border-right: 1px solid #0F4A5A;
        padding-top: 20px;
        z-index: 999;
    }

    .sidebar-icons a {
        display: block;
        text-align: center;
        padding: 22px 0;
        font-size: 24px;
        text-decoration: none;
        color: #BFE8FF;
    }

    .sidebar-icons a:hover {
        background-color: #0F4A5A;
        color: white;
    }

    /* MAIN CONTENT AREA */
    .main-section {
        margin-left: 90px;
        margin-top: 10px;
        height: calc(100vh - 190px);
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }

    /* KPI ROW */
    .kpi-row {
        flex: 0 0 130px;
        display: flex;
        gap: 15px;
    }

    .panel {
        flex: 1;
        background: linear-gradient(135deg, #073442, #0D3F52);
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #0F4A5A;
        color: #D4ECF2;
        box-shadow: 0px 0px 8px rgba(0,0,0,0.3);
    }

    /* CHART */
    .chart-area {
        flex: 1;
        overflow: hidden;
        margin-top: 10px;
    }

    .chart-area iframe {
        height: 100% !important;
    }

    /* TABLE */
    .table-area {
        flex: 0 0 210px;
        margin-top: 10px;
        overflow: hidden;
    }

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# HERO BANNER
# ------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-text">
        NYC TAXI ANOMALY INTELLIGENCE
        <div class="hero-small">Real-time insights · Forecasting · Behaviour shifts</div>
    </div>
    <img class="taxi-icon" src="https://images.unsplash.com/photo-1534237710431-e2fc698436d0">
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# LEFT SIDEBAR ICON BAR
# ------------------------------------------------------------
st.markdown("""
<div class="sidebar-icons">
    <a href="#">⚠️</a>
    <a href="#">📈</a>
    <a href="#">📊</a>
    <a href="#">🧠</a>
    <a href="#">⚙️</a>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# MAIN SECTION WRAPPER
# ------------------------------------------------------------
st.markdown("<div class='main-section'>", unsafe_allow_html=True)

# ------------------------------------------------------------
# SIDEBAR — UPLOAD DATA
# ------------------------------------------------------------
df_file = st.sidebar.file_uploader("Upload NYC Taxi Hourly CSV", type=["csv"])

if df_file is None:
    st.warning("Upload your hourly NYC Taxi dataset to continue.")
    st.stop()

df = pd.read_csv(df_file)
df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])
df = df.sort_values("pickup_hour")

metrics = [
    "trip_count",
    "total_amount",
    "avg_speed_mph",
    "avg_trip_distance",
    "avg_trip_time_min",
    "avg_tip_percentage",
    "avg_fare_per_trip",
]

# ------------------------------------------------------------
# MULTI-METRIC ANOMALY ENGINE
# ------------------------------------------------------------
features = df[metrics]

# Z-Score
z = (features - features.mean()) / features.std()
df["zscore"] = z.abs().max(axis=1)
df["z_anom"] = df["zscore"] > 2.5

# IsolationForest
scale = StandardScaler().fit_transform(features)
iso = IsolationForest(contamination=0.06, random_state=42)
df["iso"] = iso.fit_predict(scale)
df["iso"] = df["iso"].apply(lambda x: x == -1)

# Combined
df["anom"] = df["z_anom"] | df["iso"]

total_anom = int(df["anom"].sum())
critical_iso = int(df["iso"].sum())
severity = round(df["zscore"].mean() * 3, 1)
normality = round(100 - (total_anom / len(df) * 100), 1)

# ------------------------------------------------------------
# KPI PANEL ROW
# ------------------------------------------------------------
st.markdown("<div class='kpi-row'>", unsafe_allow_html=True)

st.markdown(f"""
<div class='panel'>
    <h4>Overall Anomaly Score</h4>
    <h2>{severity} / 10</h2>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class='panel'>
    <h4>Total Anomaly Points</h4>
    <h2>{total_anom}</h2>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class='panel'>
    <h4>Critical (IsolationForest)</h4>
    <h2>{critical_iso}</h2>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class='panel'>
    <h4>Estimated Normal Behaviour</h4>
    <h2>{normality}%</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # END KPI ROW

# ------------------------------------------------------------
# METRIC SELECTION
# ------------------------------------------------------------
metric = st.selectbox("Select a metric", metrics, index=0)

# ------------------------------------------------------------
# ANOMALY CHART DISPLAY
# ------------------------------------------------------------
st.markdown("<div class='chart-area'>", unsafe_allow_html=True)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["pickup_hour"],
    y=df[metric],
    mode="lines",
    line=dict(color="#53C1FF", width=2),
    name=metric,
))

anom_df = df[df["anom"]]
fig.add_trace(go.Scatter(
    x=anom_df["pickup_hour"],
    y=anom_df[metric],
    mode="markers",
    marker=dict(color="yellow", size=9),
    name="Anomalies",
))

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0D3A4A",
    plot_bgcolor="#0A3040",
    margin=dict(l=10, r=10, t=10, b=10),
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# ANOMALY TABLE (FIXED HEIGHT)
# ------------------------------------------------------------
st.markdown("<div class='table-area'>", unsafe_allow_html=True)

anom_table = df[df["anom"]].nlargest(30, "zscore")
st.dataframe(anom_table[["pickup_hour", metric, "zscore"]])

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)  # END MAIN SECTION
