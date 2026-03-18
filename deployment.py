import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="SolarGen AI | Energy Intelligence",
    page_icon="☀️",
    layout="wide"
)

# --- UI ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0b1220, #020617);
    color: #f9fafb;
}

[data-testid="stSidebar"] {
    background: #020617 !important;
    border-right: 1px solid #1f2937;
}

[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

div[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1f2937;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.4);
    transition: 0.3s;
}

div[data-testid="metric-container"]:hover {
    transform: translateY(-5px);
    border: 1px solid #facc15;
}

div[data-testid="stMetricLabel"] {
    color: #9ca3af !important;
}

div[data-testid="stMetricValue"] {
    color: #f9fafb !important;
    font-weight: 800;
    font-size: 1.8rem;
}

.hero-text {
    color: #facc15;
    font-size: 3rem;
    font-weight: 800;
}

.sub-text {
    color: #9ca3af;
    margin-bottom: 2rem;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #facc15, #eab308);
}

hr {
    border-color: #1f2937;
}
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        return joblib.load("solar_pipeline.pkl")
    except:
        return None

pipeline = load_model()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <h2 style='color:#facc15;'>⚡ SolarGen AI</h2>
    <p style='color:#9ca3af;'>Energy Intelligence Suite v2.5</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### ⚡ Solar Input Parameters")
    irrad = st.slider("Solar Irradiance (W/m²)", 0.0, 1500.0, 800.0)

    st.markdown("### 🌡 Thermal Parameters")
    temp_amb = st.slider("Ambient Temperature (°C)", 0.0, 60.0, 25.0)
    temp_mod = st.slider("Module Temperature (°C)", 0.0, 80.0, 45.0)

    st.markdown("### 🕒 Temporal Parameters")
    hour = st.select_slider("Hour of Day", range(24), value=12)

    col1, col2 = st.columns(2)
    month = col1.selectbox("Month", range(1, 13), index=5)
    day = col2.number_input("Day", 1, 31, 15)

# --- HEADER ---
st.markdown('<p class="hero-text">☀️ Solar Energy Intelligence Platform</p>', unsafe_allow_html=True)
st.markdown(
'<p class="sub-text">Real-time predictive analytics and performance monitoring for photovoltaic systems.</p>',
unsafe_allow_html=True
)

# --- INPUT DATA ---
input_df = pd.DataFrame([{
    "IRRADIATION": irrad,
    "AMBIENT_TEMPERATURE": temp_amb,
    "MODULE_TEMPERATURE": temp_mod,
    "hour": hour,
    "day": day,
    "month": month
}])

# --- PREDICTION ---
if pipeline is not None:

    prediction = pipeline.predict(input_df)
    final_output = max(0.0, float(prediction[0]))

    capacity = 5000
    load_percent = (final_output / capacity) * 100

    # --- METRICS ---
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("⚡ AC Power Output", f"{final_output:,.2f} kW")
    c2.metric("☀️ Solar Irradiance", f"{irrad} W/m²")

    delta = temp_mod - temp_amb
    c3.metric("🌡 Module Thermal Gradient", f"{delta:.1f} °C")

    c4.metric("📊 Capacity Utilization", f"{load_percent:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- STATUS ---
    col_main, col_status = st.columns([2, 1])

    with col_main:
        st.markdown("### 📊 Generation Utilization Overview")

        st.progress(min(1.0, load_percent / 100))
        st.write(f"**{final_output:,.2f} kW / {capacity} kW Installed Capacity**")

        st.info(
            f"📅 {day}/{month} | ⏰ {hour}:00\n\n"
            f"Operating Condition: {'High Irradiance Regime ☀️' if irrad > 1000 else 'Standard Irradiance Conditions'}"
        )

    with col_status:
        st.markdown("### 🔔 Operational Status")
# Dynamic capacity
capacity = max(1000, final_output * 1.3)
load_percent = (final_output / capacity) * 100

# Better thresholds
if load_percent <= 2:
    st.error("🛑 NO GENERATION")
elif load_percent <= 25:
    st.warning("⚠️ LOW GENERATION")
elif load_percent <= 60:
    st.info("⚡ NORMAL PERFORMANCE")
else:
    st.success("✅ OPTIMAL / PEAK")

    # --- AI INSIGHT ---
    st.markdown("### 🤖 AI Insight")

    # simulate full day
    hours = list(range(24))
    power_predictions = []

    for h in hours:
        temp_df = pd.DataFrame([{
            "IRRADIATION": irrad,
            "AMBIENT_TEMPERATURE": temp_amb,
            "MODULE_TEMPERATURE": temp_mod,
            "hour": h,
            "day": day,
            "month": month
        }])
        pred = pipeline.predict(temp_df)
        power_predictions.append(max(0.0, float(pred[0])))

    peak_hour = hours[np.argmax(power_predictions)]

    st.success(f"Peak generation expected around **{peak_hour}:00 hrs**.")

    # --- GRAPHS ---
    st.markdown("### 📈 Performance Analytics")

    df_hourly = pd.DataFrame({
        "Hour": hours,
        "Power (kW)": power_predictions
    })

    fig1 = px.line(df_hourly, x="Hour", y="Power (kW)", markers=True,
                   title="Daily Power Output Forecast")
    fig1.update_layout(template="plotly_dark")
    st.plotly_chart(fig1, use_container_width=True)

    # Irradiance vs Power
    irradiance_range = np.linspace(0, 1500, 30)
    power_curve = []

    for irr in irradiance_range:
        temp_df = pd.DataFrame([{
            "IRRADIATION": irr,
            "AMBIENT_TEMPERATURE": temp_amb,
            "MODULE_TEMPERATURE": temp_mod,
            "hour": hour,
            "day": day,
            "month": month
        }])
        pred = pipeline.predict(temp_df)
        power_curve.append(max(0.0, float(pred[0])))

    df_curve = pd.DataFrame({
        "Irradiance": irradiance_range,
        "Power": power_curve
    })

    fig2 = px.line(df_curve, x="Irradiance", y="Power",
                   title="Power vs Solar Irradiance")
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

    # --- DATA ---
    with st.expander("📄 Operational Telemetry Log"):
        st.dataframe(input_df, use_container_width=True)

else:
    st.error("❌ Model file 'solar_pipeline.pkl' not found.")

# --- FOOTER ---
st.markdown("---")
st.caption("© 2026 SolarGen AI Systems | Real-Time Energy Intelligence Dashboard")
