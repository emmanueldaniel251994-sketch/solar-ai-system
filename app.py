import streamlit as st
import pandas as pd

from solar_calculator import (
    calculate_solar_capacity,
    calculate_daily_generation,
    calculate_battery_backup,
    calculate_energy_balance
)
from data_manager import load_solar_data, average_generation, average_consumption, days_with_deficit
from predictor import prepare_features, train_model, predict_next_day
from anomaly_detector import detect_statistical_anomalies, detect_ml_anomalies, diagnose_low_generation
from ai_assistant import build_context, ask_solar_assistant
st.set_page_config(page_title="Tambo Solar AI Dashboard", layout="wide")

st.title("☀️ Tambo Solar AI Dashboard")
st.caption("AI-Powered Solar Energy Monitoring & Prediction System")

# ---- Sidebar: system inputs ----
st.sidebar.header("System Configuration")
panel_wattage = st.sidebar.number_input("Panel wattage (W)", value=450.0)
number_of_panels = st.sidebar.number_input("Number of panels", value=4, step=1)
peak_sun_hours = st.sidebar.number_input("Peak sun hours", value=5.0)
battery_capacity = st.sidebar.number_input("Battery capacity (kWh)", value=5.0)
daily_consumption = st.sidebar.number_input("Daily consumption (kWh)", value=8.0)

# ---- Calculator section ----
total_capacity = calculate_solar_capacity(panel_wattage, number_of_panels)
daily_generation = calculate_daily_generation(total_capacity, peak_sun_hours)
backup_days = calculate_battery_backup(battery_capacity, daily_consumption)
energy_balance = calculate_energy_balance(daily_generation, daily_consumption)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Capacity", f"{total_capacity/1000:.2f} kW")
col2.metric("Est. Daily Generation", f"{daily_generation:.2f} kWh")
col3.metric("Battery Backup", f"{backup_days:.2f} days")
col4.metric("Energy Balance", f"{energy_balance:.2f} kWh")

if energy_balance > 0:
    st.success("System is producing a surplus.")
elif energy_balance < 0:
    st.warning("System is running a deficit.")
else:
    st.info("Production matches consumption.")

st.divider()

# ---- Historical data section ----
st.header("📊 Historical Data")

df = load_solar_data("solar_data.csv")
st.dataframe(df, use_container_width=True)

col1, col2 = st.columns(2)
col1.metric("Average Generation", f"{average_generation(df):.2f} kWh")
col2.metric("Average Consumption", f"{average_consumption(df):.2f} kWh")

deficit_days = days_with_deficit(df)
if len(deficit_days) > 0:
    st.warning(f"{len(deficit_days)} day(s) with energy deficit")
    st.dataframe(deficit_days, use_container_width=True)

st.divider()

# ---- Charts section ----
st.header("📈 Performance Charts")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Generation vs Consumption")
    st.line_chart(df.set_index("date")[["solar_generation_kwh", "consumption_kwh"]])
with col2:
    st.subheader("Battery Level Over Time")
    st.line_chart(df.set_index("date")[["battery_percent"]])

st.subheader("Generation vs Temperature")
st.scatter_chart(df, x="temperature_c", y="solar_generation_kwh")

st.divider()

# ---- AI Prediction section ----
st.header("🤖 AI Prediction")

features, target = prepare_features(df)
model, error = train_model(features, target)

latest = df.sort_values("date").iloc[-1]
predicted = predict_next_day(
    model,
    latest["solar_generation_kwh"],
    latest["temperature_c"],
    latest["consumption_kwh"]
)

col1, col2 = st.columns(2)
col1.metric("Model Error (MAE)", f"{error:.2f} kWh")
col2.metric("Predicted Tomorrow", f"{predicted:.2f} kWh")

st.divider()
st.header("⚠️ Fault Detection")

stat_anomalies, mean_gen, std_gen = detect_statistical_anomalies(df)
ml_anomalies = detect_ml_anomalies(df)

col1, col2 = st.columns(2)
col1.metric("Statistical Anomalies", len(stat_anomalies))
col2.metric("ML-Detected Anomalies", len(ml_anomalies))

if len(stat_anomalies) > 0:
    st.warning("Days with unusually low generation:")
    st.dataframe(stat_anomalies[["date", "solar_generation_kwh", "expected_range"]], use_container_width=True)

    worst_day = stat_anomalies.iloc[0]
    st.subheader(f"Diagnosis for {worst_day['date'].date()}")
    reasons = diagnose_low_generation(
        worst_day["solar_generation_kwh"], mean_gen,
        worst_day["temperature_c"], worst_day["battery_percent"]
    )
    for r in reasons:
        st.write(f"- {r}")
else:
    st.success("No anomalies detected in current data.")

st.caption(f"Based on {latest['date'].date()}: generation {latest['solar_generation_kwh']} kWh, "
           f"temperature {latest['temperature_c']}°C, consumption {latest['consumption_kwh']} kWh")

st.divider()
st.header("🤖 Ask the Solar Assistant")

data_context = build_context(df, stat_anomalies, ml_anomalies, mean_gen, std_gen, predicted)

user_question = st.text_input("Ask a question about your solar system:")

if st.button("Ask") and user_question:
    with st.spinner("Thinking..."):
        answer = ask_solar_assistant(user_question, data_context)
    st.write(answer)