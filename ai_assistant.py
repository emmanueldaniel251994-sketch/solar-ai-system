import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=api_key)


def build_context(df, stat_anomalies, ml_anomalies, mean_gen, std_gen, prediction=None):
    """
    Summarizes your solar data into text the AI can reason over.
    """
    latest = df.sort_values("date").iloc[-1]

    context = f"""
Solar System Data Summary:
- Average daily generation: {mean_gen:.2f} kWh (std dev: {std_gen:.2f})
- Latest day ({latest['date'].date()}):
    Generation: {latest['solar_generation_kwh']} kWh
    Consumption: {latest['consumption_kwh']} kWh
    Temperature: {latest['temperature_c']}°C
    Battery: {latest['battery_percent']}%
- Number of statistically low-generation days: {len(stat_anomalies)}
- Number of ML-flagged anomaly days: {len(ml_anomalies)}
"""

    if prediction is not None:
        context += f"- Predicted generation for tomorrow: {prediction:.2f} kWh\n"

    if len(stat_anomalies) > 0:
        context += "\nLow generation days:\n"
        for _, row in stat_anomalies.iterrows():
            context += f"  {row['date'].date()}: {row['solar_generation_kwh']} kWh\n"

    return context


def ask_solar_assistant(question, data_context):
    system_prompt = (
        "You are a helpful solar energy diagnostic assistant. "
        "You are given real data from a user's solar power system. "
        "Answer their question clearly and practically, referencing "
        "the actual numbers in the data where relevant. Keep answers "
        "concise and actionable. If the data doesn't fully explain "
        "something, say so honestly rather than guessing."
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=500,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Solar system data:\n{data_context}\n\nQuestion: {question}"}
        ]
    )

    return response.choices[0].message.content