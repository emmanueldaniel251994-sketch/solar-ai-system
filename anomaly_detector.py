import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_statistical_anomalies(df, column="solar_generation_kwh", threshold=1.5):
    """
    Flags days where generation is more than `threshold` standard
    deviations below the mean — a simple, explainable method.
    """
    mean = df[column].mean()
    std = df[column].std()

    lower_bound = mean - (threshold * std)

    anomalies = df[df[column] < lower_bound].copy()
    anomalies["expected_range"] = f"{lower_bound:.2f}+ kWh"

    return anomalies, mean, std


def detect_ml_anomalies(df):
    """
    Uses Isolation Forest to detect unusual days across
    generation, consumption, temperature and battery together.
    """
    features = df[["solar_generation_kwh", "consumption_kwh", "temperature_c", "battery_percent"]]

    model = IsolationForest(contamination=0.15, random_state=42)
    predictions = model.fit_predict(features)

    df_copy = df.copy()
    df_copy["anomaly"] = predictions
    # -1 means anomaly, 1 means normal

    anomalies = df_copy[df_copy["anomaly"] == -1]

    return anomalies


def diagnose_low_generation(current_generation, expected_generation, temperature, battery_percent):
    """
    Gives a simple rule-based explanation for why generation might be low.
    This is the seed of your Phase 7 AI assistant.
    """
    reasons = []

    if temperature > 32:
        reasons.append("High temperature can reduce panel efficiency.")

    if battery_percent < 30:
        reasons.append("Low battery may indicate insufficient charging over recent days.")

    deficit = expected_generation - current_generation
    if deficit > 0:
        reasons.append(f"Generation is {deficit:.2f} kWh below your average — possible shading, dirt on panels, or cloud cover.")

    if not reasons:
        reasons.append("No obvious cause found in current data. Consider checking wiring, inverter status, or panel cleanliness manually.")

    return reasons