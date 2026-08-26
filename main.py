from solar_calculator import (
    calculate_solar_capacity,
    calculate_daily_generation,
    calculate_battery_backup,
    calculate_energy_balance
)
from data_manager import (
    load_solar_data,
    summarize_data,
    average_generation,
    average_consumption,
    days_with_deficit,
    lowest_battery_day
)
from visualizer import (
    plot_generation_vs_consumption,
    plot_battery_level,
    plot_generation_vs_temperature,
    plot_correlation_heatmap
)
from predictor import prepare_features, train_model, predict_next_day

from anomaly_detector import detect_statistical_anomalies, detect_ml_anomalies, diagnose_low_generation  


print("=" * 50)
print("       TAMBO SOLAR AI SYSTEM")
print("       SOLAR ENERGY CALCULATOR")
print("=" * 50)


# Solar panel information
panel_wattage = float(input("Enter panel wattage (W): "))
number_of_panels = int(input("Enter number of panels: "))

# Solar conditions
peak_sun_hours = float(input("Enter average peak sun hours: "))

# Battery and consumption
battery_capacity = float(input("Enter battery capacity (kWh): "))
daily_consumption = float(input("Enter daily energy consumption (kWh): "))


# Calculations
total_capacity = calculate_solar_capacity(
    panel_wattage,
    number_of_panels
)

daily_generation = calculate_daily_generation(
    total_capacity,
    peak_sun_hours
)

backup_days = calculate_battery_backup(
    battery_capacity,
    daily_consumption
)

energy_balance = calculate_energy_balance(
    daily_generation,
    daily_consumption
)


# Results
print("\n" + "=" * 50)
print("             SOLAR SYSTEM REPORT")
print("=" * 50)

print(f"Total Solar Capacity: {total_capacity:.0f} W")
print(f"Total Solar Capacity: {total_capacity / 1000:.2f} kW")
print(f"Estimated Daily Generation: {daily_generation:.2f} kWh")
print(f"Battery Capacity: {battery_capacity:.2f} kWh")
print(f"Estimated Backup: {backup_days:.2f} days")
print(f"Daily Consumption: {daily_consumption:.2f} kWh")
print(f"Energy Balance: {energy_balance:.2f} kWh")


if energy_balance > 0:
    print("\nSTATUS: Solar production exceeds consumption.")
elif energy_balance < 0:
    print("\nSTATUS: Solar production is below consumption.")
else:
    print("\nSTATUS: Solar production matches consumption.")

print("=" * 50)

print("\n" + "=" * 50)
print("           HISTORICAL DATA ANALYSIS")
print("=" * 50)

df = load_solar_data("solar_data.csv")

print("\nFirst few records:")
print(df.head())

print("\nStatistical Summary:")
print(summarize_data(df))

avg_gen = average_generation(df)
avg_con = average_consumption(df)

print(f"\nAverage Daily Generation: {avg_gen:.2f} kWh")
print(f"Average Daily Consumption: {avg_con:.2f} kWh")

deficit_days = days_with_deficit(df)
print(f"\nDays with energy deficit: {len(deficit_days)}")
if len(deficit_days) > 0:
    print(deficit_days[["date", "solar_generation_kwh", "consumption_kwh"]])

low_day = lowest_battery_day(df)
print(f"\nLowest battery day: {low_day['date'].date()} at {low_day['battery_percent']}%")

print("=" * 50)

print("\n" + "=" * 50)
print("           GENERATING VISUALIZATIONS")
print("=" * 50)

plot_generation_vs_consumption(df)
plot_battery_level(df)
plot_generation_vs_temperature(df)
plot_correlation_heatmap(df)

print("\nAll charts saved in your project folder.")
print("=" * 50)

print("\n" + "=" * 50)
print("           AI SOLAR PREDICTION")
print("=" * 50)

features, target = prepare_features(df)
model, error = train_model(features, target)

print(f"\nModel trained. Average prediction error: {error:.2f} kWh")

# Use the most recent day's data to predict tomorrow
latest = df.sort_values("date").iloc[-1]

predicted = predict_next_day(
    model,
    latest["solar_generation_kwh"],
    latest["temperature_c"],
    latest["consumption_kwh"]
)

print(f"\nBased on today ({latest['date'].date()}):")
print(f"  Generation: {latest['solar_generation_kwh']} kWh")
print(f"  Temperature: {latest['temperature_c']}°C")
print(f"  Consumption: {latest['consumption_kwh']} kWh")
print(f"\nPredicted generation for tomorrow: {predicted:.2f} kWh")
print("=" * 50)

print("\n" + "=" * 50)
print("           ANOMALY / FAULT DETECTION")
print("=" * 50)

stat_anomalies, mean_gen, std_gen = detect_statistical_anomalies(df)
print(f"\nAverage generation: {mean_gen:.2f} kWh (±{std_gen:.2f})")
print(f"Statistical anomalies found: {len(stat_anomalies)}")
if len(stat_anomalies) > 0:
    print(stat_anomalies[["date", "solar_generation_kwh", "expected_range"]])

ml_anomalies = detect_ml_anomalies(df)
print(f"\nML-based anomalies found: {len(ml_anomalies)}")
if len(ml_anomalies) > 0:
    print(ml_anomalies[["date", "solar_generation_kwh", "consumption_kwh", "temperature_c", "battery_percent"]])

if len(stat_anomalies) > 0:
    worst_day = stat_anomalies.iloc[0]
    print(f"\nDiagnosing worst day: {worst_day['date'].date()}")
    reasons = diagnose_low_generation(
        worst_day["solar_generation_kwh"],
        mean_gen,
        worst_day["temperature_c"],
        worst_day["battery_percent"]
    )
    for r in reasons:
        print(f"  - {r}")

print("=" * 50)