import matplotlib.pyplot as plt
import seaborn as sns


def plot_generation_vs_consumption(df, save_path="generation_vs_consumption.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["solar_generation_kwh"], marker="o", label="Generation (kWh)")
    plt.plot(df["date"], df["consumption_kwh"], marker="o", label="Consumption (kWh)")
    plt.title("Solar Generation vs Consumption")
    plt.xlabel("Date")
    plt.ylabel("Energy (kWh)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_battery_level(df, save_path="battery_level.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["battery_percent"], marker="o", color="green")
    plt.axhline(y=20, color="red", linestyle="--", label="Low battery threshold")
    plt.title("Battery Level Over Time")
    plt.xlabel("Date")
    plt.ylabel("Battery (%)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_generation_vs_temperature(df, save_path="generation_vs_temperature.png"):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="temperature_c", y="solar_generation_kwh", s=100)
    plt.title("Solar Generation vs Temperature")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Solar Generation (kWh)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_correlation_heatmap(df, save_path="correlation_heatmap.png"):
    plt.figure(figsize=(6, 5))
    numeric_df = df[["solar_generation_kwh", "consumption_kwh", "temperature_c", "battery_percent"]]
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")