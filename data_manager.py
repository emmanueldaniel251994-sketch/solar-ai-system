import pandas as pd


def load_solar_data(filepath):
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    return df


def summarize_data(df):
    summary = df.describe()
    return summary


def average_generation(df):
    return df["solar_generation_kwh"].mean()


def average_consumption(df):
    return df["consumption_kwh"].mean()


def days_with_deficit(df):
    deficit_days = df[df["solar_generation_kwh"] < df["consumption_kwh"]]
    return deficit_days


def lowest_battery_day(df):
    row = df.loc[df["battery_percent"].idxmin()]
    return row