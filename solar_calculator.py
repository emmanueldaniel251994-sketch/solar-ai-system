import numpy as np


def calculate_solar_capacity(panel_wattage, number_of_panels):
    panels = np.array([panel_wattage] * number_of_panels)

    total_capacity = np.sum(panels)

    return total_capacity


def calculate_daily_generation(total_capacity_watts, peak_sun_hours):
    total_capacity_kw = total_capacity_watts / 1000

    daily_generation = total_capacity_kw * peak_sun_hours

    return daily_generation


def calculate_battery_backup(battery_capacity_kwh, daily_consumption_kwh):
    if daily_consumption_kwh <= 0:
        return 0

    backup_days = battery_capacity_kwh / daily_consumption_kwh

    return backup_days


def calculate_energy_balance(solar_generation, consumption):
    balance = solar_generation - consumption

    return balance