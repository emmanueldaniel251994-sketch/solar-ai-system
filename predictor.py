import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def prepare_features(df):
    """
    Build a feature set where each row uses TODAY's data
    to predict TOMORROW's solar generation.
    """
    data = df.copy()
    data = data.sort_values("date").reset_index(drop=True)

    # Tomorrow's generation is what we want to predict
    data["next_day_generation"] = data["solar_generation_kwh"].shift(-1)

    # Drop the last row (it has no "tomorrow" to predict)
    data = data.dropna(subset=["next_day_generation"])

    features = data[["solar_generation_kwh", "temperature_c", "consumption_kwh"]]
    target = data["next_day_generation"]

    return features, target


def train_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    error = mean_absolute_error(y_test, predictions)

    return model, error


def predict_next_day(model, today_generation, today_temperature, today_consumption):
    input_data = pd.DataFrame([{
        "solar_generation_kwh": today_generation,
        "temperature_c": today_temperature,
        "consumption_kwh": today_consumption
    }])
    prediction = model.predict(input_data)
    return prediction[0]