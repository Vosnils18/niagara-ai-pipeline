import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=["meter_timestamp"])
    df["meter_timestamp"] = pd.to_datetime(df["meter_timestamp"])
    df = df.set_index("meter_timestamp")
    return df

def add_cost_features(df, low_tariff=0.20, normal_tariff=0.25):
    df["import_low_tariff_delta_Wh"] = df["active_import_low_tariff_Wh"].diff().clip(lower=0)
    df["import_normal_tariff_delta_Wh"] = df["active_import_normal_tariff_Wh"].diff().clip(lower=0)
    df["cost_low_tariff"] = df["import_low_tariff_delta_Wh"] * (low_tariff / 1000)
    df["cost_normal_tariff"] = df["import_normal_tariff_delta_Wh"] * (normal_tariff / 1000)
    df["cost_total"] = df["cost_low_tariff"] + df["cost_normal_tariff"]
    df = df.dropna()
    return df

def create_time_features(df):
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["dayofweek"] = df.index.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    return df

def create_lag_features(df, lag=1):
    for col in [
        "current_L1_A", "current_L2_A", "current_L3_A",
        "voltage_L1_V", "voltage_L2_V", "voltage_L3_V",
        "total_active_import_power_W", "cost_total"
    ]:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

def create_cyclical_features(df):
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_minute"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["cos_minute"] = np.cos(2 * np.pi * df["minute"] / 60)
    df["sin_dayofweek"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["cos_dayofweek"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df

def prepare_features(df):
    df = create_time_features(df)
    df = create_lag_features(df, lag=1)
    df = create_cyclical_features(df)
    df = df.dropna()
    return df

def scale_data(X_train_raw, X_test_raw, y_train_raw, y_test_raw):
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_train_scaled = x_scaler.fit_transform(X_train_raw)
    X_test_scaled = x_scaler.transform(X_test_raw)
    y_train_scaled = y_scaler.fit_transform(y_train_raw.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test_raw.reshape(-1, 1))
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, x_scaler, y_scaler

def create_sequences(X, y, time_steps=144):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def preprocess_data(file_path, low_tariff=0.20, normal_tariff=0.25, test_size=0.2, time_steps=144):
    df = load_data(file_path)
    df = add_cost_features(df, low_tariff=low_tariff, normal_tariff=normal_tariff)
    df = prepare_features(df)

    features = [
        "current_L1_A", "current_L2_A", "current_L3_A",
        "voltage_L1_V", "voltage_L2_V", "voltage_L3_V", "cost_total_lag1",
        "current_L1_A_lag1", "current_L2_A_lag1", "current_L3_A_lag1",
        "voltage_L1_V_lag1", "voltage_L2_V_lag1", "voltage_L3_V_lag1",
        "total_active_import_power_W_lag1",
        "hour", "minute", "is_weekend",
        "sin_hour", "cos_hour", "sin_minute", "cos_minute",
        "sin_dayofweek", "cos_dayofweek"
    ]

    X = df[features].values
    y = df["total_active_import_power_W"].values

    split_idx = int(len(X) * (1 - test_size))
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train_raw, y_test_raw = y[:split_idx], y[split_idx:]

    X_train, X_test, y_train, y_test, x_scaler, y_scaler = scale_data(
        X_train_raw, X_test_raw, y_train_raw, y_test_raw
    )

    X_train, y_train = create_sequences(X_train, y_train, time_steps=time_steps)
    X_test, y_test = create_sequences(X_test, y_test, time_steps=time_steps)

    return X_train, X_test, y_train, y_test, x_scaler, y_scaler

if __name__ == "__main__":
    file_path = "./data/lora_data_12_6_25.csv"
    X_train, X_test, y_train, y_test, x_scaler, y_scaler = preprocess_data(
        file_path, time_steps=6  # 24 hours of 10-min intervals
    )
    print("Preprocessing complete. Data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
