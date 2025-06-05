import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit

def load_data(file_path):
    """Load data from JSON file and convert to DataFrame."""
    pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df

def resample_hourly(df):
    """Resample data to hourly averages."""
    return df.resample("1h").mean().dropna()

def create_time_features(df):
    """Create time-based features."""
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["day_of_year"] = df.index.dayofyear
    df["month"] = df.index.month
    return df

def create_lag_features(df):
    """Create lag and rolling average features."""
    df["prev_day_consumption"] = df["energy_kWh"].shift(24)
    df["rolling_7d_avg"] = df["energy_kWh"].rolling(window=24*7).mean()
    df["prev_week_consumption"] = df["energy_kWh"].shift(24*7)
    return df

def create_cyclical_features(df, col_name, period, start_num=0):
    """Create cyclical features using sine and cosine transformations."""
    df[f'sin_{col_name}'] = np.sin(2 * np.pi * (df[col_name] - start_num) / period)
    df[f'cos_{col_name}'] = np.cos(2 * np.pi * (df[col_name] - start_num) / period)
    return df

def create_interaction_features(df):
    """Create interaction features."""
    df['hour_weekend'] = df['hour'] * df['is_weekend']
    return df

def prepare_features(df):
    """Prepare all features."""
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_cyclical_features(df, 'hour', 24, 0)
    df = create_cyclical_features(df, 'dayofweek', 7, 0)
    df = create_cyclical_features(df, 'month', 12, 1)
    df = create_interaction_features(df)
    return df

def scale_data(X, y):
    """Scale features and target."""
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))
    return X_scaled, y_scaled, scaler

def create_sequences(X, y, time_steps=24):
    """Create sequences for LSTM input."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def preprocess_data(file_path, test_size=0.2, random_state=42):
    """Main preprocessing function."""
    # Load and prepare data
    df = load_data(file_path)
    df = resample_hourly(df)
    df = prepare_features(df)
    df.dropna(inplace=True)

    # Define features
    features = ['sin_hour', 'cos_hour', 'sin_dayofweek', 'cos_dayofweek', 'sin_month', 'cos_month',
                'is_weekend', 'day_of_year', 'prev_day_consumption', 'rolling_7d_avg',
                'hour_weekend', 'prev_week_consumption']

    # Prepare X and y
    X = df[features].values
    y = df['energy_kWh'].values.reshape(-1, 1)

    # Scale data
    X_scaled, y_scaled, scaler = scale_data(X, y)

    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled)

    # Split data
    # X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=test_size, random_state=random_state)
    # return X_train, X_test, y_train, y_test, scaler

    # Split data without changing order of time series
    tscv = TimeSeriesSplit()

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    file_path = "../data/lora_data_5_6_25.csv"
    X_train, X_test, y_train, y_test, scaler = preprocess_data(file_path)
    print("Preprocessing complete. Data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")