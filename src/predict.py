import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import timedelta

from preprocessing import preprocess_data  # this is your updated preprocess_data function

def load_latest_data(file_path, time_steps=144):
    """
    Load and preprocess the latest data for prediction.
    time_steps=144 for 24h if using 10-min intervals.
    """
    X_train, X_test, y_train, y_test, x_scaler, y_scaler = preprocess_data(
        file_path, time_steps=time_steps
    )
    # Use the last available sequence in your test set as the prediction seed
    initial_sequence = X_test[-1:]  # shape: (1, time_steps, num_features)
    return initial_sequence, y_scaler

def predict_future(model, initial_sequence, y_scaler, time_steps=144):
    predictions = []
    current_sequence = initial_sequence.copy()
    print("AYO", current_sequence)

    for _ in range(time_steps):
        # Predict the next hour
        next_pred = model.predict(current_sequence, verbose=0)
        predictions.append(next_pred[0, 0])

        # Update the sequence for the next prediction
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, :] = next_pred

    # Inverse transform the predictions
    predictions = y_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

def plot_with_past(file_path, predictions, time_steps, interval_minutes=10):
    """
    Plot the last 2 days of actual data, then append predictions (in orange).
    """
    # Load original df for timestamps and cost
    df = pd.read_csv(file_path, parse_dates=["meter_timestamp"])
    df["meter_timestamp"] = pd.to_datetime(df["meter_timestamp"])
    df = df.set_index("meter_timestamp")
    df = df.sort_index()

    # # Reconstruct real cost as in preprocess_data
    # df["import_low_tariff_delta_Wh"] = df["active_import_low_tariff_Wh"].diff().clip(lower=0)
    # df["import_normal_tariff_delta_Wh"] = df["active_import_normal_tariff_Wh"].diff().clip(lower=0)
    # df["cost_low_tariff"] = df["import_low_tariff_delta_Wh"] * (0.20 / 1000)
    # df["cost_normal_tariff"] = df["import_normal_tariff_delta_Wh"] * (0.25 / 1000)
    # df["cost_total"] = df["cost_low_tariff"] + df["cost_normal_tariff"]
    # df = df.dropna()

    # Get the last 2 days of real data (288 points for 10min interval)
    past_points = 288
    real_past = df["total_active_import_power_W"].iloc[-past_points:]
    last_time = real_past.index[-1]

    # Future timestamps
    future_times = [last_time + timedelta(minutes=interval_minutes * (i + 1)) for i in range(len(predictions))]

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(real_past.index, real_past.values, label="Actual Past 2 Days", color="blue")
    plt.plot(future_times, predictions, label="Predicted Future", color="orange")
    plt.axvline(x=last_time, color="gray", linestyle="--", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Load (Watt)")
    plt.title("Energy Cost: Past 2 Days (10-min) + Next 24 Steps Prediction")
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig('data/output/future_predictions.png')
    plt.show()

def main():
    model = load_model('./models/lstm_energy_prediction_model.keras', compile=False)

    file_path = "./data/lora_data_12_6_25.csv"
    # For 24 steps into future at 10min/step = next 4 hours
    time_steps = 288  # 24 hours context for LSTM
    num_steps = 144    # predict next 24 steps (4 hours)

    initial_sequence, y_scaler = load_latest_data(file_path, time_steps=time_steps)

    predictions = predict_future(model, initial_sequence, y_scaler, num_steps)
    plot_with_past(file_path, predictions, time_steps, interval_minutes=10)

    print("Predicted Watts (next 24 x 10-min intervals):")
    for i, pred in enumerate(predictions):
        print(f"{i+1:02d} x 10min from last data point: {pred:.4f} W")
    print("Plot saved as 'data/output/future_predictions.png'")

if __name__ == "__main__":
    main()
