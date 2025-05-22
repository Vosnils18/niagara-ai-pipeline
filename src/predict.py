import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocess import preprocess_data, create_sequences
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def load_latest_data(file_path, num_hours=24):
    """Load and preprocess the latest data for prediction."""
    _, _, _, _, scaler = preprocess_data(file_path)
    df = pd.read_json(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.sort_index().last('7D')  # Use the last 7 days of data
    
    # Preprocess the latest data
    df = preprocess_data(df)
    features = ['sin_hour', 'cos_hour', 'sin_dayofweek', 'cos_dayofweek', 'sin_month', 'cos_month',
                'is_weekend', 'day_of_year', 'prev_day_consumption', 'rolling_7d_avg',
                'hour_weekend', 'prev_week_consumption']
    X = df[features].values
    X_scaled = scaler.transform(X)
    X_seq = create_sequences(X_scaled, np.zeros((X_scaled.shape[0], 1)), time_steps=24)[0]
    return X_seq[-1:], scaler, df.index[-1]

def predict_future(model, initial_sequence, scaler, num_hours):
    """Predict future energy consumption."""
    predictions = []
    current_sequence = initial_sequence.copy()

    for _ in range(num_hours):
        # Predict the next hour
        next_pred = model.predict(current_sequence)
        predictions.append(next_pred[0, 0])

        # Update the sequence for the next prediction
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, :] = next_pred

    # Inverse transform the predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

def plot_predictions(predictions, start_time, num_hours):
    """Plot the predicted energy consumption."""
    timestamps = [start_time + timedelta(hours=i) for i in range(1, num_hours + 1)]
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, predictions, label='Predicted')
    plt.title('Predicted Energy Consumption')
    plt.xlabel('Time')
    plt.ylabel('Energy (kWh)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('future_predictions.png')
    plt.close()

def main():
    # Load the trained model
    model = load_model('../models/lstm_energy_prediction_model.h5')

    # Load and preprocess the latest data
    file_path = "../data/sample_niagara_data_realistic.json"
    initial_sequence, scaler, last_timestamp = load_latest_data(file_path)

    # Predict for the next 24 hours (you can change this to predict for more hours or days)
    num_hours = 24
    predictions = predict_future(model, initial_sequence, scaler, num_hours)

    # Plot and save the predictions
    plot_predictions(predictions, last_timestamp, num_hours)

    # Print the predictions
    for i, pred in enumerate(predictions):
        timestamp = last_timestamp + timedelta(hours=i+1)
        print(f"{timestamp}: {pred:.2f} kWh")

    print(f"Predictions saved as '../data/output/future_predictions.png'")

if __name__ == "__main__":
    main()