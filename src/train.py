import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from preprocessing import preprocess_data

def create_model(input_shape):
    """Create a dual-layer LSTM model with dropout."""
    model = Sequential([
        LSTM(128, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.05),
        LSTM(64, activation='relu'),
        Dense(16),
        Dense(1)
    ])
    return model

def train_model(X_train, y_train, X_test, y_test, epochs=200, batch_size=8):
    """Train the LSTM model."""
    model = create_model((X_train.shape[1], X_train.shape[2]))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('models/best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate the trained model."""
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test)
    
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print("LSTM MSE:", mse)
    print("LSTM R²:", r2)
    
    return y_true, y_pred

def plot_results(y_true, y_pred, history):
    """Plot the prediction results and training history."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title('LSTM: Actual vs Predicted Energy Consumption')
    plt.xlabel('Sample')
    plt.ylabel('Energy (kWh)')
    plt.legend()
    plt.savefig('prediction_results.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Load and preprocess data
    file_path = "./data/lora_data_5_6_25.csv"
    X_train, X_test, y_train, y_test, x_scaler, y_scaler = preprocess_data(file_path)
    print("X_train shape (should be N, 144, 24):", X_train.shape)
    
    # Train model
    model, history = train_model(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    y_true, y_pred = evaluate_model(model, X_test, y_test, y_scaler)
    
    # Plot results
    plot_results(y_true, y_pred, history)
    
    # Save the model
    model.save('./models/lstm_energy_prediction_model.keras')
    print("Model saved as './models/lstm_energy_prediction_model.keras'")

if __name__ == "__main__":
    main()