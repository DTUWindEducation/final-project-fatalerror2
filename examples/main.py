# Import Packages
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input
import tensorflow as tf
import time

# --- Start Timer for Execution ---
start_time = time.time()

# --- Ensure Reproducibility ---
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# --- User Inputs ---
variable_name = "Power"                     # Variable to plot
site_index = 2                              # Site index (1 to 4)
starting_time = "2021-01-10 06:00"          # Prediction start time
ending_time = "2021-01-11 18:00"            # Prediction end time
lookback_hours = 6                          # Number of past hours for prediction
training_window_months = 12                 # Months of data used for training

# --- Define File Paths ---
script_dir = Path(__file__).resolve().parent
inputs_dir = script_dir.parent / 'inputs'

site_files = {
    1: inputs_dir / 'Location1.csv',
    2: inputs_dir / 'Location2.csv',
    3: inputs_dir / 'Location3.csv',
    4: inputs_dir / 'Location4.csv',
}

# --- Function: Load and Filter by Site ---
def load_and_filter_by_site(site_files, site_index):
    dfs = []
    for idx, path in site_files.items():
        print(f"Trying to load: {path}")
        df = pd.read_csv(path)
        print(f"✅ Loaded: {path.name} with {len(df)} rows")
        df['Site'] = f'Location{idx}'
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    site_label = f'Location{site_index}'
    site_df = combined_df[combined_df['Site'] == site_label].copy()
    site_df['Time'] = pd.to_datetime(site_df['Time'])

    # Add time features
    site_df['hour'] = site_df['Time'].dt.hour
    site_df['hour_sin'] = np.sin(2 * np.pi * site_df['hour'] / 24)
    site_df['hour_cos'] = np.cos(2 * np.pi * site_df['hour'] / 24)

    return site_df, site_label

# --- Function: Plot Measured Data ---
def filter_and_plot(site_df, variable_name, start_time, end_time, site_label):
    filtered_df = site_df[(site_df['Time'] >= pd.to_datetime(start_time)) &
                          (site_df['Time'] <= pd.to_datetime(end_time))]

    plt.figure(figsize=(12, 6))
    plt.plot(filtered_df['Time'], filtered_df[variable_name], label=variable_name)
    plt.title(f"{variable_name} at {site_label}")
    plt.xlabel('Time')
    plt.ylabel(variable_name)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Function: Create Stacked LSTM Model ---
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Function: Plot Persistence Model ---
def plot_persistence_model(y_test, times):
    if len(y_test) < 2:
        print("⚠️ Not enough data for persistence model.")
        return

    # Build persistence predictions: next hour = current hour
    y_persistence = np.roll(y_test, 1)
    y_persistence[0] = y_test[0]

    # Compute metrics
    mse = mean_squared_error(y_test[1:], y_persistence[1:])
    mae = mean_absolute_error(y_test[1:], y_persistence[1:])
    rmse = np.sqrt(mse)

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(times, y_test, label='Measured Power', color='black', marker='^', linestyle='--')
    plt.plot(times, y_persistence, label='Persistence Model', color='blue', linestyle=':')
    plt.title(f"Persistence Model vs Measured Power\nMAE={mae:.4f}, RMSE={rmse:.4f}")
    plt.xlabel('Time')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n--- Persistence Model ---")
    print(f"MSE:  {mse:.5f}\nMAE:  {mae:.5f}\nRMSE: {rmse:.5f}")

# --- Function: Forecast and Evaluate Model ---
def plot_forecast_vs_actual(site_df, start_time, end_time, site_label, model_func, lookback_hours):
    features = [
        'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
        'windspeed_10m', 'windspeed_100m', 'winddirection_10m',
        'winddirection_100m', 'windgusts_10m', 'Power',
        'hour_sin', 'hour_cos'
    ]

    scaler = MinMaxScaler()
    site_df_scaled = site_df.copy()
    site_df_scaled[features] = scaler.fit_transform(site_df[features])

    # Define training window
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)
    earliest_train_date = start_dt - pd.DateOffset(months=training_window_months)
    train_df = site_df[(site_df['Time'] >= earliest_train_date) & (site_df['Time'] < start_dt)]

    # Prepare training data
    X_train, y_train = [], []
    for i in range(lookback_hours, len(train_df) - 1):
        window = site_df_scaled.iloc[i - lookback_hours:i][features]
        if len(window) == lookback_hours:
            X_train.append(window.values)
            y_train.append(site_df.iloc[i + 1]['Power'])

    # Prepare test data
    X_test, y_test, times = [], [], []
    for i in range(lookback_hours, len(site_df_scaled) - 1):
        current_time = site_df_scaled.iloc[i + 1]['Time']
        if start_dt <= current_time <= end_dt:
            window = site_df_scaled.iloc[i - lookback_hours + 1:i + 1][features]
            if len(window) == lookback_hours:
                X_test.append(window.values)
                y_test.append(site_df.iloc[i + 1]['Power'])
                times.append(current_time)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # Validate minimum data
    if X_train.shape[0] < 10:
        print("❌ Not enough training data. Try selecting a later start date.")
        return None, None, None, None, None
    if X_test.shape[0] < 1:
        print("❌ No valid test samples found. Check your end date.")
        return None, None, None, None, None

    # Train the model
    model = model_func(input_shape=X_train.shape[1:])
    model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=32,
        verbose=0,
        validation_split=0.1,
        shuffle=False,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
    )

    # Predict
    predictions = model.predict(X_test).flatten()

    # Compute Metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)

    # Plot Results
    plt.figure(figsize=(14, 6))
    plt.plot(times, y_test, label='Measured Power', color='black', marker='^', linestyle='--')
    plt.plot(times, predictions, label='Predicted Power (LSTM)', color='orange', linestyle='-')
    plt.title(f"Predicted vs Measured Power at {site_label}\nMAE={mae:.4f}, RMSE={rmse:.4f}")
    plt.xlabel('Time')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return predictions, y_test, mse, mae, rmse, times

# --- Main Execution ---
if __name__ == "__main__":
    print("\n🚀 Computation started")

    site_df, site_name = load_and_filter_by_site(site_files, site_index)
    filter_and_plot(site_df, variable_name, starting_time, ending_time, site_name)

    predictions, y_test, mse, mae, rmse, times = plot_forecast_vs_actual(
        site_df, starting_time, ending_time, site_name, create_lstm_model, lookback_hours
    )

    # Summary of LSTM model results
    print("\n--- LSTM Model ---")
    print(f"MSE:  {mse:.5f}")
    print(f"MAE:  {mae:.5f}")
    print(f"RMSE: {rmse:.5f}")

    # Plot persistence model for comparison
    plot_persistence_model(np.array(y_test), times)

    # End Timer
    end_time = time.time()
    total_time = end_time - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(f"\n⏱️ Total execution time: {minutes} min {seconds} sec")
