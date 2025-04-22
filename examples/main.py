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

faster_mode = True

# --- Start Timer for Execution ---
start_time = time.time()

# --- Ensure Reproducibility ---
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# --- User Inputs ---
variable_name = "Power"
site_index = 2
starting_time = "2021-01-10"
ending_time = "2021-01-11"
lookback_hours = 18
training_window_months = 6

# --- File Paths ---
script_dir = Path(__file__).resolve().parent
inputs_dir = script_dir.parent / 'inputs'
site_files = {
    1: inputs_dir / 'Location1.csv',
    2: inputs_dir / 'Location2.csv',
    3: inputs_dir / 'Location3.csv',
    4: inputs_dir / 'Location4.csv',
}

# --- Load and Filter by Site ---
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
    site_df['hour'] = site_df['Time'].dt.hour
    site_df['hour_sin'] = np.sin(2 * np.pi * site_df['hour'] / 24)
    site_df['hour_cos'] = np.cos(2 * np.pi * site_df['hour'] / 24)
    return site_df, site_label

# --- Plot a Variable Over Time ---
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

# --- Define LSTM Model Architecture ---
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Persistence Baseline Model ---
def plot_persistence_model(y_test, times):
    if len(y_test) < 2:
        print("⚠️ Not enough data for persistence model.")
        return
    y_persistence = np.roll(y_test, 1)
    y_persistence[0] = y_test[0]
    mse = mean_squared_error(y_test[1:], y_persistence[1:])
    mae = mean_absolute_error(y_test[1:], y_persistence[1:])
    rmse = np.sqrt(mse)
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

# --- LSTM Forecast Using Full Known Features ---
def plot_forecast_vs_actual(site_df, start_time, end_time, site_label, model_func, lookback_hours):
    features = [
        'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
        'windspeed_10m', 'windspeed_100m', 'winddirection_10m',
        'winddirection_100m', 'windgusts_10m', 'Power',
        'hour_sin', 'hour_cos']

    scaler = MinMaxScaler()
    site_df_scaled = site_df.copy()
    site_df_scaled[features] = scaler.fit_transform(site_df[features])
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)
    earliest_train_date = start_dt - pd.DateOffset(months=training_window_months)
    train_df = site_df[(site_df['Time'] >= earliest_train_date) & (site_df['Time'] < start_dt)]

    X_train, y_train = [], []
    for i in range(lookback_hours, len(train_df) - 1):
        window = site_df_scaled.iloc[i - lookback_hours:i][features]
        X_train.append(window.values)
        y_train.append(site_df.iloc[i + 1]['Power'])

    X_test, y_test, times = [], [], []
    for i in range(lookback_hours, len(site_df_scaled) - 1):
        current_time = site_df_scaled.iloc[i + 1]['Time']
        if start_dt <= current_time <= end_dt:
            window = site_df_scaled.iloc[i - lookback_hours + 1:i + 1][features]
            X_test.append(window.values)
            y_test.append(site_df.iloc[i + 1]['Power'])
            times.append(current_time)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    if len(X_train) < 10 or len(X_test) < 1:
        print("❌ Not enough data.")
        return None, None, None, None, None, None

    model = model_func(input_shape=X_train[0].shape)
    model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0,
              validation_split=0.1, shuffle=False,
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

    predictions = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)

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

# --- 24-Hour Forecast (No Future Weather) ---
def forecast_next_24h_no_future_weather(site_df, start_time, site_label, model_func, lookback_hours):
    print("\n🔮 Forecasting next 24 hours using only past data (no future weather)...")

    features = [
        'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
        'windspeed_10m', 'windspeed_100m', 'winddirection_10m',
        'winddirection_100m', 'windgusts_10m', 'Power',
        'hour_sin', 'hour_cos'
    ]

    # Normalize features, but use a separate scaler for 'Power' only
    scaler_features = MinMaxScaler()
    scaler_power = MinMaxScaler()

    site_df_scaled = site_df.copy()
    site_df_scaled[features] = scaler_features.fit_transform(site_df[features])
    site_df_scaled['Power'] = scaler_power.fit_transform(site_df[['Power']])

    # Forecast from the provided starting time
    forecast_start = pd.to_datetime(start_time)
    forecast_end = forecast_start + pd.Timedelta(hours=23)
    print(f"📅 Forecasting from {forecast_start} to {forecast_end}")

    # Extract training data before forecast start
    train_end = forecast_start
    train_start = train_end - pd.DateOffset(months=training_window_months)
    train_mask = (site_df_scaled['Time'] >= train_start) & (site_df_scaled['Time'] < train_end)
    train_df_scaled = site_df_scaled[train_mask].reset_index(drop=True)
    train_df_original = site_df[train_mask].reset_index(drop=True)

    print(f"📅 Training data window: {train_start} to {train_end}")
    print(f"🧪 Total training samples in train_df_scaled: {len(train_df_scaled)}")
    print(f"🔁 Required minimum samples: {lookback_hours + 1}")

    X_train, y_train = [], []
    for i in range(lookback_hours, len(train_df_scaled) - 1):
        window = train_df_scaled.iloc[i - lookback_hours:i][features]
        X_train.append(window.values)
        y_train.append(train_df_scaled.iloc[i + 1]['Power'])

    if len(X_train) == 0:
        print("❌ No training samples found. Aborting forecast.")
        return

    model = model_func(input_shape=X_train[0].shape)
    model.fit(np.array(X_train), np.array(y_train),
              epochs=150, batch_size=32, verbose=0,
              validation_split=0.1, shuffle=False,
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

    # Extract initial sequence
    sequence_df = site_df_scaled[site_df_scaled['Time'] < forecast_start].tail(lookback_hours)
    if len(sequence_df) < lookback_hours:
        print(f"❌ Extracted sequence too short: expected {lookback_hours}, got {len(sequence_df)}.")
        return
    sequence = sequence_df[features].values.copy()

    # Get actual power series (same as in main plots)
    actual_series_mask = (site_df['Time'] >= forecast_start) & (site_df['Time'] <= forecast_end)
    actual_series = site_df[actual_series_mask].copy()

    # Resample to hourly data, taking the mean of numeric columns only
    actual_series.set_index('Time', inplace=True)
    actual_series = actual_series.resample('H').mean(numeric_only=True).reset_index()

    # Reindex for lookup
    actual_series.set_index('Time', inplace=True)

    predicted_power, actual_power, future_times = [], [], []

    for i in range(24):
        input_seq = np.expand_dims(sequence[-lookback_hours:], axis=0)
        pred_scaled = model.predict(input_seq, verbose=0).flatten()[0]
        
        # Inverse scale the predicted power using the Power-only scaler
        real_power = scaler_power.inverse_transform([[pred_scaled]])[0][0]
        predicted_power.append(real_power)

        forecast_time = forecast_start + pd.Timedelta(hours=i)
        future_times.append(forecast_time)

        if forecast_time in actual_series.index:
            actual_power.append(actual_series.loc[forecast_time]['Power'])
        else:
            # Find the nearest timestamp if exact match fails
            nearest_time = actual_series.index[np.argmin(np.abs(actual_series.index - forecast_time))]
            actual_power.append(actual_series.loc[nearest_time]['Power'])

    # Plot aligned forecast
    plt.figure(figsize=(14, 6))
    plt.plot(future_times, predicted_power, label='Forecasted Power', linestyle='-', color='orange')
    plt.plot(future_times, actual_power, label='Measured Power', linestyle='--', color='black', marker='^')
    plt.title(f"24h Forecast vs Measured Power\n{forecast_start.strftime('%Y-%m-%d %H:%M')} to {forecast_end.strftime('%Y-%m-%d %H:%M')} at {site_label}")
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Estimate capacity factor
    capacity_factor = np.nanmean(predicted_power)
    print(f"\n📊 Estimated Capacity Factor (next 24h): {capacity_factor:.2%}")






# --- Main Execution ---
if __name__ == "__main__":
    print("\n🚀 Computation started")
    site_df, site_name = load_and_filter_by_site(site_files, site_index)
    filter_and_plot(site_df, variable_name, starting_time, ending_time, site_name)

    if not faster_mode :    
    
        predictions, y_test, mse, mae, rmse, times = plot_forecast_vs_actual(
            site_df, starting_time, ending_time, site_name, create_lstm_model, lookback_hours)
    
        print("\n--- LSTM Model ---")
        print(f"MSE:  {mse:.5f}")
        print(f"MAE:  {mae:.5f}")
        print(f"RMSE: {rmse:.5f}")
    
        plot_persistence_model(np.array(y_test), times)
    
    forecast_next_24h_no_future_weather(site_df, starting_time, site_name, create_lstm_model, lookback_hours)


    end_time = time.time()
    total_time = end_time - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(f"\n⏱️ Total execution time: {minutes} min {seconds} sec")
