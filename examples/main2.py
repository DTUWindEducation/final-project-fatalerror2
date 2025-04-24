# --- Import Packages ---
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

# --- Config ---
faster_mode = False
seed = 42

# --- Reproducibility ---
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# --- User Inputs ---
variable_name = "Power"
site_index = 2
starting_time = "2021-06-15"
ending_time = "2021-06-16"
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

# --- Load and Preprocess ---
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

# --- Plot Time Series ---
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

# --- LSTM Model ---
def create_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Forecast Power Hourly ---
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

    model = model_func(X_train[0].shape)
    model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0,
              validation_split=0.1, shuffle=False,
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

    predictions = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)

    plt.figure(figsize=(14, 6))
    plt.plot(times, y_test, label='Measured Power', color='black', linestyle='--')
    plt.plot(times, predictions, label='Predicted Power (LSTM)', color='orange')
    plt.title(f"Predicted vs Measured Power at {site_label}\nMAE={mae:.4f}, RMSE={rmse:.4f}")
    plt.xlabel('Time')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return predictions, y_test, mse, mae, rmse, times

# --- Prepare Daily Data for CF ---
def prepare_lstm_daily_cf_data(df, features, start_date, end_date):
    X, y, times = [], [], []
    date_range = pd.date_range(start=start_date, end=end_date - pd.Timedelta(hours=23), freq='D')

    for day_start in date_range:
        day_end = day_start + pd.Timedelta(hours=23)
        day_df = df[(df['Time'] >= day_start) & (df['Time'] <= day_end)]
        if len(day_df) == 24:
            X.append(day_df[features].values)
            y.append(day_df['Power'].mean())
            times.append(day_start)

    return np.array(X), np.array(y), times

# --- LSTM Model for CF ---
def create_lstm_cf_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()
    print("\n🚀 Computation started")

    site_df, site_name = load_and_filter_by_site(site_files, site_index)
    filter_and_plot(site_df, variable_name, starting_time, ending_time, site_name)

    if not faster_mode:
        predictions, y_test, mse, mae, rmse, times = plot_forecast_vs_actual(
            site_df, starting_time, ending_time, site_name, create_lstm_model, lookback_hours
        )
        print("\n--- LSTM Model ---")
        print(f"MSE:  {mse:.5f}")
        print(f"MAE:  {mae:.5f}")
        print(f"RMSE: {rmse:.5f}")

    # --- Predict Capacity Factor ---
    features = [
        'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
        'windspeed_10m', 'windspeed_100m', 'winddirection_10m',
        'winddirection_100m', 'windgusts_10m', 'hour_sin', 'hour_cos'
    ]
    train_start = pd.to_datetime(starting_time) - pd.DateOffset(months=training_window_months)
    train_end = pd.to_datetime(starting_time) - pd.Timedelta(hours=1)
    test_start = pd.to_datetime(starting_time)
    test_end = pd.to_datetime(starting_time) + pd.Timedelta(hours=23)

    X_train, y_train, _ = prepare_lstm_daily_cf_data(site_df, features, train_start, train_end)
    X_test, y_test, _ = prepare_lstm_daily_cf_data(site_df, features, test_start, test_end)

    if len(X_train) > 10 and len(X_test) == 1:
        model = create_lstm_cf_model(X_train[0].shape)
        model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0,
                  validation_split=0.1,
                  callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

        predicted_cf = model.predict(X_test)[0][0]
        actual_cf = y_test[0]

        print("\n--- Capacity Factor (LSTM Daily Model) ---")
        print(f"Actual Capacity Factor:    {actual_cf:.4f}")
        print(f"Predicted Capacity Factor: {predicted_cf:.4f}")
        print(f"Percentual Error:          {abs(actual_cf - predicted_cf) / actual_cf * 100:.2f}%")
    else:
        print("⚠️ Not enough data for LSTM CF prediction.")

    # --- Execution Time ---
    total_time = time.time() - start_time
    print(f"\n⏱️ Total execution time: {int(total_time // 60)} min {int(total_time % 60)} sec")
