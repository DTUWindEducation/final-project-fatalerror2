# Import Packages
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input
import tensorflow as tf
import time

# -- Start timer
start_time = time.time()

# -- Make training deterministic
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# --- User Inputs ---
variable_name = "Power"
site_index = 2
starting_time = "2021-01-01"
ending_time = "2021-01-02"
lookback_hours = 24

# --- File Paths ---
script_dir = Path(__file__).resolve().parent
inputs_dir = script_dir.parent / 'inputs'

site_files = {
    1: inputs_dir / 'Location1.csv',
    2: inputs_dir / 'Location2.csv',
    3: inputs_dir / 'Location3.csv',
    4: inputs_dir / 'Location4.csv',
}

# --- Function 1: Load and filter by site ---
def load_and_filter_by_site(site_files, site_index):
    dfs = []
    for idx, path in site_files.items():
        print(f"Trying to load: {path}")
        df = pd.read_csv(path)
        df['Site'] = f'Location{idx}'
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    site_df = combined_df[combined_df['Site'] == f'Location{site_index}'].copy()
    site_df['Time'] = pd.to_datetime(site_df['Time'])

    # ⏰ Add time-based features
    site_df['hour'] = site_df['Time'].dt.hour
    site_df['hour_sin'] = np.sin(2 * np.pi * site_df['hour'] / 24)
    site_df['hour_cos'] = np.cos(2 * np.pi * site_df['hour'] / 24)
    site_df['Power_lag1'] = site_df['Power'].shift(1)
    site_df.dropna(inplace=True)

    return site_df, f'Location{site_index}'

# --- Function 2: Filter and plot any variable ---
def filter_and_plot(site_df, variable_name, start_time, end_time, site_label):
    filtered_df = site_df[
        (site_df['Time'] >= pd.to_datetime(start_time)) &
        (site_df['Time'] <= pd.to_datetime(end_time))
    ]
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_df['Time'], filtered_df[variable_name], label=variable_name)
    plt.title(f"{variable_name} at {site_label}")
    plt.xlabel('Time')
    plt.ylabel(variable_name)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Function 3: Neural network (MLP) ---
def create_mlp_model(input_shape):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Function 4: Forecast and evaluate ---
def plot_forecast_vs_actual(site_df, start_time, end_time, site_label, model_func, lookback_hours):
    features = [
        'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
        'windspeed_10m', 'windspeed_100m', 'winddirection_10m',
        'winddirection_100m', 'windgusts_10m', 'Power',
        'hour_sin', 'hour_cos', 'Power_lag1'
    ]

    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)

    # Restrict training to the last 6 months only
    train_df = site_df[(site_df['Time'] >= '2020-07-01') & (site_df['Time'] < start_dt)].copy()
    test_df = site_df[(site_df['Time'] >= start_dt) & (site_df['Time'] <= end_dt)].copy()

    # Scale based on training data only
    scaler = MinMaxScaler()
    scaler.fit(train_df[features])
    site_df_scaled = site_df.copy()
    site_df_scaled[features] = scaler.transform(site_df[features])

    # Prepare test data
    X_test, y_test, times = [], [], []
    for i in range(lookback_hours, len(site_df_scaled) - 1):
        current_time = site_df_scaled.iloc[i + 1]['Time']
        if start_dt <= current_time <= end_dt:
            window = site_df_scaled.iloc[i - lookback_hours + 1:i + 1][features]
            if len(window) == lookback_hours:
                X_test.append(window.values.flatten())
                y_test.append(site_df.iloc[i + 1]['Power'])
                times.append(current_time)

    # Prepare training data
    X_train, y_train = [], []
    for i in range(lookback_hours, len(train_df) - 1):
        window = train_df.iloc[i - lookback_hours:i][features]
        if len(window) == lookback_hours:
            X_train.append(window.values.flatten())
            y_train.append(train_df.iloc[i + 1]['Power'])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    if X_train.shape[0] < 10 or X_test.shape[0] < 1:
        print("\u274c Not enough data.")
        return None, None, None, None, None

    # Train model
    model = model_func(input_shape=X_train.shape[1])
    model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=0,
              validation_split=0.1, shuffle=False,
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

    predictions = model.predict(X_test).flatten()

    # Metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)

    # Persistence baseline
    persistence_preds = np.roll(y_test, 1)
    persistence_preds[0] = y_test[0]

    # Plot all three series
    plt.figure(figsize=(14, 6))
    plt.plot(times, y_test, label='Actual Power', color='black', marker='^', linestyle='--')
    plt.plot(times, predictions, label='Predicted Power (NN)', color='orange', linestyle='--')
    plt.plot(times, persistence_preds, label='Persistence Model', color='green', marker='o', linestyle=':')
    plt.title(f"Predicted vs Actual Power at {site_label}")
    plt.xlabel('Time')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return predictions, y_test, mse, mae, rmse

# --- Main Execution ---
if __name__ == "__main__":
    print("Computation started")
    site_df, site_name = load_and_filter_by_site(site_files, site_index)
    filter_and_plot(site_df, variable_name, starting_time, ending_time, site_name)
    predictions, y_test, mse, mae, rmse = plot_forecast_vs_actual(
        site_df, starting_time, ending_time, site_name, create_mlp_model, lookback_hours)

    print("\n--- Neural Network Model ---")
    print(f"Mean Squared Error (MSE):  {mse:.5f}")
    print(f"Mean Absolute Error (MAE): {mae:.5f}")
    print(f"Root Mean Square Error (RMSE):    {rmse:.5f}")

    end_time = time.time()
    minutes = int((end_time - start_time) // 60)
    seconds = int((end_time - start_time) % 60)
    print(f"\n\u23f1\ufe0f Total execution time: {minutes} min {seconds} sec")



