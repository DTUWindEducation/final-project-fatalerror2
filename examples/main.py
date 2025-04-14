# Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input

# --- User Inputs ---
variable_name = "Power"       # e.g., "temperature_2m", "Power", etc.
site_index = 2                # 1 to 4
starting_time = "2021-01-01"  # Included; YYYY-MM-DD
ending_time = "2021-01-02"    # Excluded (1-day prediction)
lookback_hours = 6            # History window for NN input

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
    site_to_process = f'Location{site_index}'
    site_df = combined_df[combined_df['Site'] == site_to_process].copy()
    site_df['Time'] = pd.to_datetime(site_df['Time'])
    return site_df, site_to_process

# --- Function 2: Filter by time and plot selected variable ---
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

# --- Function 3: Neural network model (MLP) ---
def create_mlp_model(input_shape):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    model.add(Dense(256, activation='relu'))  # Increased from 128
    model.add(Dense(128, activation='relu'))  # Increased from 64
    model.add(Dense(64, activation='relu'))   # Increased from 32
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Function 4: Forecast and plot prediction vs real ---
def plot_forecast_vs_actual(site_df, start_time, end_time, site_label, model_func, lookback_hours=6):
    features = ['temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
                'windspeed_10m', 'windspeed_100m', 'winddirection_10m',
                'winddirection_100m', 'windgusts_10m', 'Power']

    scaler = MinMaxScaler()
    site_df_scaled = site_df.copy()
    site_df_scaled[features] = scaler.fit_transform(site_df[features])

    X_test, y_test, times = [], [], []
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)

    for i in range(lookback_hours, len(site_df_scaled) - 1):
        current_time = site_df_scaled.iloc[i + 1]['Time']  # prediction time
        if start_dt <= current_time <= end_dt:
            input_window = site_df_scaled.iloc[i - lookback_hours + 1:i + 1][features]
            if len(input_window) == lookback_hours:
                X_test.append(input_window.values.flatten())
                y_test.append(site_df.iloc[i + 1]['Power'])
                times.append(current_time)


    train_df = site_df[site_df['Time'] < pd.to_datetime(start_time)]
    X_train, y_train = [], []
    for i in range(lookback_hours, len(train_df) - 1):
        past_window = train_df.iloc[i - lookback_hours:i][features]
        if len(past_window) == lookback_hours:
            X_train.append(past_window.values.flatten())
            y_train.append(train_df.iloc[i + 1]['Power'])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # --- Safety Checks ---
    if X_train.shape[0] < 10:
        print("❌ Not enough training data. Try selecting a later start date.")
        return None, None, None, None, None

    if X_test.shape[0] < 1:
        print("❌ No valid test samples found. Check your end date.")
        return None, None, None, None, None

    # --- Train model ---
    model = model_func(input_shape=X_train.shape[1])
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0,
              validation_split=0.1, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

    predictions = model.predict(X_test).flatten()

    # --- Metrics ---
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)

    # --- Plot ---
    plt.figure(figsize=(14, 6))
    plt.plot(times, y_test, label='Actual Power', color='black')
    plt.plot(times, predictions, label='Predicted Power', color='orange', linestyle='--')
    plt.title(f"Predicted vs Actual Power at {site_label}\nMAE={mae:.4f}, RMSE={rmse:.4f}")
    plt.xlabel('Time')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return predictions, y_test, mse, mae, rmse


# --- Main Execution ---
if __name__ == "__main__":
    site_df, site_name = load_and_filter_by_site(site_files, site_index)
    filter_and_plot(site_df, variable_name, starting_time, ending_time, site_name)
    predictions, y_test, mse, mae, rmse = plot_forecast_vs_actual(
        site_df, starting_time, ending_time, site_name, create_mlp_model, lookback_hours
    )
