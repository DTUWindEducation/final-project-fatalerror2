import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import joblib
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input
from tensorflow.keras.models import load_model

# --- Reproducibility ---
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)





# --- SVM Utilities ---
def load_site_data(site_index):
    folder_path = Path(__file__).parents[1]
    site_path = folder_path / f"inputs/Location{site_index}.csv"
    df = pd.read_csv(site_path)
    df['Time'] = pd.to_datetime(df['Time'])
    return df, site_path

def prepare_features(df, num_lags=2):
    df_copy = df.copy()
    features = [
        'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
        'windspeed_10m', 'windspeed_100m',
        'winddirection_10m', 'winddirection_100m',
        'windgusts_10m'
    ]

    for lag in range(1, num_lags + 1):
        df_copy[f'Power_t-{lag}'] = df_copy['Power'].shift(lag)
        features.append(f'Power_t-{lag}')

    df_copy['Power+1'] = df_copy['Power'].shift(-1)
    df_copy.dropna(inplace=True)
    X = df_copy[features].values
    y = df_copy['Power+1'].values
    return X, y, features

def train_and_save_svm(site_index, num_lags=2):
    folder_path = Path(__file__).parents[1]
    model_path = folder_path / f"outputs/Location{site_index}_svr_model.pkl"
    scaler_path = folder_path / f"outputs/Location{site_index}_scaler.pkl"

    if model_path.exists() and scaler_path.exists():
        print("✅ Loaded existing SVM model and scaler.")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        df, _ = load_site_data(site_index)
        X, y, _ = prepare_features(df, num_lags=num_lags)
        split_index = int(len(X) * 0.8)
        X_test = scaler.transform(X[split_index:])
        y_test = y[split_index:]
        y_pred = model.predict(X_test)

    else:
        print("🛠️ Training new SVM model...")
        df, _ = load_site_data(site_index)
        X, y, _ = prepare_features(df, num_lags=num_lags)

        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = SVR(kernel='rbf')
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        print("💾 SVM model and scaler saved.")

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return mae, mse, rmse

def plot_prediction_vs_actual(site_index, site_name, start_time, end_time, num_lags=2):
    df, _ = load_site_data(site_index)
    for lag in range(1, num_lags + 1):
        df[f'Power_t-{lag}'] = df['Power'].shift(lag)

    mask = (df['Time'] >= pd.to_datetime(start_time)) & (df['Time'] <= pd.to_datetime(end_time))
    subset = df.loc[mask].copy()
    subset.dropna(inplace=True)

    features = [
        'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
        'windspeed_10m', 'windspeed_100m',
        'winddirection_10m', 'winddirection_100m',
        'windgusts_10m'
    ] + [f'Power_t-{lag}' for lag in range(1, num_lags + 1)]

    folder_path = Path(__file__).parents[1]
    model = joblib.load(folder_path / f"outputs/Location{site_index}_svr_model.pkl")
    scaler = joblib.load(folder_path / f"outputs/Location{site_index}_scaler.pkl")

    X_subset = subset[features].values
    X_scaled = scaler.transform(X_subset)
    subset['Predicted_Power'] = model.predict(X_scaled)

    plt.figure(figsize=(12, 5))
    plt.plot(subset['Time'], subset['Power'], 'k^-', label="Measured Power")
    plt.plot(subset['Time'], subset['Predicted_Power'], 'b.-', label="Predicted Power (SVM)")
    plt.title(f"{site_name} - Measured vs Predicted Power (SVM)")
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return subset[['Time', 'Power', 'Predicted_Power']]

# --- LSTM Utilities ---
def load_and_filter_by_site(inputs_dir, site_index):
    path = inputs_dir / f'Location{site_index}.csv'
    df = pd.read_csv(path)
    df['Site'] = f'Location{site_index}'
    df['Time'] = pd.to_datetime(df['Time'])
    df['hour'] = df['Time'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df, f'Location{site_index}'

def filter_and_plot(df, variable, site_index, start, end, site_name):
    filtered = df[(df['Time'] >= pd.to_datetime(start)) & (df['Time'] <= pd.to_datetime(end))]
    plt.figure(figsize=(12, 6))
    plt.plot(filtered['Time'], filtered[variable], color='black', linestyle='-', marker='^', label=variable)
    plt.title(f"Location {site_index} - {variable}")
    plt.xlabel('Time')
    plt.ylabel(variable)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def create_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    return model

def plot_forecast_vs_actual(df, start, end, site_name, model_func, lookback):
    features = [
        'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
        'windspeed_10m', 'windspeed_100m', 'winddirection_10m',
        'winddirection_100m', 'windgusts_10m', 'Power',
        'hour_sin', 'hour_cos'
    ]

    start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)

    folder_path = Path(__file__).parents[1] / "outputs"
    model_path = folder_path / f"{site_name}_lstm_model.h5"
    scaler_path = folder_path / f"{site_name}_lstm_scaler.pkl"
    folder_path.mkdir(parents=True, exist_ok=True)

    # Load or fit scaler
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        print("✅ Loaded existing scaler.")
    else:
        print("🛠️ Training new scaler...")
        scaler = MinMaxScaler()
        scaler.fit(df[features])
        joblib.dump(scaler, scaler_path)
        print("💾 Scaler saved.")

    df_scaled = df.copy()
    df_scaled[features] = scaler.transform(df[features])

    # Training based on 80% split
    split_index = int(len(df_scaled) * 0.8)
    train_df = df_scaled.iloc[:split_index]
    raw_train_df = df.iloc[:split_index]

    if not model_path.exists():
        print("🛠️ Training new LSTM model...")
        X_train, y_train = [], []
        for i in range(lookback, len(train_df) - 1):
            window = train_df.iloc[i - lookback:i][features]
            X_train.append(window.values)
            y_train.append(raw_train_df.iloc[i + 1]['Power'])

        X_train, y_train = np.array(X_train), np.array(y_train)

        if len(X_train) < 10:
            print("❌ Not enough training data.")
            return None, None, None, None, None, None

        model = model_func(X_train[0].shape)
        model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0,
                  validation_split=0.1, shuffle=False,
                  callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

        model.save(model_path)
        print("💾 Model trained and saved.")
    else:
        model = load_model(model_path)
        print("✅ Loaded existing LSTM model.")

    # Prepare test data (full df, filtered by date range)
    X_test, y_test, times = [], [], []
    for i in range(lookback, len(df_scaled) - 1):
        current_time = df_scaled.iloc[i + 1]['Time']
        if start_dt <= current_time <= end_dt:
            window = df_scaled.iloc[i - lookback + 1:i + 1][features]
            X_test.append(window.values)
            y_test.append(df.iloc[i + 1]['Power'])
            times.append(current_time)

    if len(X_test) < 1:
        print("❌ Not enough test data.")
        return None, None, None, None, None, None

    X_test, y_test = np.array(X_test), np.array(y_test)
    predictions = model.predict(X_test).flatten()

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)

    plt.figure(figsize=(14, 6))
    plt.plot(times, y_test, 'k^-', label='Measured Power')
    plt.plot(times, predictions, 'b.-', label='Predicted Power (LSTM)')
    plt.title(f"{site_name} - Predicted vs Measured Power (Neural Networks)")
    plt.xlabel('Time')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return predictions, y_test, mse, mae, rmse, times

# --- Persistance Utilities ---

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
    plt.plot(times, y_test, label='Measured Power', color='black', marker='^', linestyle='-')
    plt.plot(times, y_persistence, label='Persistence Model', color='blue', marker='o', linestyle='-')
    plt.title("Persistence Model vs Measured Power")
    plt.xlabel('Time')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return mae, mse, rmse



# --- Metric Utilities ---
def print_evaluation_metrics(mae, mse, rmse):
    print(f"mean squared error (MSE):  {mse:.5f}")
    print(f"mean absolute error (MAE):  {mae:.5f}")
    print(f"root mean square error (RMSE): {rmse:.5f}")