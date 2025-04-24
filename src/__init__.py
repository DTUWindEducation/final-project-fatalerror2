import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import joblib


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
    df, _ = load_site_data(site_index)
    X, y, _ = prepare_features(df, num_lags=num_lags)

    # Chronological split (no shuffle)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svr = SVR(kernel='rbf')
    svr.fit(X_train_scaled, y_train)

    y_pred = svr.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    folder_path = Path(__file__).parents[1]
    joblib.dump(svr, folder_path / f"outputs/Location{site_index}_svr_model.pkl")
    joblib.dump(scaler, folder_path / f"outputs/Location{site_index}_scaler.pkl")

    return mae, mse, rmse


def plot_prediction_vs_actual(site_index, start_time, end_time, num_lags=2):
    df, _ = load_site_data(site_index)

    # Create lag features manually for the full dataset
    for lag in range(1, num_lags + 1):
        df[f'Power_t-{lag}'] = df['Power'].shift(lag)

    # Filter and drop missing values
    mask = (df['Time'] >= pd.to_datetime(start_time)) & (df['Time'] <= pd.to_datetime(end_time))
    subset = df.loc[mask].copy()
    subset.dropna(inplace=True)

    # Define features
    features = [
        'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
        'windspeed_10m', 'windspeed_100m',
        'winddirection_10m', 'winddirection_100m',
        'windgusts_10m'
    ] + [f'Power_t-{lag}' for lag in range(1, num_lags + 1)]

    # Load model and scaler
    folder_path = Path(__file__).parents[1]
    model = joblib.load(folder_path / f"outputs/Location{site_index}_svr_model.pkl")
    scaler = joblib.load(folder_path / f"outputs/Location{site_index}_scaler.pkl")

    # Prepare features and predict
    X_subset = subset[features].values
    X_scaled = scaler.transform(X_subset)
    subset['Predicted_Power'] = model.predict(X_scaled)

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(subset['Time'], subset['Power'], label="Actual Power")
    plt.plot(subset['Time'], subset['Predicted_Power'], label="Predicted Power (SVM)")
    plt.title(f"Site {site_index} - Predicted vs Actual Power with SVM")
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return subset[['Time', 'Power', 'Predicted_Power']]