"""Supplimentary functions"""
import random
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# =============================================
# REPRODUCIBILITY SETUP
# =============================================
# Set all random seeds for consistent results across runs
random.seed(42)        # Python random module
np.random.seed(42)     # NumPy random generator
tf.random.set_seed(42) # TensorFlow random generator

# =============================================
# DATA LOADING AND PREPROCESSING UTILITIES
# =============================================
def load_site_data(site_index):
    """Load data for a specific site and convert time column to datetime."""
    folder_path = Path(__file__).parents[1]
    site_path = folder_path / f"inputs/Location{site_index}.csv"
    df = pd.read_csv(site_path)
    df['Time'] = pd.to_datetime(df['Time'])
    return df

def load_and_filter_by_site(inputs_dir, site_index):
    """Load site data and add time-related features (sine/cosine of hour)."""
    path = inputs_dir / f'Location{site_index}.csv'
    df = pd.read_csv(path)
    df['Site'] = f'Location{site_index}'
    df['Time'] = pd.to_datetime(df['Time'])
    df['hour'] = df['Time'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df

def determine_winner(models_metrics):
    """
    models_metrics: dict
        Format -> {'ModelName': {'MAE': float, 'MSE': float, 'RMSE': float}}
    """

    # Step 1: Check if one model is best in all three metrics
    best_mae = min(models_metrics, key=lambda x: models_metrics[x]['MAE'])
    best_mse = min(models_metrics, key=lambda x: models_metrics[x]['MSE'])
    best_rmse = min(models_metrics, key=lambda x: models_metrics[x]['RMSE'])

    if best_mae == best_mse == best_rmse:
        print(f"\n🏆 Winner: {best_mae} (best in all metrics)\n")
        return best_mae

    # Step 2: Otherwise, pick by lowest RMSE
    rmse_values = {model: metrics['RMSE'] for model, metrics in models_metrics.items()}
    sorted_rmse = sorted(rmse_values.items(), key=lambda item: item[1])

    first, second = sorted_rmse[0], sorted_rmse[1]
    diff_percentage = abs(first[1] - second[1]) / first[1]

    if diff_percentage < 0.01:  # less than 1% difference
        # Tie-breaker: Use MAE
        mae_values = {model: metrics['MAE'] for model, metrics in models_metrics.items()}
        winner = min(mae_values, key=mae_values.get)
        print(f"\n🏆 Winner (tie-breaker by MAE): {winner}\n")
        return winner
    else:
        winner = first[0]
        print(f"\n🏆 Winner (by lowest RMSE): {winner}\n")
        return winner

# =============================================
# SVM MODEL UTILITIES
# =============================================
def prepare_features(df, num_lags=2):
    """
    Prepare features for SVM model by adding lagged power values.
    Returns:
        X: Feature matrix
        y: Target values (power at t+1)
        features: List of feature names
    """
    df_copy = df.copy()
    features = [
        'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
        'windspeed_10m', 'windspeed_100m',
        'winddirection_10m', 'winddirection_100m',
        'windgusts_10m'
    ]

    # Add lagged power features
    for lag in range(1, num_lags + 1):
        df_copy[f'Power_t-{lag}'] = df_copy['Power'].shift(lag)
        features.append(f'Power_t-{lag}')

    # Create target variable (power at next time step)
    df_copy['Power+1'] = df_copy['Power'].shift(-1)
    df_copy.dropna(inplace=True)
    x = df_copy[features].values
    y = df_copy['Power+1'].values
    return x, y, features


def create_lstm_model(input_shape):
    """Create LSTM model architecture."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=MeanSquaredError())
    return model


