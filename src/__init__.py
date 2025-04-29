import pandas as pd
import numpy as np
from pathlib import Path
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Input

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
    X = df_copy[features].values
    y = df_copy['Power+1'].values
    return X, y, features


def create_lstm_model(input_shape):
    """Create LSTM model architecture."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    return model

