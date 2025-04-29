"""
Check functions works as expected.
"""
""""""
import os
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


#test the function load_site_data using pytest
from src.load_data import load_site_data


def test_load_site_data():
    """Test the load_site_data function."""
    # Given
    site_index = 1
    power_exp = 0.1635  # Expected value of power

    # When
    df, _ = load_site_data(site_index)

    # Then
    assert np.isclose(df['Power'][0], power_exp), "Power value does not match the expected value."


#test the function train_and_save_svm using pytest
from src import train_and_save_svm


def test_train_and_save_svm():
    """Test the train_and_save_svm function."""
    # Given
    site_index = 1
    model_path = Path("models/svm_model_1.pkl")

    # When
    train_and_save_svm(site_index)

    # Then
    assert model_path.is_file(), "Model file was not created."

    # Clean up
    model_path.unlink()


#test the function create_lstm_model from the __init__ file in src folder using pytest 

from src import create_lstm_model

def test_create_lstm_model():
    """Test the create_lstm_model function."""
    # Given
    input_shape = (10, 1)  # Example input shape

    # When
    model = create_lstm_model(input_shape)

    # Then
    assert model is not None, "Model was not created."

    # Check if the model has the expected layers
    assert len(model.layers) > 0, "Model has no layers."

    # Check if the first layer is an LSTM layer
    assert 'LSTM' in model.layers[0].name, "First layer is not an LSTM layer."


from src import plot_forecast_vs_actual

def test_plot_forecast_vs_actual():
    """Test the plot_forecast_vs_actual function with mocking."""
    # Given
    df = pd.DataFrame({
        'Time': pd.date_range(start="2017-01-19", periods=100, freq='H'),
        'temperature_2m': np.random.rand(100),
        'relativehumidity_2m': np.random.rand(100),
        'dewpoint_2m': np.random.rand(100),
        'windspeed_10m': np.random.rand(100),
        'windspeed_100m': np.random.rand(100),
        'winddirection_10m': np.random.rand(100),
        'winddirection_100m': np.random.rand(100),
        'windgusts_10m': np.random.rand(100),
        'Power': np.random.rand(100),
        'hour_sin': np.sin(np.linspace(0, 2 * np.pi, 100)),
        'hour_cos': np.cos(np.linspace(0, 2 * np.pi, 100)),
    })
    start_time = "2017-01-20"
    end_time = "2017-01-21"
    site_index = 1
    lookback_hours = 18

    # Mock the model function to avoid actual training
    mock_model_func = MagicMock(return_value=create_lstm_model((lookback_hours, len(df.columns) - 1)))

    # Mock file I/O operations (e.g., saving/loading models and scalers)
    with patch("src.joblib.load", return_value=None), \
         patch("src.joblib.dump"), \
         patch("src.load_model", return_value=mock_model_func), \
         patch("src.MinMaxScaler") as mock_scaler:
        
        # Mock the scaler's fit and transform methods
        mock_scaler.return_value.fit.return_value = None
        mock_scaler.return_value.transform.return_value = df.copy()

        # When
        predictions, y_test, mse, mae, rmse, times = plot_forecast_vs_actual(
            df, start_time, end_time, site_index, mock_model_func, lookback_hours
        )

        # Then
        assert predictions is not None, "Predictions were not created."
        assert y_test is not None, "y_test was not created."
        assert mse is not None, "MSE was not calculated."
        assert mae is not None, "MAE was not calculated."
        assert rmse is not None, "RMSE was not calculated."
        assert times is not None, "Times were not created."
        assert len(predictions) == len(y_test), "Predictions and y_test lengths do not match."