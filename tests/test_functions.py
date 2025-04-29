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
import tempfile

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


#test the function load_site_data using pytest
from src import load_site_data


def test_load_site_data():
    """Test the load_site_data function. 
    Specifically, check if the function correctly loads data for a given site index.
    in this case the power from location 1. Also check if the data is in the correct format."""
    # Given
    site_index = 1
    power_exp = 0.1635  # Expected value of power

    # When
    df = load_site_data(site_index)  # Correctly handle the single return value

    # Then
    assert not df.empty, "DataFrame is empty."
    assert 'Time' in df.columns, "Time column is missing."
    assert pd.api.types.is_datetime64_any_dtype(df['Time']), "Time column is not in datetime format."
    assert 'Power' in df.columns, "Power column is missing."
    assert np.isclose(df['Power'].iloc[0], power_exp), "Power value does not match the expected value."

#test the function train_and_save_svm using pytest
from src import train_and_save_svm, prepare_features


def test_train_and_save_svm():
    """Test the train_and_save_svm function."""
    # Given
    site_index = 1
    num_lags = 2

     # Mock the load_site_data function to return synthetic data
    with patch("src.load_site_data") as mock_load_site_data, \
         patch("src.prepare_features") as mock_prepare_features, \
         patch("src.joblib.dump") as mock_joblib_dump:

        # Create synthetic data
        mock_df = pd.DataFrame({
            'Time': pd.date_range(start="2022-01-01", periods=100, freq='h'),
            'Power': np.random.rand(100),
            'temperature_2m': np.random.rand(100),
            'relativehumidity_2m': np.random.rand(100),
            'dewpoint_2m': np.random.rand(100),
            'windspeed_10m': np.random.rand(100),
            'windspeed_100m': np.random.rand(100),
            'winddirection_10m': np.random.rand(100),
            'winddirection_100m': np.random.rand(100),
            'windgusts_10m': np.random.rand(100),
        })
        mock_load_site_data.return_value = mock_df

        # Mock prepare_features to return data with the correct number of features
        num_features = 13  # To be updated
        mock_X = np.random.rand(80, num_features)  # 80 rows, num_features columns
        mock_y = np.random.rand(80)  # 80 target values
        mock_prepare_features.return_value = (mock_X, mock_y, None)

        # When
        mae, mse, rmse = train_and_save_svm(site_index, num_lags)

        # Then
        assert mae > 0, "MAE should be greater than 0."
        assert mse > 0, "MSE should be greater than 0."
        assert rmse > 0, "RMSE should be greater than 0."
        #mock_joblib_dump.assert_called()  # Ensure the model and scaler were saved
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