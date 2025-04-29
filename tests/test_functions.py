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
