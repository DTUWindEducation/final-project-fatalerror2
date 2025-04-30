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
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf


# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import determine_winner
from src import load_site_data
from src.WindPowerForecaster import WindPowerForecaster
from src import load_and_filter_by_site, prepare_features, create_lstm_model


#test the function load_site_data using pytest

####################################
# Tests for functions in __init__.py
####################################

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

def validate_prepare_features_output(X, y, features, mock_df, num_lags):
    """Helper function to validate the output of prepare_features. See function below for mock data"""
    # Check the shape of the feature matrix and target variable
    assert X.shape[0] == len(mock_df) - num_lags - 1, "Feature matrix has incorrect number of rows."
    assert X.shape[1] == len(mock_df.columns) + num_lags - 1, "Feature matrix has incorrect number of columns."
    assert y.shape[0] == len(mock_df) - num_lags - 1, "Target vector has incorrect number of rows."

    # Check that lagged features are included
    expected_features = [
        'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
        'windspeed_10m', 'windspeed_100m',
        'winddirection_10m', 'winddirection_100m',
        'windgusts_10m', 'Power_t-1', 'Power_t-2'
    ]
    assert features == expected_features, "Feature list does not match expected features."

    # Check that the target variable is shifted correctly
    assert np.array_equal(y, mock_df["Power"].iloc[num_lags + 1:].values), "Target variable is incorrect."


def test_prepare_features():
    """Test the prepare_features function."""
    # Given
    mock_df = pd.DataFrame({
        'Power': np.random.rand(10),
        'temperature_2m': np.random.rand(10),
        'relativehumidity_2m': np.random.rand(10),
        'dewpoint_2m': np.random.rand(10),
        'windspeed_10m': np.random.rand(10),
        'windspeed_100m': np.random.rand(10),
        'winddirection_10m': np.random.rand(10),
        'winddirection_100m': np.random.rand(10),
        'windgusts_10m': np.random.rand(10),
    })
    num_lags = 2

    # When
    X, y, features = prepare_features(mock_df, num_lags)

    # Then
    validate_prepare_features_output(X, y, features, mock_df, num_lags)

def test_create_lstm_model():
    """Test the create_lstm_model function."""
    # Given
    input_shape = (10, 5)  #Just an example shape, adjust if needed

    # When
    model = create_lstm_model(input_shape)

    # Then
    assert model is not None, "Model should be created successfully."
    assert len(model.layers) > 0, "Model should have layers."
    assert model.input_shape == (None, 10, 5), "Input shape of the model is incorrect."
    assert isinstance(model.layers[0], tf.keras.layers.LSTM), "First layer should be LSTM."

#################################################
#Tests for functions in WindPowerForecaster class
#################################################

def test_train_and_save_svm():
    """Test the train_and_save_svm function."""
    # Given
    site_index = 1
    start_time = "2022-01-01"
    end_time = "2022-12-31"
    num_lags = 2

    #create instant of windpowerforecaster
    forecaster = WindPowerForecaster(site_index=site_index, start_time=start_time, end_time=end_time)

    # Mock the load_site_data function to return synthetic data
    with patch("src.WindPowerForecaster.load_site_data") as mock_load_site_data, \
        patch("src.WindPowerForecaster.prepare_features") as mock_prepare_features, \
        patch("src.WindPowerForecaster.joblib.dump") as mock_joblib_dump, \
        patch("pathlib.Path.exists", return_value=False): #Force fresh training scenario

        
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
        mae, mse, rmse = forecaster.train_and_save_svm(num_lags)

        # Then
        assert mae > 0, "MAE should be greater than 0."
        assert mse > 0, "MSE should be greater than 0."
        assert rmse > 0, "RMSE should be greater than 0."
        mock_joblib_dump.assert_called()  # Ensure the model and scaler were saved


def test_determine_winner_best_in_all_metrics():
    """Test when one model is the best in all three metrics."""
    # Given
    models_metrics = {
        "ModelA": {"MAE": 10, "MSE": 100, "RMSE": 10},
        "ModelB": {"MAE": 15, "MSE": 150, "RMSE": 12},
        "ModelC": {"MAE": 20, "MSE": 200, "RMSE": 14},
    }

    # When
    winner = determine_winner(models_metrics)

    # Then
    assert winner == "ModelA", "ModelA should be the winner as it is best in all metrics."


def test_determine_winner_lowest_rmse():
    """Test when the winner is determined by the lowest RMSE."""
    # Given
    models_metrics = {
        "ModelA": {"MAE": 15, "MSE": 150, "RMSE": 12},
        "ModelB": {"MAE": 10, "MSE": 100, "RMSE": 10},
        "ModelC": {"MAE": 20, "MSE": 200, "RMSE": 14},
    }

    # When
    winner = determine_winner(models_metrics)

    # Then
    assert winner == "ModelB", "ModelB should be the winner as it has the lowest RMSE."


def test_determine_winner_tie_breaker_by_mae():
    """Test when there is a tie in RMSE, and the winner is determined by MAE."""
    # Given
    models_metrics = {
        "ModelA": {"MAE": 10, "MSE": 100, "RMSE": 10},
        "ModelB": {"MAE": 8, "MSE": 100, "RMSE": 10},  # Tie in RMSE, but lower MAE
        "ModelC": {"MAE": 12, "MSE": 120, "RMSE": 10},
    }

    # When
    winner = determine_winner(models_metrics)

    # Then
    assert winner == "ModelB", "ModelB should be the winner as it has the lowest MAE in a tie."