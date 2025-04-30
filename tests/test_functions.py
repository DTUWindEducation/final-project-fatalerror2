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
from src import determine_winner
from src import load_site_data
from src.WindPowerForecaster import WindPowerForecaster
from src import load_and_filter_by_site


#test the function load_site_data using pytest


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

def test_load_and_filter_by_site():
    """Test the load_and_filter_by_site function."""
    # Given
    inputs_dir = Path(__file__).parents[1] / "inputs"
    site_index = 3

    # When
    df = load_and_filter_by_site(inputs_dir, site_index)

    # Then
    assert not df.empty, "DataFrame is empty."
    assert 'hour' in df.columns, "Hour column is missing."
    assert 'hour_sin' in df.columns, "Hour sine column is missing."
    assert 'hour_cos' in df.columns, "Hour cosine column is missing."


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