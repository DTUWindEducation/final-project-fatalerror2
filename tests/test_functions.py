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
from pandas import Timestamp
from unittest.mock import MagicMock, patch
import tempfile
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend


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


def test_load_and_filter_by_site():
    """Test the load_and_filter_by_site function. 
    Specifically, check if the function correctly loads and filters data for a given site index."""
    # Given
    inputs_dir = Path(__file__).parents[1] / "inputs"
    site_index = 1
    power_exp = 0.1635  # Expected value of power

    # When
    df = load_and_filter_by_site(inputs_dir, site_index)

    # Then
    assert not df.empty, "DataFrame is empty."
    assert 'Time' in df.columns, "Time column is missing."
    assert pd.api.types.is_datetime64_any_dtype(df['Time']), "Time column is not in datetime format."
    assert 'hour' in df.columns, "Hour column is missing."
    assert 'hour_sin' in df.columns, "Hour sine column is missing."
    assert 'hour_cos' in df.columns, "Hour cosine column is missing."
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


#################################################
#Tests for functions in WindPowerForecaster class
#################################################

def test_filter_and_plot():
    """Test the filter_and_plot function."""
    # Given
    site_index = 1
    start_time = "2022-01-01"
    end_time = "2022-01-02"
    inputs_dir = Path("mock_inputs_dir")  # Mock directory for inputs
    variable = "Power"

    # Create an instance of WindPowerForecaster
    forecaster = WindPowerForecaster(site_index=site_index, start_time=start_time, end_time=end_time)

    # Mock data
    mock_data = {
        "Time": pd.date_range(start="2022-01-01", periods=48, freq="h"),
        "Power": np.random.rand(48),
    }
    mock_df = pd.DataFrame(mock_data)

    # Correctly patch load_and_filter_by_site
    with patch("src.WindPowerForecaster.load_and_filter_by_site", return_value=mock_df) as mock_load_and_filter_by_site, \
         patch("matplotlib.pyplot.show") as mock_plt_show:
        # When
        forecaster.filter_and_plot(inputs_dir, variable)

        # Then
        mock_load_and_filter_by_site.assert_called_once_with(inputs_dir, site_index)
        mock_plt_show.assert_called_once()

        # Verify the filtered data
        filtered_df = mock_df[
            (mock_df["Time"] >= pd.to_datetime(start_time)) &
            (mock_df["Time"] <= pd.to_datetime(end_time))
        ]
        assert not filtered_df.empty, "Filtered DataFrame should not be empty."
        assert filtered_df["Time"].min() >= pd.to_datetime(start_time), "Filtered data starts before start_time."
        assert filtered_df["Time"].max() <= pd.to_datetime(end_time), "Filtered data ends after end_time."


def test_split_train_test():
    # Given
    site_index = 1
    start_time = "2023-01-01"
    end_time = "2023-01-31"

    # When
    forecaster = WindPowerForecaster(site_index=site_index, start_time=start_time, end_time=end_time)
    x = np.arange(100).reshape(-1, 1)  # Mock feature data
    y = np.arange(100)  # Mock target data
    x_train, x_test, y_train, y_test = forecaster.split_train_test(x, y)

    # Then
    assert len(x_train) == 80
    assert len(x_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20
    np.testing.assert_array_equal(x_train, x[:80])
    np.testing.assert_array_equal(x_test, x[80:])
    np.testing.assert_array_equal(y_train, y[:80])
    np.testing.assert_array_equal(y_test, y[80:])


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


def test_print_evaluation_metrics(capsys):
    """Test the print_evaluation_metrics function."""
    # Given
    mae = 1.23456
    mse = 2.34567
    rmse = 3.45678

    # When
    forecaster = WindPowerForecaster(site_index=1, start_time="2020-11-15", end_time="2020-11-16")
    forecaster.print_evaluation_metrics(mae, mse, rmse)

    # Then
    captured = capsys.readouterr()
    expected_output = (
        "\n=== MODEL EVALUATION METRICS ===\n"
        "Mean Absolute Error (MAE):   1.23456\n"
        "Mean Squared Error (MSE):    2.34567\n"
        "Root Mean Squared Error (RMSE): 3.45678\n\n"
    )
    assert captured.out == expected_output


def test_plot_svm_result():
    """Test the plot_svm_result function."""
    # Given
    site_index = 1
    start_time = "2020-01-01"
    end_time = "2020-12-31"
    num_lags = 5

    # When
    forecaster = WindPowerForecaster(site_index=site_index, start_time=start_time, end_time=end_time)
    fig = forecaster.plot_svm_result(num_lags)

    # Then
    assert isinstance(fig, plt.Figure)


def test_plot_lstm_result():
    """Test the plot_lstm_result function."""
    # Given
    site_index = 1
    start_time = "2020-11-15"
    end_time = "2020-11-16"
    lookback_hours = 5

    # When
    forecaster = WindPowerForecaster(site_index=site_index, start_time=start_time, end_time=end_time)
    prediction, y_test, mse, mae, rmse, times, fig = forecaster.plot_lstm_result(
        model_funct=create_lstm_model,
        lookback_hours=lookback_hours
    )

    # Then
    assert isinstance(fig, plt.Figure)
    assert len(y_test) == len(times)


def test_plot_persistence_result():
    """Test the plot_persistence_result function."""
    # Given
    site_index = 1
    start_time = "2020-11-15"
    end_time = "2020-11-16"
    num_lags = 5
    mae_exp = 0.0157166666
    mse_exp = 0.0004309275
    rmse_exp = 0.020758793

    # When
    forecaster = WindPowerForecaster(site_index=site_index, start_time=start_time, end_time=end_time)
    y_test = [0.8603, 0.8648, 0.8693, 0.8738, 0.8784, 0.8789, 0.8753, 0.8718, 0.8683, 0.8648, 
              0.8612, 0.8577, 0.8365, 0.7977, 0.7588, 0.7199, 0.6811, 0.6422, 0.6321, 0.6508, 
              0.6694, 0.688, 0.7067, 0.7253, 0.7439]
    times = [Timestamp('2020-11-15 00:00:00'), Timestamp('2020-11-15 01:00:00'), Timestamp('2020-11-15 02:00:00'),
             Timestamp('2020-11-15 03:00:00'), Timestamp('2020-11-15 04:00:00'), Timestamp('2020-11-15 05:00:00'),
             Timestamp('2020-11-15 06:00:00'), Timestamp('2020-11-15 07:00:00'), Timestamp('2020-11-15 08:00:00'),
             Timestamp('2020-11-15 09:00:00'), Timestamp('2020-11-15 10:00:00'), Timestamp('2020-11-15 11:00:00'),
             Timestamp('2020-11-15 12:00:00'), Timestamp('2020-11-15 13:00:00'), Timestamp('2020-11-15 14:00:00'),
             Timestamp('2020-11-15 15:00:00'), Timestamp('2020-11-15 16:00:00'), Timestamp('2020-11-15 17:00:00'),
             Timestamp('2020-11-15 18:00:00'), Timestamp('2020-11-15 19:00:00'), Timestamp('2020-11-15 20:00:00'),
             Timestamp('2020-11-15 21:00:00'), Timestamp('2020-11-15 22:00:00'), Timestamp('2020-11-15 23:00:00'),
             Timestamp('2020-11-16 00:00:00')]
    mae, mse, rmse = forecaster.plot_persistence_result(y_test, times)

    # Then
    assert np.isclose(mae, mae_exp, rtol=1e-5), "MAE does not match expected value."
    assert np.isclose(mse, mse_exp, rtol=1e-5), "MSE does not match expected value."
    assert np.isclose(rmse, rmse_exp, rtol=1e-5), "RMSE does not match expected value."


def test_create_lagged_cf_features():
    # Given
    site_index = 1
    start_time = "2023-01-01"
    end_time = "2023-01-31"
    cp_exp = 0.19415

    # When
    df = load_site_data(site_index)
    forecaster = WindPowerForecaster(site_index=site_index, start_time=start_time, end_time=end_time)
    daily_cf = forecaster.compute_daily_capacity_factor(df)

    # Then
    assert not daily_cf.empty, "Daily capacity factor DataFrame is empty."
    assert np.isclose(daily_cf['Capacity_Factor'].iloc[0], cp_exp), "Capacity factor value does not match the expected value."


def test_create_lagged_cf_features():
    # Given
    site_index = 1
    start_time = "2023-01-01"
    end_time = "2023-01-31"
    feature_len_exp = 10

    # When
    df = load_site_data(site_index)
    forecaster = WindPowerForecaster(site_index=site_index, start_time=start_time, end_time=end_time)
    daily_cf = forecaster.compute_daily_capacity_factor(df)
    df2, feature = forecaster.create_lagged_cf_features(daily_cf, num_lags=10)

    # Then
    assert not df2.empty, "Daily capacity factor DataFrame is empty."
    assert np.isclose(len(feature), feature_len_exp), "Feature length does not match expected value."


def test_train_capacity_factor_model():
    # Given
    site_index = 1
    start_time = "2023-01-01"
    end_time = "2023-01-31"
    feature_len_exp = 10

    # When
    forecaster = WindPowerForecaster(site_index=site_index, start_time=start_time, end_time=end_time)
    model, scaler = forecaster.train_capacity_factor_model(num_lags=10)

    # Then
    assert isinstance(model, SVR)
    assert isinstance(scaler, StandardScaler)
    assert hasattr(forecaster, 'x_train')
    assert hasattr(forecaster, 'x_test')
    assert isinstance(forecaster.x_train, np.ndarray)
    assert isinstance(forecaster.x_test, np.ndarray)
    assert len(forecaster.x_train) > 0
    assert len(forecaster.x_test) > 0
    assert hasattr(forecaster, 'cf_model')
    assert hasattr(forecaster, 'scaler')


def test_predict_capacity_factor(capsys):
    # Given
    site_index = 1
    start_time = "2020-11-15"
    end_time = "2020-11-16"

    # When
    forecaster = WindPowerForecaster(site_index=site_index, start_time=start_time, end_time=end_time)
    forecaster.train_capacity_factor_model(num_lags=10)
    forecaster.predict_capacity_factor()
    captured = capsys.readouterr()

    # Then
    assert "=== 📈 Capacity Factor Forecast Result ===\n" in captured.out
    assert "Prediction Date: 2020-11-15\n" in captured.out
    assert "Predicted CF:    0.4960\n" in captured.out
    assert "Actual CF:       0.5889\n" in captured.out
    assert "Percentual Error: 15.79%\n" in captured.out

