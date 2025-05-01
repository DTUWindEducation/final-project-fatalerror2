import os
import sys
from pathlib import Path
import numpy as np
import time

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import load_site_data, prepare_features, create_lstm_model, determine_winner
from src.WindPowerForecaster import WindPowerForecaster

# Start timing
execution_start_time = time.time()  

# --- User Configurations ---
site_index = 1
variable_name = "Power"
start_time = "2020-11-15"
end_time = "2020-11-16"

forecaster = WindPowerForecaster(site_index=site_index, start_time=start_time, end_time=end_time)

# --- Data Files Directory ---
project_root = Path(__file__).resolve().parent.parent
inputs_dir = Path(project_root / "inputs")

# Functional requirements #1: Load and parse the provided input data
df = load_site_data(site_index)

# Functional requirements #2: Plot timeseries of a selected variable for a given site within a specific perid
forecaster.filter_and_plot(inputs_dir, variable_name)

# --- Machine Learning Configurations ---
num_lags_SVM = 5 # SVM Model and CF Prediction
lookback_hours = 18 # Neural Network Model
num_lags_CT_prediction = 10 # CF Prediction number of days as input features to predict the next day.

# --- Prepare Data ---
# Functional requirements #4: Split the dataset into training dataset and test dataset
x, y, _ = prepare_features(df, num_lags=num_lags_SVM)
x_train, x_test, y_train, y_test = forecaster.split_train_test(x, y, split_ratio=0.8)

# --- Run SVM ---
# Functional requirements #3: Compute MAE, MSE, RMSE for SVM model
# Functional requirements #6 & #7: Plot predictions against real power time series for SVM model
svm_mae, svm_mse, svm_rmse = forecaster.train_and_save_svm(num_lags=num_lags_SVM)
forecaster.plot_svm_result(num_lags=num_lags_SVM)

# --- Run LSTM ---
# Functional requirements #3: Compute MAE, MSE, RMSE for LSTM model
# Functional requirements #6 & #7: Plot predictions against real power time series for LSTM model
_, y_test, lstm_mse, lstm_mae, lstm_rmse, times, _ = forecaster.plot_lstm_result(model_funct=create_lstm_model,
                                                                                 lookback_hours=lookback_hours)

# --- Run Persistence Model ---
# Functional requirements #3: Compute MAE, MSE, RMSE for Persistence model
# Functional requirements #5 & #7: Plot predictions against real power time series for Persistence model
persistence_mae, persistence_mse, persistence_rmse = forecaster.plot_persistence_result(np.array(y_test), times)

# --- Print the evaluation metrics for the three models ---
print("\n--- SVM Evaluation ---")
forecaster.print_evaluation_metrics(svm_mae, svm_mse, svm_rmse)
print("\n--- LSTM Evaluation ---")
forecaster.print_evaluation_metrics(lstm_mae, lstm_mse, lstm_rmse)
print("\n--- Persistance Evaluation ---")
forecaster.print_evaluation_metrics(persistence_mae, persistence_mse, persistence_rmse)

# --- Extra Function - Determine Best Model --- .

print("\n--- Determining Best Model ---")

models_metrics = {
    'SVM': {'MAE': svm_mae, 'MSE': svm_mse, 'RMSE': svm_rmse},
    'LSTM': {'MAE': lstm_mae, 'MSE': lstm_mse, 'RMSE': lstm_rmse},
    'Persistence': {'MAE': persistence_mae, 'MSE': persistence_mse, 'RMSE': persistence_rmse}
}

determine_winner(models_metrics)

# --- Extra Function - Daily Capacity Factor Prediction ---

print("\n--- Predicting Next Day Capacity Factor ---")
print("\n--- Training Capacity Factor Model ---")
forecaster.train_capacity_factor_model(num_lags=num_lags_CT_prediction)

print("\n--- Predicting Capacity Factor for Start Date ---")
forecaster.predict_capacity_factor()


# Compute and print total time
total_time = time.time() - execution_start_time
print(f"\nTotal execution time: {int(total_time // 60)} min {int(total_time % 60)} sec")
