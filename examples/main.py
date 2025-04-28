import os
import sys
from pathlib import Path
import numpy as np
import time

# Start timing
execution_start_time = time.time()  

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#Import Functions
from src import (train_and_save_svm, 
                plot_prediction_vs_actual, 
                print_evaluation_metrics,
                load_and_filter_by_site, 
                filter_and_plot, 
                create_lstm_model, 
                plot_forecast_vs_actual,
                plot_persistence_model)

# --- User Configurations ---
site_index = 1
variable_name = "Power"
start_time = "2017-01-20"
end_time = "2017-01-21"

# --- Machine Learning Configurations ---
num_lags = 5 # SVM Model
lookback_hours = 18 # Neural Network Model


# --- Data Files Directory ---
project_root = Path(__file__).resolve().parent.parent
inputs_dir = project_root / "inputs"

# --- Plot Desired Variable for Selected Site ---
site_df = load_and_filter_by_site(inputs_dir, site_index)
filter_and_plot(site_df, variable_name, site_index, start_time, end_time)

print("\n--- Starting Machine Learning Models... ---")

# --- Run SVM ---
print("\n--- Loading SVM Model ...---")
mae, mse, rmse = train_and_save_svm(site_index=site_index, num_lags=num_lags)
print("\n--- Starting Prediction ... ---")

print("\n--- SVM Evaluation ---")
print_evaluation_metrics(mae, mse, rmse)

plot_prediction_vs_actual(site_index, start_time, end_time, num_lags=num_lags)

# --- Run LSTM ---
print("\n--- Initializing Neural Network LSTM Training... ---")

predictions, y_test, mse, mae, rmse, times = plot_forecast_vs_actual(
    site_df, start_time, end_time, site_index,
    model_func=create_lstm_model,
    lookback=lookback_hours)

print("\n--- LSTM Evaluation ---")
print_evaluation_metrics(mae, mse, rmse)

# --- Persistence Model Evaluation ---
print("\n--- Generating Persistance Model ---")
mae, mse, rmse = plot_persistence_model(np.array(y_test), times)
print("\n--- Persistance Evaluation ---")
print_evaluation_metrics(mae, mse, rmse)


# Compute and print total time
total_time = time.time() - execution_start_time
print(f"\n⏱️ Total execution time: {int(total_time // 60)} min {int(total_time % 60)} sec")