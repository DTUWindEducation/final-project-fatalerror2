import os
import sys
from pathlib import Path

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import train_and_save_svm, plot_prediction_vs_actual
from src import load_and_filter_by_site, filter_and_plot, create_lstm_model, plot_forecast_vs_actual

# --- Configurations ---
site_index = 1
num_lags = 5
start_time = "2017-01-05"
end_time = "2017-01-10"
lookback_hours = 18
training_window_months = 6

# --- Run SVM ---
print("\n--- SVM Training ---")
mae, mse, rmse = train_and_save_svm(site_index=site_index, num_lags=num_lags)
print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
plot_prediction_vs_actual(site_index, start_time, end_time, num_lags=num_lags)

# --- Run LSTM ---
print("\n--- LSTM Training ---")
project_root = Path(__file__).resolve().parent.parent
inputs_dir = project_root / "inputs"
site_df, site_name = load_and_filter_by_site(inputs_dir, site_index)
filter_and_plot(site_df, "Power", start_time, end_time, site_name)

predictions, y_test, mse, mae, rmse, times = plot_forecast_vs_actual(
    site_df, start_time, end_time, site_name,
    model_func=create_lstm_model,
    lookback=lookback_hours,
    training_months=training_window_months
)

if predictions is not None:
    print("\n--- LSTM Evaluation ---")
    print(f"MSE:  {mse:.5f}")
    print(f"MAE:  {mae:.5f}")
    print(f"RMSE: {rmse:.5f}")
