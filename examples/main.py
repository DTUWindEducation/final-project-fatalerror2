import os
import sys

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import train_and_save_svm, plot_prediction_vs_actual


# Train using 4 lag terms
mae, mse, rmse = train_and_save_svm(site_index=1, num_lags=5)
print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

# Plot using the same number of lags
plot_prediction_vs_actual(site_index=1, start_time="2017-01-05", end_time="2017-01-10", num_lags=5)