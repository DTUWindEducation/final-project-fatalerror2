from svm import *

# Train using 4 lag terms
mae, mse, rmse = train_and_save_svm(site_index=1, num_lags=5)
print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

# Plot using the same number of lags
plot_prediction_vs_actual(site_index=1, start_time="2017-01-05", end_time="2017-01-10", num_lags=5)