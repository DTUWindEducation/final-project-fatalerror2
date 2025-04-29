"""Class for wind power forecasting"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from tensorflow.python.keras.models import load_model
from keras.src.callbacks import EarlyStopping
from pathlib import Path

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import load_and_filter_by_site, load_site_data, prepare_features

class WindPowerForecaster:
    def __init__(self, site_index, start_time, end_time):
        self.site_index = site_index
        self.start_time = start_time
        self.end_time = end_time

    def filter_and_plot(self, inputs_dir, variable):
        """
        Helper function to filter and plot time series data.
        Args:
            inputs_dir (_type_): _description_
            variable (_type_): _description_
        """
        site_df = load_and_filter_by_site(inputs_dir, self.site_index)
        filtered = site_df[(site_df['Time'] >= pd.to_datetime(self.start_time)) & (site_df['Time'] <= pd.to_datetime(self.end_time))]
        plt.figure(figsize=(12, 6))
        plt.plot(filtered['Time'], filtered[variable], color='black', linestyle='-', marker='^', label=variable)
        plt.title(f"Location {self.site_index} - {variable}")
        plt.xlabel('Time')
        plt.ylabel(variable)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def train_and_save_svm(self, num_lags):
        """
        Train or load SVM model for a specific site.
        Handles data splitting, scaling, training, and evaluation.
        """
        # File Path Setup
        folder_path = Path(__file__).parents[1]
        model_path = folder_path / f"outputs/Location{self.site_index}_svr_model.pkl"
        scaler_path = folder_path / f"outputs/Location{self.site_index}_scaler.pkl"

        # Model Loading/Training
        if model_path.exists() and scaler_path.exists():
            print("Loaded existing SVM model and scaler.")
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            # Load data for evaluation
            df = load_site_data(self.site_index)
            X, y, _ = prepare_features(df, num_lags=num_lags)
            
            # Data splitting (80-20 split)
            split_index = int(len(X) * 0.8)
            X_test = scaler.transform(X[split_index:])
            y_test = y[split_index:]
            
            # Prediction
            y_pred = model.predict(X_test)

        else:
            print("Training new SVM model...")
            df = load_site_data(self.site_index)
            X, y, _ = prepare_features(df, num_lags=num_lags)

            # Data Splitting
            split_index = int(len(X) * 0.8)
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]

            # Feature Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Model Training
            model = SVR(kernel='rbf')
            model.fit(X_train_scaled, y_train)

            # Prediction
            y_pred = model.predict(X_test_scaled)

            # Model Saving
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            print("💾 SVM model and scaler saved.")

        # Evaluation
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        return mae, mse, rmse

    def print_evaluation_metrics(self, mae, mse, rmse):
        """
        Print formatted evaluation metrics for model comparison.
        """
        # Print evaluation metrics
        print("\n=== MODEL EVALUATION METRICS ===")
        print(f"Mean Squared Error (MSE):    {mse:.5f}")
        print(f"Mean Absolute Error (MAE):   {mae:.5f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.5f}\n")

    def plot_svm_result(self, num_lags):
        """
        Plot SVM predictions vs actual values for a specific time period.
        """
        # Data Preparation
        df = load_site_data(self.site_index)
        for lag in range(1, num_lags + 1):
            df[f'Power_t-{lag}'] = df['Power'].shift(lag)

        # Filter by time range
        mask = (df['Time'] >= pd.to_datetime(self.start_time)) & (df['Time'] <= pd.to_datetime(self.end_time))
        subset = df.loc[mask].copy()
        subset.dropna(inplace=True)

        # Feature Selection
        features = [
            'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
            'windspeed_10m', 'windspeed_100m',
            'winddirection_10m', 'winddirection_100m',
            'windgusts_10m'
        ] + [f'Power_t-{lag}' for lag in range(1, num_lags + 1)]

        # Model Loading
        folder_path = Path(__file__).parents[1]
        model = joblib.load(folder_path / f"outputs/Location{self.site_index}_svr_model.pkl")
        scaler = joblib.load(folder_path / f"outputs/Location{self.site_index}_scaler.pkl")

        # Prediction
        X_subset = subset[features].values
        X_scaled = scaler.transform(X_subset)
        subset['Predicted_Power'] = model.predict(X_scaled)

        # Plotting
        plt.figure(figsize=(12, 5))
        plt.plot(subset['Time'], subset['Power'], 'k^-', label="Measured Power")
        plt.plot(subset['Time'], subset['Predicted_Power'], 'b.-', label="Predicted Power (SVM)")
        plt.title(f"Location{self.site_index} - Measured vs Predicted Power (SVM)")
        plt.xlabel("Time")
        plt.ylabel("Power")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_lstm_result(self, model_funct, lookback_hours):
        """
        Train or load LSTM model and plot predictions vs actual values.
        Handles data preparation, scaling, windowing, and evaluation.
        """
        features = [
            'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
            'windspeed_10m', 'windspeed_100m', 'winddirection_10m',
            'winddirection_100m', 'windgusts_10m', 'Power',
            'hour_sin', 'hour_cos'
        ]

        # Start and End Time
        start_dt, end_dt = pd.to_datetime(self.start_time), pd.to_datetime(self.end_time)

        # File Path Setup
        folder_path = Path(__file__).parents[1] / "outputs"
        model_path = folder_path / f"Location{self.site_index}_lstm_model.h5"
        scaler_path = folder_path / f"Location{self.site_index}_lstm_scaler.pkl"
        folder_path.mkdir(parents=True, exist_ok=True)

        # Load Data
        inputs_dir = Path(__file__).parents[1] / "inputs"
        site_df = load_and_filter_by_site(inputs_dir, self.site_index)

        # Feature Scaling
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            print("Loaded existing scaler.")
        else:
            print("Training new scaler...")
            scaler = MinMaxScaler()
            scaler.fit(site_df[features])
            joblib.dump(scaler, scaler_path)
            print("💾 Scaler saved.")

        df_scaled = site_df.copy()
        df_scaled[features] = scaler.transform(site_df[features])

        # Data Splitting (80-20)
        split_index = int(len(df_scaled) * 0.8)
        train_df = df_scaled.iloc[:split_index]
        raw_train_df = site_df.iloc[:split_index]

        # Model Training/Loading
        if not model_path.exists():
            print("🛠️ Training new LSTM model...")

            # Create sliding windows for LSTM
            X_train, y_train = [], []
            for i in range(lookback_hours, len(train_df) - 1):
                window = train_df.iloc[i - lookback_hours:i][features]
                X_train.append(window.values)
                y_train.append(raw_train_df.iloc[i + 1]['Power'])

            X_train, y_train = np.array(X_train), np.array(y_train)

            if len(X_train) < 10:
                print("Not enough training data.")
                return None, None, None, None, None, None

            # Train model with early stopping
            model = model_funct(X_train[0].shape)
            model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0,
                    validation_split=0.1, shuffle=False,
                    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

            model.save(model_path)
            print("Model trained and saved.")
        else:
            model = load_model(model_path)
            print("Loaded existing LSTM model.")

        # Test Data Preparation
        X_test, y_test, times = [], [], []
        for i in range(lookback_hours, len(df_scaled) - 1):
            current_time = df_scaled.iloc[i + 1]['Time']
            if start_dt <= current_time <= end_dt:
                window = df_scaled.iloc[i - lookback_hours + 1:i + 1][features]
                X_test.append(window.values)
                y_test.append(site_df.iloc[i + 1]['Power'])
                times.append(current_time)

        if len(X_test) < 1:
            print("Not enough test data.")
            return None, None, None, None, None, None

        X_test, y_test = np.array(X_test), np.array(y_test)

        # Prediction
        predictions = model.predict(X_test).flatten()

        # Evaluation Metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)

        # Plotting
        plt.figure(figsize=(14, 6))
        plt.plot(times, y_test, 'k^-', label='Measured Power')
        plt.plot(times, predictions, 'b.-', label='Predicted Power (LSTM)')
        plt.title(f"Location{self.site_index} - Predicted vs Measured Power (Neural Networks)")
        plt.xlabel('Time')
        plt.ylabel('Power')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return predictions, y_test, mse, mae, rmse, times


    def plot_persistence_result(self, y_test, times):
        """
        Implement and plot persistence model (naive forecast).
        The persistence model uses the previous value as the prediction.
        """
        if len(y_test) < 2:
            print("Not enough data for persistence model.")
            return

        # Create persistence predictions (shift actual values by 1)
        y_persistence = np.roll(y_test, 1)
        y_persistence[0] = y_test[0]  # Handle first value

        # Calculate metrics (skip first point)
        mse = mean_squared_error(y_test[1:], y_persistence[1:])
        mae = mean_absolute_error(y_test[1:], y_persistence[1:])
        rmse = np.sqrt(mse)

        # Plot results
        plt.figure(figsize=(14, 6))
        plt.plot(times, y_test, label='Measured Power', color='black', marker='^', linestyle='-')
        plt.plot(times, y_persistence, label='Persistence Model', color='blue', marker='o', linestyle='-')
        plt.title("Persistence Model vs Measured Power")
        plt.xlabel('Time')
        plt.ylabel('Power')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return mae, mse, rmse
