"""Class for wind power forecasting"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from tensorflow.keras.models import load_model # pylint: disable=E0401,E0611
from tensorflow.keras.callbacks import EarlyStopping # pylint: disable=E0401,E0611

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import load_and_filter_by_site, load_site_data, prepare_features

class WindPowerForecaster:
    """Class for wind power forecasting"""
    def __init__(self, site_index, start_time, end_time):
        self.site_index = site_index
        self.start_time = start_time
        self.end_time = end_time

    def filter_and_plot(self, inputs_dir, variable):
        """Helper function to filter and plot time series data"""
        site_df = load_and_filter_by_site(inputs_dir, self.site_index)
        filtered = site_df[
            (site_df['Time'] >= pd.to_datetime(self.start_time)) &
            (site_df['Time'] <= pd.to_datetime(self.end_time))]
        filtered_time = filtered['Time']
        filtered_variable = filtered[variable]

        plt.figure(figsize=(12, 6))
        plt.plot(filtered_time, filtered_variable, color='black',
                 linestyle='-', marker='^', label=variable)
        plt.title(f"Location {self.site_index} - {variable}")
        plt.xlabel('Time')
        plt.ylabel(variable)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def split_train_test(self, x, y, split_ratio=0.8):
        """
        Split data into training and testing sets.
        :param x: Features
        :param y: Target variable
        :param split_ratio: Ratio for splitting the data
        :return: x_train, x_test, y_train, y_test
        """
        split_index = int(len(x) * split_ratio)
        x_train, x_test = x[:split_index], x[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        return x_train, x_test, y_train, y_test

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
            x, y, _ = prepare_features(df, num_lags=num_lags)

            # Data splitting (80-20 split)
            _, x_test, _, y_test = self.split_train_test(x, y, split_ratio=0.8)
            x_test = scaler.transform(x_test)
            # split_index = int(len(x) * 0.8)
            # x_test = scaler.transform(x[split_index:])
            # y_test = y[split_index:]

            # Prediction
            y_pred = model.predict(x_test)

        else:
            print("Training new SVM model...")
            df = load_site_data(self.site_index)
            x, y, _ = prepare_features(df, num_lags=num_lags)

            # Data Splitting
            x_train, x_test, y_train, y_test = self.split_train_test(x, y, split_ratio=0.8)

            # Feature Scaling
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)

            # Model Training
            model = SVR(kernel='rbf')
            model.fit(x_train_scaled, y_train)

            # Prediction
            y_pred = model.predict(x_test_scaled)

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
        print(f"Mean Absolute Error (MAE):   {mae:.5f}")
        print(f"Mean Squared Error (MSE):    {mse:.5f}")
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
        start = pd.to_datetime(self.start_time)
        end = pd.to_datetime(self.end_time)
        mask = (df['Time'] >= start) & (df['Time'] <= end)
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
        x_subset = subset[features].values
        x_scaled = scaler.transform(x_subset)
        subset['Predicted_Power'] = model.predict(x_scaled)

        # Plotting
        fig = plt.figure(figsize=(12, 5))
        plt.plot(subset['Time'], subset['Power'], 'k^-', label="Measured Power")
        plt.plot(subset['Time'], subset['Predicted_Power'],
                 'b.-', label="Predicted Power (SVM)")
        plt.title(f"Location{self.site_index} - Measured vs Predicted Power (SVM)")
        plt.xlabel("Time")
        plt.ylabel("Power")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return fig

    def plot_lstm_result(self, model_funct, lookback_hours):
        """Train or load LSTM model and plot predictions vs actual values."""
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
            x_train, y_train = [], []
            for i in range(lookback_hours, len(train_df) - 1):
                window = train_df.iloc[i - lookback_hours:i][features]
                x_train.append(window.values)
                y_train.append(raw_train_df.iloc[i + 1]['Power'])

            x_train, y_train = np.array(x_train), np.array(y_train)

            if len(x_train) < 10:
                print("Not enough training data.")
                return None, None, None, None, None, None

            # Train model with early stopping
            model = model_funct(x_train[0].shape)
            model.fit(x_train, y_train, epochs=150, batch_size=32, verbose=0,
                    validation_split=0.1, shuffle=False,
                    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

            model.save(model_path)
            print("Model trained and saved.")
        else:
            model = load_model(model_path)
            print("Loaded existing LSTM model.")

        # Test Data Preparation
        x_test, y_test, times = [], [], []
        for i in range(lookback_hours, len(df_scaled) - 1):
            current_time = df_scaled.iloc[i + 1]['Time']
            if start_dt <= current_time <= end_dt:
                window = df_scaled.iloc[i - lookback_hours + 1:i + 1][features]
                x_test.append(window.values)
                y_test.append(site_df.iloc[i + 1]['Power'])
                times.append(current_time)

        if len(x_test) < 1:
            print("Not enough test data.")
            return None

        x_test, y_test = np.array(x_test), np.array(y_test)

        # Prediction
        predictions = model.predict(x_test).flatten()

        # Evaluation Metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)

        # Plotting
        fig = plt.figure(figsize=(14, 6))
        plt.plot(times, y_test, 'k^-', label='Measured Power')
        plt.plot(times, predictions, 'b.-', label='Predicted Power (LSTM)')
        plt.title(f"Location{self.site_index} - Predicted vs Measured Power (Neural Networks)")
        plt.xlabel('Time')
        plt.ylabel('Power')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return predictions, y_test, mse, mae, rmse, times, fig

    def plot_persistence_result(self, y_test, times):
        """
        Implement and plot persistence model (naive forecast).
        The persistence model uses the previous value as the prediction.
        """
        if len(y_test) < 2:
            print("Not enough data for persistence model.")
            return None

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
        plt.plot(times, y_persistence, label='Persistence Model',
                 color='blue', marker='o', linestyle='-')
        plt.title("Persistence Model vs Measured Power")
        plt.xlabel('Time')
        plt.ylabel('Power')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return mae, mse, rmse

    def compute_daily_capacity_factor(self, site_df):
        """Compute daily capacity factor from hourly power data."""
        site_df['Date'] = site_df['Time'].dt.date
        daily_cf = site_df.groupby('Date')['Power'].mean().reset_index()
        daily_cf.rename(columns={'Power': 'CapacityFactor'}, inplace=True)
        return daily_cf

    def create_lagged_cf_features(self, daily_cf, num_lags=10):
        """Create lag features from daily capacity factors."""
        df = daily_cf.copy()
        for lag in range(1, num_lags + 1):
            df[f'CF_t-{lag}'] = df['CapacityFactor'].shift(lag)
        df['Target'] = df['CapacityFactor'].shift(-1)
        df.dropna(inplace=True)
        feature_cols = [f'CF_t-{lag}' for lag in range(1, num_lags + 1)]
        return df, feature_cols

    def train_capacity_factor_model(self, num_lags):
        """Train or load a model to predict daily capacity factor using past N days as features."""

        # Setup paths
        folder_path = Path(__file__).parents[1] / "outputs"
        model_path = folder_path / f"Location{self.site_index}_cf_model.pkl"
        scaler_path = folder_path / f"Location{self.site_index}_cf_scaler.pkl"
        folder_path.mkdir(parents=True, exist_ok=True)

        # Load site data
        inputs_dir = Path(__file__).parents[1] / "inputs"
        site_df = load_and_filter_by_site(inputs_dir, self.site_index)

        # Compute daily capacity factor
        daily_cf = self.compute_daily_capacity_factor(site_df)

        # Create lag features
        df_lagged, feature_cols = self.create_lagged_cf_features(daily_cf, num_lags=num_lags)

        # Save lagged dataset for prediction
        self.daily_cf = daily_cf
        self.feature_cols = feature_cols
        self.num_lags = num_lags
        self.df_lagged = df_lagged

        # Prepare input x and output y
        x = df_lagged[feature_cols].values
        y = df_lagged['Target'].values

        # Train/Test split (80% training, 20% testing)
        split_index = int(len(x) * 0.8)
        self.x_train, self.x_test = x[:split_index], x[split_index:]
        self.y_train, self.y_test = y[:split_index], y[split_index:]

        if model_path.exists() and scaler_path.exists():
            print("💾 Loading existing CF model and scaler...")
            self.cf_model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        else:
            print("🛠️ Training new CF model with Grid Search...")

            # Scale features
            self.scaler = StandardScaler()
            self.x_train_scaled = self.scaler.fit_transform(self.x_train)
            self.x_test_scaled = self.scaler.transform(self.x_test)

            # Define SVR and grid search parameters
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.2],
                'gamma': ['scale', 'auto']
            }

            grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, n_jobs=-1, verbose=0)
            grid_search.fit(self.x_train_scaled, self.y_train)

            # Best model
            self.cf_model = grid_search.best_estimator_

            # Save model and scaler
            joblib.dump(self.cf_model, model_path)
            joblib.dump(self.scaler, scaler_path)
            print("✅ CF model (tuned) and scaler saved.")

        return self.cf_model, self.scaler

    def predict_capacity_factor(self):
        """Predict capacity factor for start_time based on past N days and trained model."""
        predict_date = pd.to_datetime(self.start_time).date()

        if predict_date not in self.df_lagged['Date'].values:
            print(f"⚠️ Date {predict_date} not found or insufficient past days for prediction.")
            return

        # Find feature row for this specific prediction date
        row = self.df_lagged[self.df_lagged['Date'] == predict_date]

        if row.empty:
            print(f"⚠️ Not enough historical data before {predict_date} to build features.")
            return

        x_pred = row[self.feature_cols].values
        x_pred_scaled = self.scaler.transform(x_pred)

        # Model predicts capacity factor
        y_pred = self.cf_model.predict(x_pred_scaled)[0]

        # Actual capacity factor for the target day
        actual_cf = row['Target'].values[0]

        # Compute percentual error
        percentual_error = abs((y_pred - actual_cf) / actual_cf) * 100

        print("\n=== 📈 Capacity Factor Forecast Result ===")
        print(f"Prediction Date: {predict_date}")
        print(f"Predicted CF:    {y_pred:.4f}")
        print(f"Actual CF:       {actual_cf:.4f}")
        print(f"Percentual Error: {percentual_error:.2f}%")
