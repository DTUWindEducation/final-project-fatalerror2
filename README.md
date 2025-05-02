[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zjSXGKeR)
# Our Great Package

Team: Fatal Error

## Overview

This project is a final assignment for the DTU course Scientific Programming for Wind Energy (46120). Our objective is to develop and compare different wind power forecasting models based on real-world data from 2017 to 2021. We implemented baseline models such as persistence model, and machine learning models including Support Vector Machines (SVM) and Long Short-Term Memory (LSTM) neural networks. Performance is evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

## Quick-start guide

1. Requirements:

* Python 3.X
* Libraries: `os`, `sys`, `pathlib`, `time`, `pandas`, `tensorflow`, `matplotlib`, `numpy`

2. Run the code:

* Set the `site_index` for the target location
* Assign the `start_time` and the `end_time` for the target time window

## Architecture

* `main.py`: Orchestrates loading data, training models, and generating plots.
* `src/__init__.py`: Contains utility functions for data loading, preprocessing, SVM setup, and LSTM model creation.
* `src/WindPowerForecaster.py`: Main forecasting class implementing methods for model training, evaluation, plotting, and daily CF prediction.
* `inputs/`: Folder containing real-world time series data for four different sites.
* `outputs/`: Folder where model and scaler are stored.

## Peer review

[ADD TEXT HERE!]


## Main.py Description

The `main.py` script serves as the central control for the entire wind power forecasting pipeline:

1. **Setup**: Initializes parameters and sets site/time configuration.
2. **Data Loading**: Loads and filters real-world data.
3. **Modeling**:

   * Trains/loads and evaluates **SVM**, **LSTM**, and **persistence** models.
   * Each model's results are plotted.

4. **Comparison**: Chooses the best model based on RMSE (with MAE tie-breaker).
5. **Extra Functionality**: Trains and predicts **daily capacity factor** using lagged SVM models.

![](arquitecture.svg)

## `src/__init__.py` Utility Functions

* **`load_site_data(site_index)`**: Loads data for a specific site.
* **`load_and_filter_by_site(inputs_dir, site_index)`**: Loads and adds hourly features.
* **`prepare_features(df, num_lags)`**: Prepares lagged features and labels for SVM.
* **`create_lstm_model(input_shape)`**: Builds and compiles an LSTM forecasting model.
* **`determine_winner(models_metrics)`**: Selects the best model by evaluation metric.

---

## `WindPowerForecaster` Class

The `WindPowerForecaster` class (in `src/WindPowerForecaster.py`) encapsulates all model workflows:

### Data

* **`filter_and_plot(inputs_dir, variable)`**: Filters and visualizes time series.
* **`split_train_test(x, y)`**: Splits data into training and test sets.

### Forecasting

* **`train_and_save_svm(num_lags)`**: Train or load a saved SVM model.
* **`plot_svm_result(num_lags)`**: Plot the SVM predictions vs actual values.
* **`plot_lstm_result(model_funct, lookback_hours)`**: Run and plots the LSTM predictions.
* **`plot_persistence_result(y_test, times)`**: Plot the persistence predictions.

### Evaluation

* **`print_evaluation_metrics(mae, mse, rmse)`**: Prints the error metrics.

### Capacity Factor Forecast

* **`train_capacity_factor_model(num_lags)`**: Builds or loads SVM for next-day capacity factor.
* **`predict_capacity_factor()`**: Predicts and compares daily capacity factor.

---
