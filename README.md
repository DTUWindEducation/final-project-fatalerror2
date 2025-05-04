[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zjSXGKeR)
# Our Great Package

Team: Fatal Error

## Overview

This project is a final assignment for the DTU course Scientific Programming for Wind Energy (46120). Our objective is to develop and compare different wind power forecasting models based on real-world data from 2017 to 2021. We implemented baseline models such as persistence model, and machine learning models including Support Vector Machines (SVM) and Long Short-Term Memory (LSTM) neural networks. Performance is evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

## Quick-start guide

1. Requirements:

* Python version 3.12 or older. 
* Libraries: `os`, `sys`, `pathlib`, `time`, `pandas`, `tensorflow`, `matplotlib`, `numpy`, `scikit-learn`, `scipy`, `pytest`
* The repo contains an environment.yml file for creating a new environment if this is desired, with a compatible python version and dependencies. To create it, run the command 'conda env create -f environment.yml' in a terminal at the top level of the directory. 
* After creating a fresh environment, run the command "pip install -e ." to install all dependencies. 


2. Run the code:

* Set the `site_index` for the target location
* Assign the `start_time` and the `end_time` for the target time window
* To run the code from a terminal, run the command "python examples/main.py" using the compatible environment. If package is run using an IDE, run the main.py script. 


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

![](main.drawio.svg)


## Main.py Inputs Description

* site_index = integer from 1 to 4 for the desired location. 
* start_time = start date for the prediction model, i.e "2020-11-15". Make sure not to set a date too close to the         beginning of the dataset as it might not have enough to "look back" at predicting. 
* end_time = end date for the prediction model, i.e "2020-11-16".

* variable_name = name of variable of the data you wish to plot between start_time and end_time. It can be: 
temperature_2m, relativehumidity_2m, dewpoint_2m, windspeed_10m, windspeed_100m, winddirection_10m, winddirection_100m, windgusts_10m, Power. Refer to inputs folder for a description on these variables. 

* project_root = Path(__file__).resolve().parent.parent ; inputs_dir = Path(project_root / "inputs"). Directory for data. Do not change unless intended.

* num_lags_SVM = 5 # number of days to look back for the SVM Model. 
* lookback_hours = 18 # number of hours to look back for the LSTM Neural Network Model.
* num_lags_CT_prediction = 10 # number of days to look back for the SVM Model used in CF Prediction for next day. The model uses start_time as the date to predict. 


## `src/__init__.py` Utility Functions

* **`load_site_data(site_index)`**: Loads data for a specific site.
* **`load_and_filter_by_site(inputs_dir, site_index)`**: Loads and adds hourly features.
* **`prepare_features(df, num_lags)`**: Prepares lagged features and labels for SVM.
* **`create_lstm_model(input_shape)`**: Builds and compiles an LSTM forecasting model.
* **`determine_winner(models_metrics)`**: Selects the best model by evaluation metric.

---

## `WindPowerForecaster` Class

The `WindPowerForecaster` class (in `src/WindPowerForecaster.py`) encapsulates all model workflows:

![](WindPowerForecaster.drawio.svg)



### Data

* **`filter_and_plot(inputs_dir, variable)`**: Filters and visualizes time series.
* **`split_train_test(x, y)`**: Splits data into training and test sets.

### Forecasting

All predictive models are working on the principle of next-hour prediction. For each predicted hour in the plotted times series, the models are fed with the established look-back window features. 

* **`train_and_save_svm(num_lags)`**: Train or load a saved SVM model.
* **`plot_svm_result(num_lags)`**: Plot the SVM predictions vs actual values.
* **`plot_lstm_result(model_funct, lookback_hours)`**: Run and plots the LSTM predictions.
* **`plot_persistence_result(y_test, times)`**: Plot the persistence predictions.

### Evaluation 

Prints the Mean Absolute Error (MAE), Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) for the models.

* **`print_evaluation_metrics(mae, mse, rmse)`**: Prints the error metrics.


### Determine Winner - EXTRA FUNCTIONALITY

Selects the best-performing model based on error metrics (MAE, MSE, RMSE). If no single model is best across all three, it defaults to the one with the lowest RMSE, using MAE as a tie-breaker if RMSEs are very close. 


### Capacity Factor Forecast - EXTRA FUNCTIONALITY

For predicting next-day Capacity Factor based on the capacity factors of the look-back window established. Only the capacity factor is used as feature. 

* **`train_capacity_factor_model(num_lags)`**: Builds or loads SVM for next-day capacity factor.
* **`predict_capacity_factor()`**: Predicts and compares daily capacity factor.

---
