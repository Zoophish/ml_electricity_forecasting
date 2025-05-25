# Electricity Price Forecasting Models

This repository contains the implementation of my thesis **"Machine Learning for Optimisation of Electricity Trading in a Real-World Microgrid"**   

## Repository Structure:

- `dataset/`   
    - CSVs for publically available data   
    - processing scripts   
- `features/`   
    - TSFEL feature config files for feature enrichment   
    - Time series seature visualiser   
- `models/`   
    - `day_ahead_bench.py`: Training pipeline for day-ahead Torch models   
    - `intraday_bench.py`: Training pipeline for intraday/imbalance forecasts


## Installation

This repo contains scripts rather than an installable package, but requires several other packages listed below.

### Requirements

This project requires [timemachine](https://github.com/Zoophish/timemachine) to be installed.

**pip Packages**:
```
torch
numpy
polars
pandas
sklearn
tslearn
tsfel
matplotlib
elexonpy
```


## Training & Running Models

The EPEX and Elexon price time series csv files (`market_data.csv` and `market_data_types.csv`) have been omitted from the repository and need manually adding into `/data`.   

You can then run the model pipeline scripts in `/models`. These can be configured to train/load the model and run an interactive forecasting plotter.


## Overview of Project

This project investigated the usage of various ML model architectures on forecasting prices of different electricity markets in Great Britain.

Several models were benchmarked on the EPEX GB day-ahead price profile: MLP, LSTM, BiLSTM, CNNLSTM, SARIMA.

An uncertainty-based Bayesian approach inspired by Monte Carlo dropout (Gal & Ghahramani, 2016) was applied to the EPEX continuous intraday & Elexon imbalance prices.

### Results

#### Day-Ahead Errors

The hyperparameters of each model were optimised using the Bayesian optimiser from Optuna. The best error metrics for each model are given below.   

| Model   | Dataset | MSE  | MAE (£) | MAPE   |
|---------|---------|------|-------- |--------|
| BiLSTM  | I       | 2.79 | 1.18    | 59.37% |
| CNNLSTM | I       | 3.85 | 1.34    | 83.41% |
| LSTM    | I       | 3.75 | 1.35    | 67.4%  |
| MLP     | I       | 4.52 | 1.46    | 119%   |
| SARIMA  | II      | 6.46 | 1.63    | 120%   |

The two-tailed Diebold-Mariano hypothesis test results for significant forecasting performance is given below.

| Model A | Model B | DM Statistic | p Value     | Better Model   |
|---------|---------|--------------|-------------|----------------|
| BiLSTM  | CNNLSTM | 0.0943       | 0.536       | *Insignificant* |
| BiLSTM  | LSTM    | -2.43        | 0.00758     | BiLSTM         |
| BiLSTM  | MLP     | -6.66        | 1.56e-11    | BiLSTM         |
| BiLSTM  | SARIMA  | -5.44        | 5.26e-8     | BiLSTM         |
| CNNLSTM | LSTM    | -21.7        | 0.0         | CNNLSTM        |
| CNNLSTM | MLP     | -16.1        | 0.0         | CNNLSTM        |
| CNNLSTM | SARIMA  | -12.3        | 0.0         | CNNLSTM        |
| LSTM    | MLP     | -10.5        | 0.0         | LSTM           |
| LSTM    | SARIMA  | -6.40        | 1.59e-10    | LSTM           |

#### Example Day-Ahead Forecasts

![](./repo_resources/BiLSTM%20Example%20Forecasts.svg)

#### Intraday & Imbalance Errors

The mean Gaussian negative log-likelihood and absolute error for the intraday and imbalance price forecasts are given below.

| Model         | Dataset | Target                           | NLL    | MAE (£)  |
|---------------|---------|----------------------------------|--------|----------|
| MCD MLP       | III     | EPEX Intraday Continuous Price   | -1.31  | 1.12     |
| MCD MLP       | III     | ELEXON Imbalance Price           | -0.824 | 2.19     |

#### Example Intraday Forecasts

![](./repo_resources/Intraday%20Example%20Forecasts.svg)