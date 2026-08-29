# Hurricane-Weather-Forecasting
Multivariate hurricane weather forecasting using a PyTorch multilayer perceptron (MLP).

# Project Overview
This project develops a machine-learning model to forecast future hurricane-related atmospheric conditions using historical meteorological observations. A 14-day historical observation window is used to forecast selected atmospheric conditions 7 days ahead. The model is designed as a multivariate regression problem using a PyTorch-based Multilayer Perceptron.

# Input Features
The model uses seven weather-related input features:
- Rainfall
- Wind speed
- Surface air pressure
- Downward short-wave radiation
- Downward long-wave radiation
- Wind direction sine
- Wind direction cosine
Wind direction is represented using sine and cosine transformations to preserve the circular nature of directional data.

# Model Architecture
The forecasting model is a Multilayer Perceptron (MLP) consisting of:
- Input layer: 98 features
- Hidden layers: 512, 256, 128 and 64 neurons
- ReLU activation
- Dropout: 0.3
- Output layer: 5 variables
- MSE loss
- Weight decay
- Early stopping
- ReduceLROnPlateau learning-rate scheduling

# Data Preprocessing
The preprocessing pipeline includes:
- Daily aggregation of three-hourly observations
- Merging weather variables by date and location
- Handling missing wind-direction values
- Log transformation of rainfall
- Z-score standardisation
- Outlier clipping
- Chronological dataset splitting

# Evaluation
The data was divided chronologically into training, validation and testing periods to reduce the risk of future information leakage.
The final model was evaluated on an unseen test dataset using:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

