# Forecasting Project

This repository contains the implementation of a Proof of Concept (PoC) for short-term sales forecasting, designed as part of the Data Science Assignment.

## Project Overview

The goal of this project is to forecast short-term sales. Accurate sales forecasting allows for better resource allocation, efficient operations, and improved decision-making. For this PoC, forecasts are generated for two randomly selected time series, using historical sales data.

## Dataset

The dataset is divided into:

- **Training data**: Sales from `2013-06-21` to `2015-11-15`.
- **Test data**: Sales predictions for the period `2015-11-16` to `2015-11-30`.

### Variables:

- `TSDate` (YYYY-MM-DD): Date of observation.
- `serieNames`: Product series identifier (e.g., `serie_1`, `serie_2`).
- `sales`: Sales figures (integer). Available only in the training dataset.

> Note: The figures have been transformed from the original sales values.

## Deliverables

This project provides the following deliverables:

1. **Report**: A detailed explanation of the methodology, decisions, tests, and results.
2. **Graphical Outputs**: 
   - Visualizations of the model fit on the training data (`2013-06-21` to `2015-11-15`).
3. **15-Day Forecast**: Sales predictions from `2015-11-16` to `2015-11-30`.
4. **Codebase**: The complete implementation in this repository.

## Approach and Methodology

- **Modeling**: Various forecasting methods were considered. Final selection prioritized scalability and feasibility over model complexity.
- **Feature Engineering**: Additional input features were engineered to enhance model performance (details provided in the report).
- **Evaluation Metric**: Root Mean Squared Error (RMSE) was used to evaluate forecast accuracy.

### RMSE Formula:

\[
RMSE = \sqrt{\frac{\sum_{i=1}^{n}(S_i - \hat{S}_i)^2}{n}}
\]

Where:
- \( S_i \): Actual sales on day \( i \).
- \( \hat{S}_i \): Predicted sales on day \( i \).

## Scalability and Future Work

The PoC is designed to handle larger datasets, scaling up to 55,000 product time series. Suggestions for improvements include:
- Incorporating additional factors such as marketing activities, competitor pricing, and seasonality.
- Implementing parallelized or distributed processing for scalability.
- Experimenting with advanced forecasting techniques (e.g., deep learning models).

## Usage

Clone this repository:
   ```bash
   git clone https://github.com/Mehranmzn/cl_project.git
   ```


