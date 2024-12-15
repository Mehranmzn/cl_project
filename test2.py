import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

 
df1 = pd.read_csv("ts_features_dutch_calendar_dataset.csv")

# Ensure the date column is in datetime format
df1["TSDate"] = pd.to_datetime(df1["TSDate"])

# Ensure 'TSDate' is sorted and data is ordered by 'serieNames' and 'TSDate'
df1 = df1.sort_values(by=[ "TSDate"]).reset_index(drop=True)

#if df1 sales dat ais zero put them Nan

series = df1["serieNames"].unique()
print(f"Series: {series}")
# Define train and test sets
train = df1.iloc[:-30]
test = df1.iloc[-30:]

# Define features and target
features = [col for col in df1.columns if col.startswith("lag_") or col.startswith("rolling_") or col == "season"]
target = "sales"

# Prepare train and test data
X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

# LightGBM Dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# LightGBM parameters
params = {
   # "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
}

# Train LightGBM model
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, test_data],
    valid_names=["train", "test"],
    num_boost_round=500,
)

# Make predictions
y_pred = model.predict(X_test)
y_pred2 = model.predict(X_train)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE on test set: {rmse}")

# Unique series
# series = test["serieNames"].unique()
# print(f"Series: {series}")
# # Create subplots for each series
# fig, axes = plt.subplots(len(series), 1, figsize=(12, 6 * len(series)), sharex=True)

# for i, serie in enumerate(series):
#     ax = axes[i] if len(series) > 1 else axes  # Handle single subplot case

#     # Filter test data and predictions for the current series
#     test_serie = test[test["serieNames"] == serie]
#     y_test_serie = y_test[test["serieNames"] == serie]
#     y_pred_serie = pd.Series(y_pred, index=y_test.index)[test["serieNames"] == serie]

#     # Plot actual and predicted values
#     ax.plot(test_serie["TSDate"], y_test_serie, label="Actual", marker="o", color="blue")
#     ax.plot(test_serie["TSDate"], y_pred_serie, label="Predicted", marker="x", color="red")
#     ax.set_title(f"Predicted vs Actual Sales for {serie}")
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Sales")
#     ax.legend()
#     ax.grid()

# plt.tight_layout()
# plt.show()

# # Unique series
# series = df1["serieNames"].unique()

# # Create subplots for each series (Train + Test Panels)
# fig, axes = plt.subplots(len(series), 2, figsize=(12, 6 * len(series)), sharex=True)

# # Predict on first 200 rows of training data
# y_pred2 = model.predict(X_train.iloc[:200])

# for i, serie in enumerate(series):
#     # Train Panel
#     train_serie = train[train["serieNames"] == serie].iloc[:200]
#     y_train_serie = y_train[train["serieNames"] == serie].iloc[:200]
#     y_pred2_serie = pd.Series(y_pred2, index=y_train_serie.index)[train["serieNames"] == serie].iloc[:200]

#     # Test Panel
#     test_serie = test[test["serieNames"] == serie]
#     y_test_serie = y_test[test["serieNames"] == serie]
#     y_pred_serie = pd.Series(y_pred, index=y_test.index)[test["serieNames"] == serie]

#     # Plot Train Data
#     ax_train = axes[i, 0] if len(series) > 1 else axes[0]
#     ax_train.plot(train_serie["TSDate"], y_train_serie, label="Actual (Train)", marker="o", color="blue")
#     ax_train.plot(train_serie["TSDate"], y_pred2_serie, label="Predicted (Train)", marker="x", color="orange")
#     ax_train.set_title(f"Predicted vs Actual Sales (Train) for {serie}")
#     ax_train.set_xlabel("Date")
#     ax_train.set_ylabel("Sales")
#     ax_train.legend()
#     ax_train.grid()

#     # Plot Test Data
#     ax_test = axes[i, 1] if len(series) > 1 else axes[1]
#     ax_test.plot(test_serie["TSDate"], y_test_serie, label="Actual (Test)", marker="o", color="blue")
#     ax_test.plot(test_serie["TSDate"], y_pred_serie, label="Predicted (Test)", marker="x", color="red")
#     ax_test.set_title(f"Predicted vs Actual Sales (Test) for {serie}")
#     ax_test.set_xlabel("Date")
#     ax_test.set_ylabel("Sales")
#     ax_test.legend()
#     ax_test.grid()

# plt.tight_layout()
# plt.show()
