import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
df1 = pd.read_csv("ts_features_dutch_calendar_dataset.csv")

# Ensure the date column is in datetime format
df1["TSDate"] = pd.to_datetime(df1["TSDate"])

# Ensure 'TSDate' is sorted and data is ordered by 'serieNames' and 'TSDate'
df1 = df1.sort_values(by=["TSDate"]).reset_index(drop=True)

# Replace zero sales with NaN
#df1["sales"] = df1["sales"].replace(0, np.nan)

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

# LightGBM
print("Training LightGBM...")
lgb_params = {
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
lgb_model = lgb.train(
    lgb_params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=500
)
lgb_pred = lgb_model.predict(X_test)
lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_pred))
print(f"LightGBM RMSE: {lgb_rmse}")

# XGBoost
print("Training XGBoost...")
xgb_params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 500,
}
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
print(f"XGBoost RMSE: {xgb_rmse}")

# CatBoost
print("Training CatBoost...")
cat_model = cb.CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="RMSE",
    verbose=False
)
cat_model.fit(X_train, y_train)
cat_pred = cat_model.predict(X_test)
cat_rmse = np.sqrt(mean_squared_error(y_test, cat_pred))
print(f"CatBoost RMSE: {cat_rmse}")

# Compare results
print("\nModel Comparison:")
print(f"LightGBM RMSE: {lgb_rmse}")
print(f"XGBoost RMSE: {xgb_rmse}")
print(f"CatBoost RMSE: {cat_rmse}")

# Plot predictions
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(test["TSDate"], y_test, label="Actual", marker="o", color="blue")
ax.plot(test["TSDate"], lgb_pred, label="LightGBM Predicted", marker="x", color="green")
ax.plot(test["TSDate"], xgb_pred, label="XGBoost Predicted", marker="x", color="orange")
ax.plot(test["TSDate"], cat_pred, label="CatBoost Predicted", marker="x", color="red")
ax.set_title("Predicted vs Actual Sales")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()
ax.grid()
plt.tight_layout()
plt.show()


print(df1.dtypes    )