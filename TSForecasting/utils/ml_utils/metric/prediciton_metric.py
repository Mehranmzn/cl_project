from TSForecasting.entity.artifact_entity import RegressionMetricArtifact
from TSForecasting.exception.exception import TSForecastingException
from sklearn.metrics import mean_squared_error
import sys
import numpy as np

def get_regression_score(y_true, y_pred) -> RegressionMetricArtifact:
    """
    Calculate RMSE and return a RegressionMetricArtifact object.
    """
    try:
        # Calculate RMSE
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        # Create a RegressionMetricArtifact object (assuming it exists in your codebase)
        regression_metric = RegressionMetricArtifact(rmse=rmse)
        return regression_metric

    except Exception as e:
        raise TSForecastingException(e, sys)
