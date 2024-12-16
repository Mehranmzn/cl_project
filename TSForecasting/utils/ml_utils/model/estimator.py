import os
import sys

from TSForecasting.constant.training_testing_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME
from TSForecasting.exception.exception import TSForecastingException
from TSForecasting.logging.logger import logging

class TSForecastingEstimator:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise TSForecastingException(e,sys)
    
    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise TSForecastingException(e,sys)