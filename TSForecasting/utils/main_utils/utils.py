import yaml
import os,sys
import numpy as np  
import pandas as pd
from TSForecasting.exception.exception import TSForecastingException
from TSForecasting.logging.logger import logging
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from datetime import timedelta
from TSForecasting.constant.training_testing_pipeline import DATA_LAG, DATA_WINDOW, TARGET_COLUMN, DATA_GROUPING_COLUMN


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise TSForecastingException(e, sys)
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise TSForecastingException(e, sys)
    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise TSForecastingException(e, sys) from e
    
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise TSForecastingException(e, sys) from e
    
def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise TSForecastingException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise TSForecastingException(e, sys) from e
    


def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = np.sqrt(mean_squared_error(y_train, y_train_pred))

            test_model_score = np.sqrt(mean_squared_error(y_test, y_test_pred))

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise TSForecastingException(e, sys)
    


class FeatureEngineering:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def ensure_datetime(self, date_column: str):
        self.dataframe[date_column] = pd.to_datetime(self.dataframe[date_column])

    def generate_date_features(self, date_column: str):
        self.dataframe["year"] = self.dataframe[date_column].dt.year
        self.dataframe["month"] = self.dataframe[date_column].dt.month
        self.dataframe["day"] = self.dataframe[date_column].dt.day
        self.dataframe["weekday_num"] = self.dataframe[date_column].dt.weekday  # Monday=0, Sunday=6
        self.dataframe["is_weekend"] = self.dataframe["weekday_num"] >= 5

    def get_easter(self, year):
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        return pd.Timestamp(year, month, day)

    def dutch_calendar_events(self, date):
        year = date.year
        easter = self.get_easter(year)
        events = {
            "Liberation Day": date.month == 5 and date.day == 5,
            "Valentine's Day": date.month == 2 and date.day == 14,
            "Easter": date == easter,
            "Easter Monday": date == (easter + timedelta(days=1)),
            "Christmas": date.month == 12 and date.day in [25, 26],
        }
        for event, condition in events.items():
            if condition:
                return event
        return None

    def apply_dutch_calendar(self, date_column: str):
        self.dataframe["dutch_event"] = self.dataframe[date_column].apply(self.dutch_calendar_events)

    def add_season_feature(self):
        def get_season(month):
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            elif month in [9, 10, 11]:
                return "Autumn"
        self.dataframe["season"] = self.dataframe["month"].apply(get_season)

    def add_lag_features(self, group_column: str = DATA_GROUPING_COLUMN, target_column: str = TARGET_COLUMN, lags: int = DATA_LAG):
        for lag in range(1, lags + 1):
            self.dataframe[f"lag_{lag}"] = self.dataframe.groupby(group_column)[target_column].shift(lag)

    def add_rolling_features(self, group_column: str = DATA_GROUPING_COLUMN, target_column: str = TARGET_COLUMN, window: int = DATA_WINDOW):
        self.dataframe["rolling_min"] = self.dataframe.groupby(group_column)[target_column].transform(
            lambda x: x.rolling(window=window).min())
        self.dataframe["rolling_max"] = self.dataframe.groupby(group_column)[target_column].transform(
            lambda x: x.rolling(window=window).max())
        self.dataframe["rolling_std"] = self.dataframe.groupby(group_column)[target_column].transform(
            lambda x: x.rolling(window=window).std())

    def generate_features(self, date_column: str, group_column: str, target_column: str = TARGET_COLUMN):
        self.ensure_datetime(date_column)
        self.generate_date_features(date_column)
        self.apply_dutch_calendar(date_column)
        self.add_season_feature()
        self.add_lag_features(group_column, target_column)
        self.add_rolling_features(group_column, target_column)

    def get_dataframe(self):
        return self.dataframe