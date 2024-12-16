import os
import sys
import numpy as np
import pandas as pd


"""
define the constant for the training pipeline
"""
TARGET_COLUMN = "sales"
PIPELINE_NAME: str = "TSForecasting"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "TSdata/training.csv"

TRAIN_FILE_NAME: str = "train.csv"
VAL_FILE_NAME: str = "validation.csv"
TEST_FILE_NAME: str = "test.csv"


SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yml")

SAVED_MODEL_DIR =os.path.join("saved_models")
MODEL_FILE_NAME = "model.pkl"
FEATURE_STORE_DIR: str = "feature_store"



"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "SALES"
DATA_INGESTION_DATABASE_NAME: str = "CLBLUE"
DATA_INGESTION_TABLE_NAME: str = "SALES_TS"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2
DATA_LAG: int = 7
DATA_WINDOW: int = 7
DATA_GROUPING_COLUMN: str = "serieNames"
DATA_DATE_COLUMN: str = "TSDate"

"""
Data Split related constant start and end
"""

TRAIN_START_DATE = "2013-06-21"
TRAIN_END_DATE = "2015-10-30"
VAL_START_DATE = "2015-10-31"
VAL_END_DATE = "2015-11-15"
TEST_START_DATE = "2015-11-16"
TEST_END_DATE = "2015-11-30"

"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

## kkn imputer to replace nan values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}
DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"

DATA_TRANSFORMATION_TEST_FILE_PATH: str = "validation.npy"


"""
Model Trainer ralated constant start with MODE TRAINER VAR NAME
"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05

TRAINING_BUCKET_NAME = "tsforecasting-training-data"