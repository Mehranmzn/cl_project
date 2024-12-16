import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from TSForecasting.constant.training_testing_pipeline import TARGET_COLUMN, SCHEMA_FILE_PATH
from TSForecasting.constant.training_testing_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from TSForecasting.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)
from TSForecasting.entity.config_entity import DataTransformationConfig
from TSForecasting.exception.exception import TSForecastingException 
from TSForecasting.logging.logger import logging
from TSForecasting.utils.main_utils.utils import save_numpy_array_data,save_object, read_yaml_file



class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise TSForecastingException(e,sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise TSForecastingException(e, sys)
        
    def get_data_transformer_object(cls) -> Pipeline:
        """
        Initialize the data transformer pipeline with KNNImputer and OneHotEncoder for categorical columns.
        """
        logging.info(
            "Entered get_data_transformer_object method of Transformation class"
        )
        try:
            # Initialize the KNNImputer
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(
                f"Initialized KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}"
            )

            # OneHotEncoder for categorical columns
            #categorical_columns = []

            schema = read_yaml_file(SCHEMA_FILE_PATH)

            numerical_columns = [
                column["name"]
                for column in schema["columns"]
                if column["type"] in ["INTEGER", "FLOAT"]
            ]
                                 
            #categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

            # Combine the transformations using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[

                    ("imputer", imputer, numerical_columns),  # Apply imputer to all numeric columns
                ],
                remainder="passthrough"  # Keep the remaining columns as they are
            )

            # Create the pipeline
            processor = Pipeline([("preprocessor", preprocessor)])
            return processor

        except Exception as e:
            raise TSForecastingException(e, sys)

        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            val_df = DataTransformation.read_data(self.data_validation_artifact.valid_val_file_path)

            schema = read_yaml_file(SCHEMA_FILE_PATH)
            numerical_columns = [
                column["name"]
                for column in schema["columns"]
                if column["type"] in ["INTEGER", "FLOAT"]
            ]


            # Ensure the target column is removed from the final numerical_columns for transformation
            numerical_columns_for_transform = [col for col in numerical_columns if col != TARGET_COLUMN]

            # Get the preprocessor object
            preprocessor = self.get_data_transformer_object()

            # Fit the preprocessor on numerical columns of the training data (including TARGET_COLUMN temporarily)
            preprocessor_object = preprocessor.fit(train_df[numerical_columns])

            # Transform numerical columns (excluding TARGET_COLUMN) in the input feature DataFrames
            transformed_input_train_feature = preprocessor_object.transform(train_df[numerical_columns])[:, :len(numerical_columns_for_transform)]
            transformed_input_test_feature = preprocessor_object.transform(test_df[numerical_columns])[:, :len(numerical_columns_for_transform)]
            transformed_input_val_feature = preprocessor_object.transform(val_df[numerical_columns])[:, :len(numerical_columns_for_transform)]

            # Drop the target column to form input feature DataFrames
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            input_feature_val_df = val_df.drop(columns=[TARGET_COLUMN], axis=1)

            # Replace numerical columns with transformed data
            input_feature_train_df[numerical_columns_for_transform] = transformed_input_train_feature
            input_feature_test_df[numerical_columns_for_transform] = transformed_input_test_feature
            input_feature_val_df[numerical_columns_for_transform] = transformed_input_val_feature

            # Extract target feature DataFrames
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_val_df = val_df[TARGET_COLUMN]

            # Combine the features and target columns into arrays
            train_arr = np.c_[input_feature_train_df.to_numpy(), target_feature_train_df.to_numpy()]
            test_arr = np.c_[input_feature_test_df.to_numpy(), target_feature_test_df.to_numpy()]
            val_arr = np.c_[input_feature_val_df.to_numpy(), target_feature_val_df.to_numpy()]
            
           
           

            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr,)
            save_numpy_array_data( self.data_transformation_config.transformed_val_file_path,array=val_arr,)
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object,)

            save_object( "final_model/preprocessor.pkl", preprocessor_object,)


            #preparing artifacts

            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_val_file_path=self.data_transformation_config.transformed_val_file_path
            )
            return data_transformation_artifact


            
        except Exception as e:
            raise TSForecastingException(e,sys)
