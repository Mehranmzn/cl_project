from TSForecasting.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from TSForecasting.entity.config_entity import DataValidationConfig
from TSForecasting.exception.exception import TSForecastingException 
from TSForecasting.logging.logger import logging 
from TSForecasting.constant.training_testing_pipeline import SCHEMA_FILE_PATH
from TSForecasting.utils.main_utils.utils import FeatureEngineering
from scipy.stats import ks_2samp
import pandas as pd
import os,sys
from TSForecasting.utils.main_utils.utils import read_yaml_file,write_yaml_file

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise TSForecastingException(e,sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise TSForecastingException(e,sys)
        
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            required_columns = self._schema_config
            missing_columns = [col for col in required_columns if col not in dataframe.columns]

            if missing_columns:
                logging.warning(f"Missing columns: {missing_columns}")
                # Perform feature engineering to add the missing columns
                feature_engineering = FeatureEngineering(dataframe)
                feature_engineering.generate_features(
                    date_column="TSDate",
                    group_column="serieNames",
                    target_column="sales"
                )
                dataframe = feature_engineering.get_dataframe()
                logging.info(f"Added missing columns through feature engineering.")

            # Validate again
            if len(dataframe.columns) == len(required_columns):
                return True
            return False
        except Exception as e:
            raise TSForecastingException(e, sys)
        
    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
        try:
            status=True
            report={}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                is_same_dist=ks_2samp(d1,d2)
                if threshold<=is_same_dist.pvalue:
                    is_found=False
                else:
                    is_found=True
                    status=False
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                    
                    }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            #Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report)

        except Exception as e:
            raise TSForecastingException(e,sys)
        
    
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path=self.data_ingestion_artifact.trained_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path
            val_file_path=self.data_ingestion_artifact.val_file_path

            ## read the data from train and test
            train_dataframe=DataValidation.read_data(train_file_path)
            test_dataframe=DataValidation.read_data(test_file_path)
            validation_dataframe=DataValidation.read_data(val_file_path)
            
            ## validate number of columns

            status=self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message=f"Train dataframe does not contain all columns.\n"
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message=f"Test dataframe does not contain all columns.\n"   
            status = self.validate_number_of_columns(dataframe=validation_dataframe)
            if not status:
                error_message=f"Validation dataframe does not contain all columns.\n"

            ## lets check datadrift
            status=self.detect_dataset_drift(base_df=train_dataframe,current_df=validation_dataframe)
            dir_path=os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True

            )

            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )

            validation_dataframe.to_csv(
                self.data_validation_config.valid_val_file_path, index=False, header=True
            )
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
                valid_val_file_path=self.data_ingestion_artifact.val_file_path,
                invalid_val_file_path=None,
            )
            return data_validation_artifact
        except Exception as e:
            raise TSForecastingException(e,sys)



