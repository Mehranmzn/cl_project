from datetime import datetime
import os
from TSForecasting.constant import training_testing_pipeline

print(training_testing_pipeline.PIPELINE_NAME)
print(training_testing_pipeline.ARTIFACT_DIR)


class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        if timestamp is None:
            timestamp = datetime.now()
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")

        self.pipeline_name = training_testing_pipeline.PIPELINE_NAME
        self.artifact_name = training_testing_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        self.model_dir = os.path.join("final_model")
        self.timestamp: str = timestamp




class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir:str=os.path.join(
            training_pipeline_config.artifact_dir,training_testing_pipeline.DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path: str = os.path.join(
                self.data_ingestion_dir, training_testing_pipeline.DATA_INGESTION_FEATURE_STORE_DIR, training_testing_pipeline.FILE_NAME
            )
        self.training_file_path: str = os.path.join(
                self.data_ingestion_dir, training_testing_pipeline.DATA_INGESTION_INGESTED_DIR, training_testing_pipeline.TRAIN_FILE_NAME
            )
        self.testing_file_path: str = os.path.join(
                self.data_ingestion_dir, training_testing_pipeline.DATA_INGESTION_INGESTED_DIR, training_testing_pipeline.TEST_FILE_NAME
            )
        self.val_file_path: str = os.path.join(
                self.data_ingestion_dir, training_testing_pipeline.DATA_INGESTION_INGESTED_DIR, training_testing_pipeline.VAL_FILE_NAME
            )
        self.train_test_split_ratio: float = training_testing_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        self.collection_name: str = training_testing_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = training_testing_pipeline.DATA_INGESTION_DATABASE_NAME
        self.table_name: str = training_testing_pipeline.DATA_INGESTION_TABLE_NAME
        self.train_start_date: str = training_testing_pipeline.TRAIN_START_DATE
        self.train_end_date: str = training_testing_pipeline.TRAIN_END_DATE
        self.val_start_date: str = training_testing_pipeline.VAL_START_DATE
        self.val_end_date: str = training_testing_pipeline.VAL_END_DATE
        self.test_start_date: str = training_testing_pipeline.TEST_START_DATE
        self.test_end_date: str = training_testing_pipeline.TEST_END_DATE

class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join( training_pipeline_config.artifact_dir, training_testing_pipeline.DATA_VALIDATION_DIR_NAME)
        self.valid_data_dir: str = os.path.join(self.data_validation_dir, training_testing_pipeline.DATA_VALIDATION_VALID_DIR)
        self.invalid_data_dir: str = os.path.join(self.data_validation_dir, training_testing_pipeline.DATA_VALIDATION_INVALID_DIR)
        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, training_testing_pipeline.TRAIN_FILE_NAME)
        self.valid_test_file_path: str = os.path.join(self.valid_data_dir, training_testing_pipeline.TEST_FILE_NAME)
        self.invalid_train_file_path: str = os.path.join(self.invalid_data_dir, training_testing_pipeline.TRAIN_FILE_NAME)
        self.invalid_test_file_path: str = os.path.join(self.invalid_data_dir, training_testing_pipeline.TEST_FILE_NAME)
        self.valid_val_file_path: str = os.path.join(self.valid_data_dir, training_testing_pipeline.VAL_FILE_NAME)
        self.invalid_val_file_path: str = os.path.join(self.invalid_data_dir, training_testing_pipeline.VAL_FILE_NAME)
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_testing_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_testing_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
            
        )


class DataTransformationConfig:
     def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join( training_pipeline_config.artifact_dir,training_testing_pipeline.DATA_TRANSFORMATION_DIR_NAME )
        self.transformed_train_file_path: str = os.path.join( self.data_transformation_dir,training_testing_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_testing_pipeline.TRAIN_FILE_NAME.replace("csv", "npy"),)
        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir,  training_testing_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_testing_pipeline.TEST_FILE_NAME.replace("csv", "npy"), )
        self.transformed_val_file_path: str = os.path.join(self.data_transformation_dir, training_testing_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_testing_pipeline.VAL_FILE_NAME.replace("csv", "npy"),)
        self.transformed_object_file_path: str = os.path.join( self.data_transformation_dir, training_testing_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_testing_pipeline.PREPROCESSING_OBJECT_FILE_NAME,)
        
class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_testing_pipeline.MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, training_testing_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR, 
            training_testing_pipeline.MODEL_FILE_NAME
        )
        self.expected_accuracy: float = training_testing_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold = training_testing_pipeline.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD




