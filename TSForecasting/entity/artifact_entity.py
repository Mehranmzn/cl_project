from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trained_file_path:str
    val_file_path:str
    test_file_path:str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str
    valid_val_file_path: str
    invalid_val_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_val_file_path: str

@dataclass
class RegressionMetricArtifact:
    rmse: float

    
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: RegressionMetricArtifact
    val_metric_artifact: RegressionMetricArtifact
