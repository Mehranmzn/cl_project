import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from TSForecasting.exception.exception import TSForecastingException 
from TSForecasting.logging.logger import logging
from TSForecasting.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from TSForecasting.entity.config_entity import ModelTrainerConfig
from TSForecasting.utils.ml_utils.model.estimator import TSForecastingEstimator
from TSForecasting.utils.main_utils.utils import save_object,load_object
from TSForecasting.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from TSForecasting.utils.ml_utils.metric.prediciton_metric import get_regression_score
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import mlflow
from urllib.parse import urlparse

import dagshub
dagshub.init(repo_owner='mehran1414', repo_name='cl_project', mlflow=True)





class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise TSForecastingException(e,sys)
        
    def track_mlflow(self, best_model, model_name, regressionmetric, X_train, y_true, y_pred, run_type):
        try:
            mlflow.set_registry_uri("https://dagshub.com/mehran1414/cl_project.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                # Log model metrics
                mlflow.log_metric("rmse", regressionmetric.rmse)

                # Log additional parameters or tags for clarity
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("run_type", run_type)  # "training" or "validation"
                mlflow.log_param("num_samples", len(X_train))
                mlflow.set_tag("version", "9")
                mlflow.set_tag("data_split", run_type)  # Distinguish training vs validation

                # Log example input for signature
                input_example = X_train[0].reshape(1, -1)
                mlflow.sklearn.log_model(
                    best_model, 
                    model_name, 
                    input_example=input_example
                )

                # Register the model if not using a file store
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(
                        best_model, 
                        model_name, 
                        registered_model_name="best_model_name"
                    )

                # Log predictions as an artifact (optional but useful)
                predictions_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
                predictions_csv = "/tmp/predictions.csv"  # Temporary local path
                predictions_df.to_csv(predictions_csv, index=False)
                mlflow.log_artifact(predictions_csv, artifact_path=f"predictions/{run_type}")


        except Exception as e:
            raise Exception(f"Error in MLflow tracking: {e}")



        
    def train_model(self,X_train,y_train, x_val, y_val):
        models = {

                "XGBoost":  xgb.XGBRegressor(objective='reg:squarederror'),
                "CatBoost": cb.CatBoostRegressor(verbose=0),
                "LightGBM": lgb.LGBMRegressor()
            }
        params = {
                        
                "XGBoost": {
                    'learning_rate': [0.05, 0.1],  # Learning rate
                    'max_depth': [6, 9],  # Tree depth
                    'n_estimators': [10, 50],  # Number of boosting rounds
                    'subsample': [0.8, 1.0],  # Fraction of data used for training
                    'alpha': [0.5, 0.7],  # L1 regularization
                },
                "CatBoost": {
                    'learning_rate': [0.05, 0.1],  # Learning rate
                    'depth': [6, 8],  # Tree depth
                    'iterations': [10, 50],  # Number of iterations
                },
                "LightGBM": {
                    'learning_rate': [0.05, 0.1],  # Learning rate
                    'min_data_in_leaf': [10, 20],  # Minimum samples in a leaf node
                    'num_leaves': [5, 7],  # Number of leaves
                    'max_depth': [6, 12],  # Maximum tree depth
                    'n_estimators': [10, 50],  # Number of boosting rounds
                    'lambda_l1': [0.5, 0.7],  # L1 regularization
                }
        }

                


        model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=x_val,y_test=y_val,
                                          models=models,param=params)
        
        ## To get best model score from dict
        best_model_score = max(sorted(model_report.values()))

        ## To get best model name from dict

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]


        best_model = models[best_model_name]

        best_model.fit(X_train, y_train)


        y_train_pred=best_model.predict(X_train)
            
        regression_train_metric=get_regression_score(y_true=y_train,y_pred=y_train_pred)
        
        ## Track the experiements with mlflow
        self.track_mlflow(best_model, best_model_name, regression_train_metric, X_train, y_train, y_train_pred, run_type="training")


        y_val_pred=best_model.predict(x_val)
        regression_val_metric=get_regression_score(y_true=y_val,y_pred=y_val_pred)

        self.track_mlflow(best_model, best_model_name, regression_val_metric, x_val, y_val, y_val_pred, run_type="validation")

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        TM_Model=TSForecastingEstimator(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=TM_Model)
        #model pusher
        save_object("final_model/model.pkl",best_model)
        

        ## Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=regression_train_metric,
                             val_metric_artifact=regression_val_metric
                             )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    def prepare_the_test_csv_file(self, x_test, y_test):
        try:
            # Load the pre-trained model
            chosen_model = load_object("final_model/model.pkl")
            
        
            # Recursive prediction
            y_test_pred = []  # Store predictions
            x_test_recursive = x_test.copy()  # Create a copy to update inputs

            for i in range(len(y_test)):
                # Predict for the current step
                pred = chosen_model.predict(x_test_recursive[i].reshape(1, -1))[0]
                y_test_pred.append(pred)
                
                # Update lag features in x_test_recursive with the current prediction
                if i + 1 < len(x_test_recursive):  # Avoid out-of-bound errors
                    # Shift lag features to include the new prediction
                    x_test_recursive[i + 1, -1] = pred  # Update the last lag feature


            dates = []
            base_date = datetime(2015, 11, 16)
            for i in range(len(y_test_pred) // 2):
                dates.extend([base_date + timedelta(days=i)] * 2)

            # Ensure we have enough dates for all predictions
            if len(dates) < len(y_test_pred):
                remaining = len(y_test_pred) - len(dates)
                last_date = dates[-1] if dates else base_date
                dates.extend([last_date + timedelta(days=1)] * remaining)

            # Prepare DataFrame
            df = pd.DataFrame({
                'date': dates,
                'x': list(x_test_recursive),
                'y': y_test,
                'y_pred': y_test_pred
            })

            # Save the DataFrame to a CSV file
            df.to_csv("testoutput/testoutput.csv", index=False)
            
            print(f"TEST PREDICTIONS CSV file saved ")
        
        except Exception as e:
            raise Exception(f"Error in preparing the test CSV file: {e}")
                
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            val_file_path = self.data_transformation_artifact.transformed_val_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)
            val_arr = load_numpy_array_data(val_file_path)

            x_train, y_train, x_val, y_val, x_test, y_test = (
                train_arr[:, 1:-1],
                train_arr[:, -1],
                val_arr[:, 1:-1],
                val_arr[:, -1],
                test_arr[:, 1:-1],
                test_arr[:, -1],
            )


            try:
                model_trainer_artifact = self.train_model(x_train, y_train, x_val, y_val)
                print("train_model executed successfully")
                print("Now we go to save the test file preds")
                #self.prepare_the_test_csv_file(x_test, y_test)
                print("Test file preds saved successfully")
            except Exception as e:
                print(f"Exception in train_model: {e}")
                raise
            return model_trainer_artifact

            
        except Exception as e:
            raise TSForecastingException(e,sys)