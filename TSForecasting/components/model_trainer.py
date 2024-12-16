import os
import sys
from TSForecasting.exception.exception import TSForecastingException 
from TSForecasting.logging.logger import logging
from TSForecasting.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from TSForecasting.entity.config_entity import ModelTrainerConfig
from TSForecasting.utils.ml_utils.model.estimator import TSForecastingEstimator
from TSForecasting.utils.main_utils.utils import save_object,load_object
from TSForecasting.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from TSForecasting.utils.ml_utils.metric.prediciton_metric import get_regression_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
import catboost as CatBoostRegressor
import lightgbm as LGBMRegressor

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
        
    def track_mlflow(self,best_model,classificationmetric):
        mlflow.set_registry_uri("https://dagshub.com/mehran1414/tm_data.mlflow")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            f1_score=classificationmetric.f1_score
            precision_score=classificationmetric.precision_score
            recall_score=classificationmetric.recall_score

            

            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision",precision_score)
            mlflow.log_metric("recall_score",recall_score)
            mlflow.sklearn.log_model(best_model,"model")
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model)
            else:
                mlflow.sklearn.log_model(best_model, "model")


        
    def train_model(self,X_train,y_train,x_test,y_test):
        models = {
                "Random Forest": RandomForestRegressor(verbose=1),
                "Gradient Boosting": GradientBoostingRegressor(verbose=1),
                "AdaBoost": AdaBoostRegressor(),
                "CatBoost": CatBoostRegressor(),
                "Lightgbm Regression": LGBMRegressor(),

            }
        params = {
               
                "Random Forest": {
                    'n_estimators': [50, 100],  # Fewer trees for faster execution
                    'max_depth': [5, 10],  # Balanced complexity
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt', None],  # Subset of features per split
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100],  # Moderate number of estimators for speed
                    'learning_rate': [0.05, 0.1],  # Balanced learning rates
                    'max_depth': [3, 5],  # Shallow trees to avoid overfitting
                    'subsample': [0.7, 0.9],  # Introduce randomness to improve generalization
                    'min_samples_split': [5, 10],
                    'min_samples_leaf': [2, 5],
                },
                "AdaBoost": {
                    'n_estimators': [50, 100],  # Iterations for local execution
                    'learning_rate': [0.05, 0.1],  # Slow learning rates for stability
                    'loss': ['linear', 'square'],  # Regression-specific loss functions
                },
                "CatBoost": {
                    'iterations': [100, 200],  # Moderate iterations for efficiency
                    'learning_rate': [0.05, 0.1],  # Balanced learning rates
                    'depth': [3, 5],  # Control depth for generalization
                    'l2_leaf_reg': [3, 5, 10],  # L2 regularization to prevent overfitting
                },
                "Lightgbm Regression": {
                    'n_estimators': [50, 100, 200],  # Number of boosting rounds
                    'learning_rate': [0.01, 0.05, 0.1],  # Step size for learning
                    'max_depth': [-1, 5, 10],  # Depth of the tree (-1 for unlimited)
                    'num_leaves': [15, 31, 63],  # Number of leaves in each tree
                    'min_child_samples': [10, 20, 30],  # Minimum data in a leaf node
                    'min_child_weight': [1e-3, 1e-2, 1e-1],  # Minimum sum of instance weights in a leaf
                    'colsample_bytree': [0.7, 0.9, 1.0],  # Fraction of features to consider per tree
                    'reg_alpha': [0.0, 0.1, 0.5],  # L1 regularization term
                    'reg_lambda': [0.0, 0.1, 0.5],  # L2 regularization term
                },
}

        model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                          models=models,param=params)
        
        ## To get best model score from dict
        best_model_score = max(sorted(model_report.values()))

        ## To get best model name from dict

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]
        y_train_pred=best_model.predict(X_train)

        classification_train_metric=get_regression_score(y_true=y_train,y_pred=y_train_pred)
        
        ## Track the experiements with mlflow
        self.track_mlflow(best_model,classification_train_metric)


        y_test_pred=best_model.predict(x_test)
        classification_test_metric=get_regression_score(y_true=y_test,y_pred=y_test_pred)

        self.track_mlflow(best_model,classification_test_metric)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        TM_Model=TSForecastingException(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=TSForecastingException)
        #model pusher
        save_object("final_model/model.pkl",best_model)
        

        ## Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric
                             )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact


        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact

            
        except Exception as e:
            raise TSForecastingException(e,sys)