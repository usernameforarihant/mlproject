import sys
sys.path.append("C:\\Users\\ariha\\Desktop\\Krish_Naik\\mlproject")
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsRegressor
from src.utils import save_obj, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input")
            X_train, X_test, y_train, y_test = (train_array[:, :-1], test_array[:, :-1], train_array[:, -1], test_array[:, -1])
            
            models = {
                'AdaBoostRegressor': AdaBoostRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'LinearRegression': LinearRegression(),
                'XGBRegressor': XGBRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose=0),  # Suppress output of CatBoost
                'KNeighborsRegressor': KNeighborsRegressor()
            }
            params = {
    "DecisionTreeRegressor": {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    },
    "RandomForestRegressor": {
        'n_estimators': [8, 16, 32, 64, 128, 256],
    },
    "GradientBoostingRegressor": {
        'learning_rate': [.1, .01, .05, .001],
        'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        'n_estimators': [8, 16, 32, 64, 128, 256],
    },
    "LinearRegression": {},
    "XGBRegressor": {
        'learning_rate': [.1, .01, .05, .001],
        'n_estimators': [8, 16, 32, 64, 128, 256],
    },
    "CatBoostRegressor": {
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [30, 50, 100],
    },
    "AdaBoostRegressor": {
        'learning_rate': [.1, .01, 0.5, .001],
        'n_estimators': [8, 16, 32, 64, 128, 256],
    },
    "KNeighborsRegressor": {}
}

                
            

            # model_report: dict = evaluate_models(X_train=X_train,X_test=X_test, y_train=y_train, y_test=y_test, models=models)
            model_report: dict = evaluate_models(X_train=X_train,X_test=X_test, y_train=y_train, y_test=y_test, models=models,params=params)
            best_model_score = max(list(sorted(model_report.values())))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on both testing and training data")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_sc = r2_score(y_test, predicted)
            return r2_sc

        except Exception as e:
            raise CustomException(e,sys)
