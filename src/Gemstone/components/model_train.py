#----------------------------------------- IMPORTING DEPENDENCIES -------------------------------------#
# Basic Import
import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Modelling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from Gemstone.config.exception import CustomException
from Gemstone.config.logger import logging
from Gemstone.utils.common import save_object
from Gemstone.utils.common import evaluate_models
# from Gemstone.utils.common import print_evaluated_results
# from Gemstone.utils.common import model_metrics

#------------------------------------------- FUNCTIONS/CLASSES -------------------------------------------#

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts",'model.pkl')
    

class ModelTrainer:
    
    def __init__(self) -> None:
        self.model_trainer_config=ModelTrainerConfig()
    
    def data_splitter(self, train_array, test_array):
        """
            It split the train,test data into X_train, X_test, y_train, y_test.
        """
        try:
            return (
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1]
            ) 
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def get_models(self):
        try:
            return {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "GradientBoosting Regressor":GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            } 
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def initiate_model_training(self,train_array,test_array):
        try:
            X_train, X_test, y_train, y_test = self.data_splitter(train_array,test_array)
            logging.info("Data splitted into X_train, X_test, y_train, y_test")
            models = self.get_models()
            model_report:dict = evaluate_models(X_train,X_test,y_train,y_test,models)
            print(f"\n{'---'*20}\n{model_report}\n{'---'*20}")
            
            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if best_model_score < 0.6 :
                logging.info('model has r2 Score less than 60%')
                raise CustomException('No Best Model Found',"")
            print(f"\n{'---'*20}\nBest Model Name :{best_model_name}, R2 Score : {best_model_score}\n{'---'*20}")
            logging.info(f'Best Model Name : {best_model_name} , R2 Score : {best_model_score}')
            return best_model_score
        except Exception as e:
            raise CustomException(e, sys) from e