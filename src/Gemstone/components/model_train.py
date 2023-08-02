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
from Gemstone.components.model_evaluation import Evaluation

#------------------------------------------- FUNCTIONS/CLASSES -------------------------------------------#

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts",'model.pkl')
    

class ModelTrainer:
    
    def __init__(self) -> None:
        self.model_trainer_config=ModelTrainerConfig()
        self.evaluate = Evaluation()
    
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
            logging.info("Data splitted into X_train, X_test, y_train, y_test and started Evaluate")
            models = self.get_models()
            model_report:dict = self.evaluate.initiate_multi_model_evaluation(X_train,X_test,y_train,y_test,models)
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
            logging.info(f'Best Model Name : {best_model_name} , R2 Score : {best_model_score}')
            
            logging.info("Hyperparameter tuning Started...")
            # Hyperparameter tuning on Catboost
            # Initializing catboost
            cbr = CatBoostRegressor(verbose=False)
            # Creating the hyperparameter grid
            param_dist = {'depth'          : [4,5,6,7,8,9, 10],
                          'learning_rate' : [0.01,0.02,0.03,0.04],
                          'iterations'    : [300,400,500,600]}
            #Instantiate RandomSearchCV object
            rscv = RandomizedSearchCV(cbr , param_dist, scoring='r2', cv =5, n_jobs=-1)
            # Fit the model
            rscv.fit(X_train, y_train)
            best_cbr = rscv.best_estimator_
            # the tuned parameters and score
            logging.info("-"*20)
            logging.info(f'Hyperparameter tuning Completed for Catboost Parameters : {rscv.best_params_}, Score : {rscv.best_score_}')
            
            # Initialize knn
            knn = KNeighborsRegressor()
            # parameters
            k_range = list(range(2, 31))
            param_grid = dict(n_neighbors=k_range)
            # Fitting the cv model
            grid = GridSearchCV(knn, param_grid, cv=5, scoring='r2',n_jobs=-1)
            grid.fit(X_train, y_train)
            best_knn = grid.best_estimator_
            # the tuned parameters and score
            logging.info("-"*20)
            logging.info(f'Hyperparameter tuning Completed for KNN Parameters : {grid.best_params_}, Score : {grid.best_score_}')
            
            logging.info('Voting Regressor model training started')
            # Creating final Voting regressor
            er = VotingRegressor([('cbr',best_cbr),('xgb',XGBRegressor()),('knn',best_knn)], weights=[3,2,1])
            er.fit(X_train, y_train)
            logging.info("-"*20)
            mae,mse,rmse,R2_score = self.evaluate.evaluate_single_model(X_train,X_test,y_train,y_test,er)
            logging.info(f"\nmae :{mae},\nmse :{mse},\
                \nrmse :{rmse},\nR2_score :{R2_score}")
            logging.info('Voting Regressor Training Completed')
            # saving the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = er
            )
            logging.info('Best Model pickled file saved Successful')
            return mae,mse,rmse,R2_score
        except Exception as e:
            raise CustomException(e, sys) from e