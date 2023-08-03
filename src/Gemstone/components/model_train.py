# ----------------------------------------- IMPORTING DEPENDENCIES -------------------------------------#
# Basic Import
import os
import sys
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,
                              RandomForestRegressor, VotingRegressor)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# Modelling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from Gemstone.components.model_evaluation import Evaluation
from Gemstone.config.exception import CustomException
from Gemstone.config.logger import logging
from Gemstone.utils.common import save_object

# ------------------------------------------- FUNCTIONS/CLASSES -------------------------------------------#


class Hyperparameter_Optimization:
    """
    Class for doing hyperparameter optimization.
    """

    def __init__(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
    ) -> None:
        """
        Initialize the class with the training and test data.
        Evaluation Class    
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.evaluate = Evaluation()

    def optimize_catboost(self, trial: optuna.Trial) -> float:
        try:
            """
            Method for optimizing CatBoostRegressor.
            """
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0),
                "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 30),
                "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 255),
                "verbose": 0,
            }
            model = CatBoostRegressor(**params, random_state=42)
            model.fit(self.x_train, self.y_train, eval_set=[(self.x_test, self.y_test)], early_stopping_rounds=50, verbose=100)
            y_pred = model.predict(self.x_test)
            _, _, rmse, R2_score = self.evaluate.get_metrics_scores(self.y_test, y_pred)
            print(f"r2_score : {R2_score}")
            return rmse
        except Exception as e:
            logging.info("Exited the optimize_catboost method of the Hyperparameter_Optimization class")
            raise CustomException(e, sys) from e

    def optimize_xgb(self, trial: optuna.Trial) -> float:
        try:
            """
            Method for optimizing XGBRegressor.
            """
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-8, 10.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0),
                "random_state": 42,
            }
            model = XGBRegressor(**params)
            model.fit(self.x_train, self.y_train, eval_set=[(self.x_test, self.y_test)], early_stopping_rounds=50, verbose=100)
            y_pred = model.predict(self.x_test)
            _, _, rmse, R2_score = self.evaluate.get_metrics_scores(self.y_test, y_pred)
            print(f"r2_score : {R2_score}")
            return rmse
        except Exception as e:
            logging.info("Exited the optimize_xgb method of the Hyperparameter_Optimization class")
            raise CustomException(e, sys) from e

    def optimize_knn(self, trial: optuna.Trial) -> float:
        try:
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "p": trial.suggest_int("p", 1, 2),
            }
            model = KNeighborsRegressor(**params)
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_test)
            _, _, rmse, R2_score = self.evaluate.get_metrics_scores(self.y_test, y_pred)
            print(f"r2_score : {R2_score}")
            return rmse
        except Exception as e:
            logging.info("Exited the optimize_knn method of the Hyperparameter_Optimization class")
            raise CustomException(e, sys) from e


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", 'model.pkl')


class ModelTrainer:

    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()
        self.evaluate = Evaluation()

    def data_splitter(self, train_array, test_array):
        """
            It split the train,test data into X_train, X_test, y_train, y_test.
        """
        try:
            return (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1]
            )
        except Exception as e:
            logging.info("Exited the data_splitter method of the ModelTrainer class")
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
                "GradientBoosting Regressor": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
        except Exception as e:
            logging.info("Exited the get_models method of the ModelTrainer class")
            raise CustomException(e, sys) from e

    def catboost_trainer(self, X_train, y_train, X_test, y_test, fine_tuning: bool = True):
        """
            It trains the CatBoostRegressor model.
            Args:
                X_train,y_train,X_test,y_test -> collecting data
                fine_tuning: If True, hyperparameter optimization is performed. If False, the default
                parameters are used. Defaults to True (optional).
        """
        logging.info("Started training CatBoostRegressor model.")
        try:
            if fine_tuning:
                hyper_opt = Hyperparameter_Optimization(
                    x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test
                )
                study = optuna.create_study(direction="minimize")
                study.optimize(hyper_opt.optimize_catboost, n_trials=100)
                best_params = study.best_params
                best_score = study.best_value
                print("Best Parameters:", best_params)
                print("Best RMSE:", best_score)
                model = CatBoostRegressor(**best_params, random_state=42)
            else:
                model = CatBoostRegressor()
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=100)
            return model
        except Exception as e:
            logging.info("Exited the catboost_trainer method of the ModelTrainer class")
            raise CustomException(e, sys) from e

    def xgb_trainer(self, X_train, y_train, X_test, y_test, fine_tuning: bool = True):
        """
            It trains the XGBRegressor model.
            Args:
                X_train,y_train,X_test,y_test -> collecting data
                fine_tuning: If True, hyperparameter optimization is performed. If False, the default
                parameters are used. Defaults to True (optional).
        """
        logging.info("Started training XGBRegressor model.")
        try:
            if fine_tuning:
                hyper_opt = Hyperparameter_Optimization(
                    x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test
                )
                study = optuna.create_study(direction="minimize")
                study.optimize(hyper_opt.optimize_xgb, n_trials=100)
                best_params = study.best_params
                best_score = study.best_value
                print("Best Parameters:", best_params)
                print("Best RMSE:", best_score)
                model = XGBRegressor(**best_params)
            else:
                model = XGBRegressor()
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=100)
            return model
        except Exception as e:
            logging.info("Exited the xgb_trainer method of the ModelTrainer class")
            raise CustomException(e, sys) from e

    def knn_trainer(self, X_train, y_train, X_test, y_test, fine_tuning: bool = True):
        """
        It trains the KNeighborsRegressor model.
        Args:
            X_train,y_train,X_test,y_test -> collecting data
            fine_tuning: If True, hyperparameter optimization is performed. If False, the default
            parameters are used. Defaults to True (optional).
        """
        logging.info("Started training KNeighborsRegressor model.")
        try:
            if fine_tuning:
                hyper_opt = Hyperparameter_Optimization(
                    x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test
                )
                study = optuna.create_study(direction="minimize")
                study.optimize(hyper_opt.optimize_knn, n_trials=100)
                best_params = study.best_params
                best_score = study.best_value
                print("Best Parameters:", best_params)
                print("Best RMSE:", best_score)
                model = KNeighborsRegressor(**best_params)
            else:
                model = KNeighborsRegressor()
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            logging.info("Exited the knn_trainer method of the ModelTrainer class")
            raise CustomException(e, sys) from e

    def initiate_model_training(self, train_array, test_array):
        try:
            X_train, X_test, y_train, y_test = self.data_splitter(train_array, test_array)
            logging.info("Data splitted into X_train, X_test, y_train, y_test and started Evaluate")
            models = self.get_models()
            model_report: dict = self.evaluate.initiate_multi_model_evaluation(X_train, X_test, y_train, y_test, models)
            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            _ = models[best_model_name]
            if best_model_score < 0.6:
                logging.info('model has r2 Score less than 60%')
                raise CustomException('No Best Model Found', "")
            logging.info(f'Best Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print(f"\nBefore Hyper Tune Best Model : {best_model_name}\n")
            fine_tuning = True
            logging.info("Fine Tuning Started...........")
            best_xgb_model = self.xgb_trainer(X_train, y_train, X_test, y_test, fine_tuning)
            best_cbr_model = self.catboost_trainer(X_train, y_train, X_test, y_test, fine_tuning)
            best_knn_model = self.knn_trainer(X_train, y_train, X_test, y_test, fine_tuning)
            logging.info("Fine Tuning Completed...........")
            logging.info('Voting Regressor model training started')
            # Creating final Voting regressor
            ensemble_reg = VotingRegressor([('cbr', best_cbr_model), ('xgb', best_xgb_model), ('knn', best_knn_model)], weights=[3, 2, 1])
            ensemble_reg.fit(X_train, y_train)
            logging.info("-"*20)
            mae, mse, rmse, R2_score = self.evaluate.evaluate_single_model(X_train, X_test, y_train, y_test, ensemble_reg)
            logging.info(f"\nmae :{mae},\nmse :{mse},\
                \nrmse :{rmse},\nR2_score :{R2_score}")
            logging.info('Voting Regressor Training Completed')
            # saving the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=ensemble_reg
            )
            logging.info('Best Model pickled file saved Successful')
            return mae, mse, rmse, R2_score
        except Exception as e:
            logging.info("Exited the initiate_model_training method of the ModelTrainer class")
            raise CustomException(e, sys) from e