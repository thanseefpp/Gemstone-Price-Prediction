import sys
from typing import List

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Gemstone.config.exception import CustomException
from Gemstone.config.logger import logging


class Evaluation:
    """
        Evaluation class which evaluates the model performance using the sklearn metrics.
    """

    def __init__(self) -> None:
        """Initializes the Evaluation class."""
        pass

    def mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Squared Error (MSE) is the mean of the squared errors.
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            mse: float
        """
        try:
            return mean_squared_error(y_true, y_pred)
        except Exception as e:
            logging.info(
                "Exited the mean_squared_error method of the Evaluation class")
            raise CustomException(e, sys) from e

    def mean_absolute_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error (MAE) regression loss.
        The best value is 0.0
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            mae: float
        """
        try:
            return mean_absolute_error(y_true, y_pred)
        except Exception as e:
            logging.info(
                "Exited the mean_absolute_error method of the Evaluation class")
            raise CustomException(e, sys) from e

    def r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        R2 Score (R2) is a statistical measure of how close the observed values
        are to the predicted values. It is also known as the coefficient of
        determination.

        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            r2_score: float
        """
        try:
            return r2_score(y_true, y_pred)
        except Exception as e:
            logging.info("Exited the r2_score method of the Evaluation class")
            raise CustomException(e, sys) from e

    def root_mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Squared Error (RMSE) is the square root of the mean of the
        squared errors.

        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Return:
            rmse: float
        """
        try:
            return np.sqrt(mean_squared_error(y_true, y_pred))
        except Exception as e:
            logging.info(
                "Exited the root_mean_squared_error method of the Evaluation class")
            raise CustomException(e, sys) from e

    def get_metrics_scores(self, y_true, y_pred) -> float:
        try:
            mae = self.mean_absolute_error(y_true, y_pred)
            mse = self.mean_squared_error(y_true, y_pred)
            rmse = self.root_mean_squared_error(y_true, y_pred)
            R2_score = self.r2_score(y_true, y_pred)
            return mae, mse, rmse, R2_score
        except Exception as e:
            logging.info(
                "Exited the get_metrics_scores method of the Evaluation class")
            raise CustomException(e, sys) from e

    def evaluate_single_model(self, X_train, X_test, y_train, y_test, model):
        try:
            # Predict Training data
            ytrain_pred = model.predict(X_train)
            # Predict Testing data
            ytest_pred = model.predict(X_test)
            # Get metrics Train data score
            train_mae, train_mse, train_rmse, train_R2_score = self.get_metrics_scores(y_train, ytrain_pred)
            # Get metrics Test data score
            test_mae, test_mse, test_rmse, test_R2_score = self.get_metrics_scores(y_test, ytest_pred)
            logging.info("-"*20)
            logging.info(f"Train Evaluation Scores (mae : {train_mae}), (mse : {train_mse})\
                (rmse : {train_rmse}), (r2_score : {train_R2_score})")
            logging.info("-"*20)
            logging.info(f"Test Evaluation Scores (mae : {test_mae}), (mse : {test_mse})\
                (rmse : {test_rmse}), (r2_score : {test_R2_score})")
            return test_mae, test_mse, test_rmse, test_R2_score
        except Exception as e:
            logging.info(
                "Exited the evaluate_single_model method of the Evaluation class")
            raise CustomException(e, sys) from e

    def initiate_multi_model_evaluation(self, X_train, X_test, y_train, y_test, models)-> List:
        """
            evaluating the models and return the result
        """
        try:
            report = {}
            for i in range(len(models)):
                model = list(models.values())[i]
                # Train model
                model.fit(X_train, y_train)
                # Predict Training data
                y_train_pred = model.predict(X_train)
                # Predict Testing data
                y_test_pred = model.predict(X_test)
                # Get metrics Train data score
                self.get_metrics_scores(y_train, y_train_pred)
                # Get metrics Test data score
                _, _, _, test_R2_score = self.get_metrics_scores(y_test, y_test_pred)
                logging.info(f"Model Name : {list(models.keys())[i]}")
                logging.info("-"*20)
                # appending the the model name as key and the score as value
                report[list(models.keys())[i]] = test_R2_score
            return report
        except Exception as e:
            raise CustomException(e, sys) from e