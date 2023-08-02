from Gemstone.config.logger import logging
from Gemstone.config.exception import CustomException
import numpy as np
import sys
from typing import List
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


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
            logging.info("Entered the mean_squared_error method of the Evaluation class")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"The mean squared error value is: {mse}")
            return mse
        except Exception as e:
            logging.info("Exited the mean_squared_error method of the Evaluation class")
            raise CustomException(e,sys) from e
    
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
            logging.info("Entered the mean_absolute_error method of the Evaluation class")
            mae = mean_absolute_error(y_true, y_pred)
            logging.info(f"The mean absolute error value is: {mae}")
            return mae
        except Exception as e:
            logging.info("Exited the mean_absolute_error method of the Evaluation class")
            raise CustomException(e,sys) from e

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
            logging.info("Entered the r2_score method of the Evaluation class")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"The r2 score value is: {r2}")
            logging.info("Exited the r2_score method of the Evaluation class")
            return r2
        except Exception as e:
            logging.info("Exited the r2_score method of the Evaluation class")
            raise CustomException(e,sys) from e

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
            logging.info("Entered the root_mean_squared_error method of the Evaluation class")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f"The root mean squared error value is: {rmse}")
            return rmse
        except Exception as e:
            logging.info("Exited the root_mean_squared_error method of the Evaluation class")
            raise CustomException(e,sys) from e
    
    def get_metrics_scores(self,y_true, y_pred) -> float:
        try:
            logging.info("-"*20)
            self.mean_absolute_error(y_true,y_pred)
            self.mean_squared_error(y_true,y_pred)
            self.root_mean_squared_error(y_true,y_pred)
            return self.r2_score(y_true,y_pred)
        except Exception as e:
            logging.info("Exited the get_metrics_scores method of the Evaluation class")
            raise CustomException(e,sys) from e
        
        
    def initiate_model_evaluation(self, X_train, X_test, y_train, y_test, models)-> List:
        """
            evaluating the models and return the result
        """
        try:
            report = {}
            for i in range(len(models)):
                model = list(models.values())[i]
                # Train model
                model.fit(X_train,y_train)
                # Predict Training data
                y_train_pred = model.predict(X_train)
                # Predict Testing data
                y_test_pred =model.predict(X_test)
                # Get r2 score Train data score
                train = self.get_metrics_scores(y_train, y_train_pred)
                # Get r2 score Test data score
                test_r2_score = self.get_metrics_scores(y_test,y_test_pred)
                logging.info(f"Model Name : {list(models.keys())[i]}")
                logging.info("-"*20)
                # appending the the model name as key and the score as value
                report[list(models.keys())[i]] = test_r2_score
            return report
        except Exception as e:
            raise CustomException(e,sys) from e