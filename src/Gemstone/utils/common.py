#----------------------------------------- IMPORTING DEPENDENCIES -------------------------------------#
import os
import sys
import pickle
import numpy as np 
import pandas as pd
from Gemstone.config.exception import CustomException
from Gemstone.config.logger import logging
from sklearn.metrics import r2_score
#----------------------------------------- FUNCTIONS/CLASSES -----------------------------------------#

def load_object(file_path):
    """
        Loading the object from a specific path
    """
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e, sys) from e


def save_object(file_path, obj):
    """
        saving the object in a specific path
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object Saved at : {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e
    

def evaluate_models(X_train,X_test,y_train,y_test,models):
    """
        evaluating the model
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
            # Get R2 scores for train and test data
            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys) from e