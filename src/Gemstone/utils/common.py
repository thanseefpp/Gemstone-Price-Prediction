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