import sys

import pandas as pd

from Gemstone.components.data_cleaning import DataConfig
from Gemstone.components.model_train import ModelTrainerConfig
from Gemstone.config.exception import CustomException
from Gemstone.config.logger import logging
from Gemstone.utils.common import load_object


class PredictPipeline:
    def __init__(self):
        self.trained_model_file_path = ModelTrainerConfig()
        self.preprocessor_file_path = DataConfig()

    def predict(self, features):
        try:
            model_path = self.trained_model_file_path.trained_model_file_path
            preprocessor_path = self.preprocessor_file_path.preprocessor_obj_file_path
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            logging.info("Before Starting the prediction")
            return model.predict(data_scaled)
        except Exception as e:
            logging.info(
                "Exited the predict method of the PredictPipeline class")
            raise CustomException(e, sys) from e


class CustomData:
    def __init__(
            self,
            carat: float,
            depth: float,
            table: float,
            x: float,
            y: float,
            z: float,
            cut: str,
            color: str,
            clarity: str):
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                'carat': [self.carat],
                'depth': [self.depth],
                'table': [self.table],
                'x': [self.x],
                'y': [self.y],
                'z': [self.z],
                'cut': [self.cut],
                'color': [self.color],
                'clarity': [self.clarity]
            }
            logging.info('Dataframe Collected')
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            logging.info(
                'Exited the get_data_as_data_frame method of the CustomData class')
            raise CustomException(e, sys)
