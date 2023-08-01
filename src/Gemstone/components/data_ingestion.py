#----------------------------------------- IMPORTING DEPENDENCIES -------------------------------------#
import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
#----------------------------------------- FUNCTIONS/CLASSES ------------------------------------------#

@dataclass
class DataIngestionConfig:
    """
        Data ingestion Config class which return three file paths.
    """
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class IngestData:
    """
        Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self):
        """Initialize the data ingestion class."""
        self.ingestion_config = DataIngestionConfig()

    def get_data(self) -> pd.DataFrame:
        try:
            logging.info("Collecting Dataset")
            return pd.read_csv("data/gemstone_price.csv")
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def initiate_ingest_data(self)-> pd.DataFrame:
        """
        Args:
            None
        Returns:
            df: pd.DataFrame
        """
        try:
            df = self.get_data()
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False) # saving the data
            return df
        except Exception as e:
            raise CustomException(e,sys) from e