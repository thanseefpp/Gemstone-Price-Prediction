#----------------------------------------- IMPORTING DEPENDENCIES -------------------------------------#
import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.Gemstone.config.logger import logging
from src.Gemstone.config.exception import CustomException
from zenml.steps import step

#----------------------------------------- FUNCTIONS/CLASSES ------------------------------------------#

class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self) -> None:
        """Initialize the data ingestion class."""
        pass

    def get_data(self) -> pd.DataFrame:
        try:
            logging.info("Collecting Dataset")
            return pd.read_csv("../../data/gemstone_price.csv")
        except Exception as e:
            raise CustomException(e,sys) from e
        
@step
def ingest_data() -> pd.DataFrame:
    """
    Args:
        None
    Returns:
        df: pd.DataFrame
    """
    try:
        ingest_data = IngestData()
        return ingest_data.get_data()
    except Exception as e:
        raise CustomException(e,sys) from e