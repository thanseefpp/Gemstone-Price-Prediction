#----------------------------------------- IMPORTING DEPENDENCIES -------------------------------------#
from Gemstone.config.logger import logging
from Gemstone.components.data_ingestion import IngestData
from Gemstone.components.data_cleaning import DataCleaning

#----------------------------------------- FUNCTIONS/CLASSES -------------------------------------#


def train_pipeline():
    obj = IngestData()
    data = obj.initiate_ingest_data()
    clean_obj = DataCleaning()
    train_arr,test_arr,_ = clean_obj.clean_data_and_transform(data=data)
    print('working the model')
    
if __name__ == "__main__":
    logging.info("Pipeline Running.......")
    train_pipeline()