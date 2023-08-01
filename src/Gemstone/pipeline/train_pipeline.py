from Gemstone.config.logger import logging
from Gemstone.components.data_ingestion import IngestData
from Gemstone.components.data_cleaning import DataCleaning
from Gemstone.components.model_train import ModelTrainer

def train_pipeline():
    obj = IngestData()
    data = obj.initiate_ingest_data()
    clean_obj = DataCleaning()
    train_arr,test_arr,_ = clean_obj.clean_data_and_transform(data=data)
    trainer = ModelTrainer()
    r2score = trainer.initiate_model_training(train_arr,test_arr)
    logging.info('training Completed')