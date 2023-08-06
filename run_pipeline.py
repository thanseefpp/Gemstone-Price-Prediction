from Gemstone.components.data_cleaning import DataCleaning
from Gemstone.components.data_ingestion import IngestData
from Gemstone.components.model_evaluation import Evaluation
from Gemstone.components.model_train import ModelTrainer
from Gemstone.config.logger import logging
from Gemstone.pipeline.train_pipeline import train_pipeline

if __name__ == "__main__":
    logging.info("Pipeline Running.......")
    ingest_data = IngestData()
    clean_data = DataCleaning()
    model_train = ModelTrainer()
    evaluate_model = Evaluation()
    mae, mse, rmse, r2_score = train_pipeline(
        ingest_data, clean_data, model_train, evaluate_model)
    logging.info(
        f'training done, Score is MAE :{mae} , mse: {mse}, rmse{rmse}, r2_score : {r2_score}')
