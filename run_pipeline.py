from Gemstone.config.logger import logging
from Gemstone.pipeline.train_pipeline import train_pipeline

if __name__ == "__main__":
    logging.info("Pipeline Running.......")
    train_pipeline()
