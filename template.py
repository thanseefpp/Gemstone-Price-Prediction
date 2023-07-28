#----------------------------------------------- IMPORTING DEPENDENCIES ------------------------------------------------#
import os
from pathlib import Path
import logging

#----------------------------------------------- CONFIGURING PROJECT REQ -----------------------------------------------#

logging.basicConfig(level=logging.INFO,format='[%(asctime)s]:%(message)s:')

project_name = "Gemstone"


list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/exception.py",
    f"src/{project_name}/config/logger.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/predict_pipeline.py",
    f"src/{project_name}/pipeline/train_pipeline.py",
    "research/notebooks/train.ipynb"
    "tests/__init__.py"
    "tests/data_test.py"
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
]


for item in list_of_files:
    file_path = Path(item)
    file_dir, file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for file: {file_name}")

    if not os.path.exists(file_path) or os.path.getsize(file_path) != 0:
        with open(file_path,'w') as f:
            logging.info(f"Creating an empty file: {file_path})")
    else:
        logging.info(f"{file_name} is already exists")