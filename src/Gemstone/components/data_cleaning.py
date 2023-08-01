#----------------------------------------- IMPORTING DEPENDENCIES -------------------------------------#
import os
import sys
import numpy as np 
import pandas as pd
from Gemstone.config.exception import CustomException
from Gemstone.config.logger import logging
from Gemstone.utils.common import save_object
from typing import Union
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.model_selection import train_test_split

#----------------------------------------- FUNCTIONS/CLASSES -----------------------------------------#


@dataclass
class DataConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")

class DataCleaning:
    """
        Data cleaning class which pre processes the data and divides it into train and test data.
    """

    def __init__(self) -> None:
        """Initializes the DataCleaning class."""
        self.data_config = DataConfig()
        
    def get_data_transformation_object(self):
        '''
            This function is responsible for data transformation
        '''
        try:
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())                
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinal_encoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]
            )

            logging.info(f'Categorical Columns : {categorical_cols}')
            logging.info(f'Numerical Columns   : {numerical_cols}')
            return ColumnTransformer([
                    ('num_pipeline',num_pipeline,numerical_cols),
                    ('cat_pipeline',cat_pipeline,categorical_cols)
                ])
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def preprocess_data(self, data:pd.DataFrame) -> pd.DataFrame:
        """
            Removes columns which are not required\
            and fill values where found empty
        """
        try:
            target_column_name = 'price'
            data = data.drop(["id"],axis=1)
            # this will replace the empty fields with median values
            data[target_column_name].fillna(
                data[target_column_name].median(), inplace=True
            )
            return data
            
        except Exception as e:
            raise CustomException(e, sys) from e

    def divide_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.DataFrame]:
        """
            It divides the data into train and test data.
        """
        try:
            train_df,test_df = train_test_split(
                df, test_size=0.2, random_state=42
            )
            train_df.to_csv(self.data_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.data_config.test_data_path,index=False, header=True)
            return train_df,test_df
        except Exception as e:
            raise CustomException(e, sys) from e

    def clean_data_and_transform(self,data: pd.DataFrame):
        """
        Data cleaning class which preprocesses the data and divides it into train and test data.

        Args:
            data: pd.DataFrame
        """
        try:
            logging.info("Before Cleaning the Data...")
            df = self.preprocess_data(data=data)
            train_df,test_df = self.divide_data(df)            
            preprocessing_obj = self.get_data_transformation_object()
            
            target_column_name = 'price'
            
            input_feature_train_df = train_df.drop(columns=target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=target_column_name,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys) from e
    
    




