import os
import sys #Logging and Exception
from src.logger import logging
from src.exception_handling import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass # Used in place of Init function


## Initialize the data ingestion configuration

@dataclass
class DataIngestionconfig: # Class created to save all the variables 
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw.csv')


## Create the data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig() #Inside the ingestion config variable 'train,test and raw' data path will be created
        
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')

        try:
            df = pd.read_csv(os.path.join('notebooks/data','gemstone.csv'))

            logging.info("Data set read as Pandas DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok = True)
            # This line of code ensures that the parent directory of the path stored in self.ingestion_config.raw_data_path exists 
            # before proceeding with any file operations like saving or reading data from that path
            df.to_csv(self.ingestion_config.raw_data_path, index = False)

            logging.info("Raw data is created")

            train_set,test_set = train_test_split(df,test_size=.30,random_state=1) # Splitting data into train and test

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True) # Converting train_set to csv
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Ingestion of data is completed")

            return(
                   self.ingestion_config.train_data_path,
                   self.ingestion_config.test_data_path,  
                 )

        except Exception as e:
            logging.info("Exception occured at Data Ingestion stage")
            raise CustomException(e,sys)