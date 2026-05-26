import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import os
import logging

#log dir
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'data_ingestion.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(lineno)d -  %(module)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_url: str)->pd.DataFrame:
    try:
        df=pd.read_csv(data_url)
        logger.debug(f"Data Loaded succesfully from {data_url}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"failed to load data due to parsing error : {e}")
        raise
    except Exception as e:
        logger.error(f"failed to load data due to an Unexpected error : {e}")
        raise

def preprocess_data(df: pd.DataFrame)->pd.DataFrame:
    """PreProcess this Data"""
    try:
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
        df.rename(columns={'v1':'target','v2':'text'},inplace=True)
        logger.debug(f'preprocess_data returning df of shape {df.shape}')
        return df
    except KeyError as e:
        logger.error(f"missing column in dataframe {e} ")
        raise
    except Exception as e:
        logger.error(f'Unexpected error occured during data preprocessing : {e}')

def save_data(train_data: pd.DataFrame,test_data: pd.DataFrame,data_path: str) ->None:
    "save the train and test data"
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
        logger.debug(f"Train data saved succesfully to : {raw_data_path}")
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
        logger.debug(f"Test data saved succesfully to : {raw_data_path}")
        logger.debug(f'train shape: {train_data.shape}, test shape: {test_data.shape}')

    except Exception as e :
        logger.error(f"an unexpected error occured while saving train test data : {e}")
        raise

def main():
    try:
        test_size=0.3
        data_path='https://raw.githubusercontent.com/vikashishere/YT-MLOPS-Complete-ML-Pipeline/refs/heads/main/experiments/spam.csv'
        df=load_data(data_url=data_path)
        final_df=preprocess_data(df)
        train_data,test_data=train_test_split(final_df,test_size=test_size,random_state=42)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error(f'failed to complete data ingestion due to {e}')

if __name__=='__main__':
    main()