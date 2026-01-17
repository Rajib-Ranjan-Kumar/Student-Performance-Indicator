from src.exception import CustomException
from src.logger import logging
import os,sys
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig


@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.dataIngestionConfig=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv(r"C:\ds\student\notebook\stud.csv")
            logging.info('Read the dataset as dataframe')
            
            os.makedirs(os.path.dirname(self.dataIngestionConfig.train_data_path),exist_ok=True)
            df.to_csv(self.dataIngestionConfig.raw_data_path,index=False,header=True)
            logging.info('train test split initiated')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.dataIngestionConfig.train_data_path,index=False,header=True)
            test_set.to_csv(self.dataIngestionConfig.test_data_path,index=False,header=True)
            logging.info('Ingestion of the data is completed')
            return (
                self.dataIngestionConfig.train_data_path,
                self.dataIngestionConfig.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    xtrain,ytrain,_=data_transformation.initiate_data_transformation(train_data,test_data)
    model_trainer=ModelTrainer()
    r2_scor=model_trainer.initiate_model_trainer(xtrain,ytrain)
    print(r2_scor)


    