import os,sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,
                               test_array,                   
                               ):
        try:
            logging.info("model trainning started")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                
                "LinearRegression":LinearRegression(),
                "LogisticRegression":LogisticRegression(max_iter=1000),
                "Ridge":Ridge(),
                "SVR":SVR(),
                "NaiveBayes":GaussianNB(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "XGBRegressor":XGBRegressor()
                   }
            
            model_report:dict=evaluate_models(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,models=models)
            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
             ]

            best_model = models[best_model_name]

            logging.info("model training ended here")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("saving th best model")

            return r2_score(y_test,best_model.predict(x_test))
        except Exception as e:
            raise CustomException(e,sys)


