import os,sys
import pandas as pd
import numpy as np 
import dill
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open (file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            logging.info(f"{file_path} is saving in {dir_path}")

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        logging.info("Model evaluation started")
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_test_pred = model.predict(X_test)

            # Score
            test_model_score = r2_score(y_test, y_test_pred)

            # Store score
            report[model_name] = test_model_score

        logging.info("All models evaluated successfully")
        return report

    except Exception as e:
        raise CustomException(e, sys)
