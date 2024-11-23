# Create all generic functionality
import os,sys
import pickle
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception_handling import CustomException
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(y_test,y_pred):
    r2_square = r2_score(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    return r2_square,mae,mse,rmse


def model_training(X_train,y_train,X_test,y_test,x_arr, y_df, models):
    try:
        model_list = []
        r2_list = []

        kf = KFold(n_splits = 5, shuffle=True, random_state=None) # CrossValidation Technique
    
        logging.info('Model Training Performance using K-Fold cross validation')

        for i in range(len(models)):
            model = list(models.values())[i]
            scores = cross_val_score(model, x_arr,y_df , cv = kf)

            logging.info(f'"Model Name" - {(list(models.keys())[i])}')
            logging.info("="*35)
            logging.info(f'"Scores:", {scores}')
            logging.info(f'"Mean:", {scores.mean()}')

            
            logging.info('Training the Model')
            model.fit(X_train,y_train)
           
            logging.info('Predicting test Case')
            y_pred = model.predict(X_test)
            
            r2_square,mae,mse,rmse = evaluate_models(y_test,y_pred) 

            #print(list(models.keys())[i])
            logging.info("Model Evaluation")
            logging.info(f'"RMSE:",{rmse}')
            logging.info(f'"MAE:",{mae}')
            logging.info(f'"R2 score",{r2_square*100}')
            logging.info('='*35)
       
            
            r2_list.append(r2_square*100)
            model_list.append(list(models.keys())[i])

        return r2_list, model_list
        
    except Exception as e:
        raise CustomException(e,sys)