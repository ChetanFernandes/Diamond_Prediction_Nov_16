import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.logger import logging
from src.exception_handling import CustomException
from sklearn.neighbors import KNeighborsRegressor
from src.utilis import save_object
from dataclasses import dataclass
import sys,os
from src.utilis import model_training


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_arr,test_arr,x_arr, y_df):
        try:
            logging.info("Splitting Dependent and Independent variable from train and test data")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            


            models = {
                        "LinearRegression":LinearRegression(),
                        "Lasso":Lasso(),
                        "Ridge":Ridge(),
                        "ElasticNet":ElasticNet(),
                        "AdaBoostRegressor": AdaBoostRegressor(),
                        "GradientBoostingRegressor": GradientBoostingRegressor(),
                        "XGB_Regressor" :  XGBRegressor(),
                        "DecisionTree": DecisionTreeRegressor(),
                        "Random_Forest" : RandomForestRegressor(),
                        "KNeighbours": KNeighborsRegressor()

                      }
            
            r2_list, model_list = model_training(X_train,y_train,X_test,y_test,x_arr,y_df, models)

            print(f'Model name with highest accuracy is {model_list[(r2_list.index(max(r2_list)))]} with score of {round(max(r2_list),2)}')
            logging.info(f'Model name with highest accuracy is {model_list[(r2_list.index(max(r2_list)))]} with score of {round(max(r2_list),2)}')

            best_model = model_list[(r2_list.index(max(r2_list)))]

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj =  best_model
                      )

        except Exception as e:
            raise CustomException(e,sys)



