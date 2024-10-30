import os
import urllib.request as request
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import pickle,joblib
import pandas as pd
import numpy as np
from EndToEndMLProjectGenderClassification import logger
from EndToEndMLProjectGenderClassification.config.configuration import TrainingConfig
from sklearn.model_selection import RandomizedSearchCV
from pathlib import Path
from EndToEndMLProjectGenderClassification.utils.common import read_yaml,save_json
from EndToEndMLProjectGenderClassification.hyperpatameters.params import models
import xgboost as xgb
from xgboost import XGBClassifier



class Training:
    def __init__(self,config:TrainingConfig):
        self.config= config

    def initiate_Training(self):

        with open(self.config.train_data_arr_path, 'rb') as f:
            train_data = np.load(f)

        with open(self.config.test_data_arr_path, 'rb') as f:
            test_data = np.load(f)    

        x_train,y_train,x_test,y_test=(
            train_data[:,:-1],train_data[:,-1],test_data[:,:-1],test_data[:,-1]
        )

        yamalpath=Path("model.yaml")      
        params_config=read_yaml(yamalpath)
        '''
        models = {
        'RandomForestClassifier': RandomForestClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'XGBClassifier': XGBClassifier()
        }
        '''

        best_model = None
        best_score = -float("inf")
        best_params = {}
        best_model_name = ""

        for model_name, model in models.items():
            print(f"Running RandomizedSearchCV for {model_name}...")
            param_grid = params_config['models'][model_name]

            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=3, 
                scoring='accuracy',  
                cv=3, 
                verbose=2,
                #random_state=42,
                error_score='raise'
            )
            random_search.fit(x_train, y_train)

            if random_search.best_score_ > best_score:
                best_score = random_search.best_score_
                best_model = random_search.best_estimator_
                best_params = {"best_params":random_search.best_params_}
                best_model_name = model_name

            #np.random.seed(42)
            best_model.fit(x_train,y_train)
            joblib.dump(best_model,os.path.join(self.config.root_dir,self.config.best_model))
            results={"best_model_name":best_model_name,"best_params":best_params}
            save_json(path=Path(self.config.best_model_params),data=results)        