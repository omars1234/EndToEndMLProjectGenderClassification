import os
import urllib.request as request
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import pickle,joblib
import pandas as pd
import numpy as np
from EndToEndMLProjectGenderClassification import logger
from EndToEndMLProjectGenderClassification.config.configuration import TrainingConfig



class Training:
    def __init__(self,config:TrainingConfig):
        self.config= config

    def initiate_Training(self):

        with open(self.config.train_data_arr_path, 'rb') as f:
            train_data = np.load(f)  


        x_train,y_train=(
            train_data[:,:-1],train_data[:,-1]
        )

        model=GradientBoostingClassifier(
            n_estimators= self.config.n_estimators,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state
            )
        
        model.fit(x_train,y_train)  

        joblib.dump(model,os.path.join(self.config.root_dir,self.config.model_name))      

