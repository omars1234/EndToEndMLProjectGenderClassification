import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib,json
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from pathlib import Path
from EndToEndMLProjectGenderClassification.utils.common import save_json
from EndToEndMLProjectGenderClassification.config.configuration import ModelEvaluationConfig



class ModelEvaluation:
    def __init__(self,config:ModelEvaluationConfig):
        self.config= config

    def evaluation_metrics(self,actual,pred):
        acc=accuracy_score(actual,pred) 
        preci=precision_score(actual,pred) 
        re_call=recall_score(actual,pred) 
        f1=f1_score(actual,pred) 
        return acc,preci,re_call,f1
    
    def save_results(self):
        with open(self.config.test_data_arr_path, 'rb') as f:
            test_data = np.load(f)
        #test_data=pd.read_csv(self.config.test_data_path)
        model=joblib.load(self.config.model_path)

        x_test,y_test=(
            test_data[:,:-1],test_data[:,-1]
        )
        prediction=model.predict(x_test)

        (acc,preci,re_call,f1)=self.evaluation_metrics(y_test,prediction)

        scors={"accuracy":acc,"precision":preci,"recall":re_call,"f1score":f1}

        save_json(path=Path(self.config.best_model_metrics),data=scors)