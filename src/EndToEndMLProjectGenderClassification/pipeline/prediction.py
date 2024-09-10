import joblib
import numpy as np
import pandas as pd
from pathlib import Path



class PredictionPipeline:
    def __init__(self):
        self.model=joblib.load(Path("artifacts/training/model.joblib"))

    def predict(self,input_data):
        prediction=self.model.predict(input_data)
        prediction_prob=self.model.predict_proba(input_data)
        prediction_prob=np.round(self.model.predict_proba(input_data).max(),2)
        if prediction==0:
            return "Female with probability {}".format(prediction_prob)
        else:
            return "Male with probability {}".format(prediction_prob)

        #return prediction 

           
        
    