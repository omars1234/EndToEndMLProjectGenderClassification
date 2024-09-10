from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import os
from EndToEndMLProjectGenderClassification.pipeline.prediction import PredictionPipeline



app=Flask(__name__)

@app.route('/',methods=['GET'])
def homepage():
    return render_template('index.html')


@app.route('/train',methods=["GET"])
def training():
    os.system("python main.py")
    return ("Training Successful")


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            veh_value=float(request.form['veh_value'])
            exposure=float(request.form['exposure'])
            claimcst0=int(request.form['claimcst0'])
            veh_body=int(request.form['veh_body'])
            veh_age=int(request.form['veh_age'])
            area=int(request.form['area'])
            agecat=int(request.form['agecat'])

            

            input_data = [veh_value,exposure,claimcst0,veh_body,veh_age,area,agecat]

            input_data = np.array(input_data).reshape(1, 7)

            obj = PredictionPipeline()
            predict = obj.predict(input_data)

            
            return render_template('results.html', prediction = str(predict))
    

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
        
    else:
        return render_template('index.html')        


if __name__=='__main__':
    app.run(debug=True,host="0.0.0.0",port=8080)