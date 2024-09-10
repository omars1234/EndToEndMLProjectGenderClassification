
import sys
import numpy as np 
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder,OrdinalEncoder,PowerTransformer
from EndToEndMLProjectGenderClassification import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import janitor
from imblearn.combine import SMOTETomek,SMOTEENN
from EndToEndMLProjectGenderClassification.config.configuration import DataTransfornmationConfig

class DataTransfornmation:
    def __init__(self,config:DataTransfornmationConfig):
        self.config= config

    def data_dropping(self):
        df=pd.read_csv(self.config.data_path)
        df=df.drop(self.config.drop_cols,axis=1)
        df=df.sort_values(by=self.config.ordinal_cols).reset_index().drop("index",axis=1)

        for col in df.select_dtypes(include="object"):
            df[col]=LabelEncoder().fit_transform(df[col])

        logger.info("data LabelEncoder Done")     

        col_to_move = "gender"
        if col_to_move in df.columns:
            df = df[[col for col in df.columns if col != col_to_move] + [col_to_move]]

        logger.info("Moving the Target Feature to be the last column in the data set ==> Done")      
  
       
        train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

        train_set.to_csv(os.path.join(self.config.root_dir,"train.csv"),index=False)
        test_set.to_csv(os.path.join(self.config.root_dir,"test.csv"),index=False)

        logger.info("Splitting data into train and test subsets ==> Done")   

        #df=df.clean_names()
        return train_set,test_set

    def data_encoding(self):

        transpower_pipe=Pipeline(steps=[
            ("PowerTransformer",PowerTransformer(method="yeo-johnson"))
        ])    

        preprocessor=ColumnTransformer(
            [
                ("transformer",transpower_pipe,self.config.trans_cols),
                ("StandardScaler",StandardScaler(),self.config.numerical_cols)
            ]
        )

        logger.info("Created preprocessor object using  ColumnTransformer ==> Done") 

        logger.info("data_encoding  ==> Done") 

        return preprocessor
      

    def train_test_splitting(self):      
        
        preprocessor=self.data_encoding()
    
        logger.info("got preprocessor object  ==> Done")

        train_set,test_set=self.data_dropping()

        logger.info("got train_set,test_set from data_dropping()   ==> Done")

        train_set[self.config.trans_cols]=PowerTransformer(method="yeo-johnson").fit_transform(train_set[self.config.trans_cols])
        test_set[self.config.trans_cols]=PowerTransformer(method="yeo-johnson").fit_transform(test_set[self.config.trans_cols])

        logger.info("Apply PowerTransformer() to non normal data   ==> Done")
   
        input_train_set,target_train_set=train_set.drop(self.config.target_cols,axis=1),train_set[self.config.target_cols]
        input_test_set,target_test_set=test_set.drop(self.config.target_cols,axis=1),test_set[self.config.target_cols]

        logger.info("define x,y for train and test subsets  ==> Done")

        input_train_set_arry=input_train_set
        input_test_set_arry=input_test_set

        input_train_set_arry=input_train_set
        input_test_set_arry=input_test_set

        logger.info("Apply preprocessor.fit_transform on input train and transform on input test ==> Done")     

        smt=SMOTEENN(random_state=42,sampling_strategy="minority")
        
        input_train_set_final,target_train_set_final=smt.fit_resample(input_train_set_arry,target_train_set)
        input_test_set_final,target_test_set_final=smt.fit_resample(input_test_set_arry,target_test_set)

        logger.info("Apply SMOTEENN resampleing with sampling_strategy : minority  ==> Done") 

        train_arr=np.c_[input_train_set_final,np.array(target_train_set_final)]
        test_arr=np.c_[input_test_set_final,np.array(target_test_set_final)] 

        logger.info("Apply np.c_ to create train_arr and test_arr  ==> Done") 

        with open(f"{self.config.root_dir}/final_train.npy", 'wb') as file_obj:
           np.save(file_obj, train_arr)

        with open(f"{self.config.root_dir}/final_test.npy", 'wb') as file_obj:
           np.save(file_obj, test_arr) 

        logger.info("saving train_arr and test_arr  ==> Done")    

        logger.info("Data Splitting is completed")   

              

        print("=========================")

        logger.info(input_train_set.shape) 
        logger.info(target_train_set.shape) 
        logger.info(input_test_set.shape) 
        logger.info(target_test_set.shape)

        print("=========================")

        logger.info(input_train_set_arry.shape) 
        logger.info(input_test_set_arry.shape) 

        print("=========================")

        logger.info(input_train_set_final.shape) 
        logger.info(target_train_set_final.shape) 
        logger.info(input_test_set_final.shape) 
        logger.info(target_test_set_final.shape)

        print("=========================")

        logger.info(train_arr.shape) 
        logger.info(test_arr.shape)   
    