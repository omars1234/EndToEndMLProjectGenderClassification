
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
from src.EndToEndMLProjectGenderClassification.utils.common import remove_outliers


class DataTransfornmation:
    def __init__(self,config:DataTransfornmationConfig):
        self.config= config

    def data_preperation(self):
        df=pd.read_csv(self.config.data_path)
        df=df.drop(self.config.drop_cols,axis=1)
        df["veh_value"]=df["veh_value"]*10000
        df=df[df["veh_value"]>0]
        logger.info("Dropping unneccesary cols and rows ==> Done") 

        #df=remove_outliers(df=df,data=self.config.trans_cols) 
        #logger.info("remove_outliers ==> Done")

        df=df.sort_values(by=self.config.ordinal_cols).reset_index().drop("index",axis=1)
        logger.info("sorting cols ==> Done") 

        for col in df.select_dtypes(include="object"):
            df[col]=LabelEncoder().fit_transform(df[col])
        logger.info("data LabelEncoder Done")

        train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

        train_set.to_csv(os.path.join(self.config.root_dir,"train.csv"),index=False)
        test_set.to_csv(os.path.join(self.config.root_dir,"test.csv"),index=False)
        logger.info("Splitting data into train and test subsets ==> Done")   
        return train_set,test_set         

      
    def train_test_transformation(self):      
        train_set,test_set=self.data_preperation()
        logger.info("got train_set,test_set from data_dropping()   ==> Done")

        train_set[self.config.trans_cols]=PowerTransformer(method="yeo-johnson").fit_transform(train_set[self.config.trans_cols])
        test_set[self.config.trans_cols]=PowerTransformer(method="yeo-johnson").fit_transform(test_set[self.config.trans_cols])
        logger.info("Apply PowerTransformer() to non normal data   ==> Done")
   
        input_train_set,target_train_set=train_set.drop(self.config.target_cols,axis=1),train_set[self.config.target_cols]
        input_test_set,target_test_set=test_set.drop(self.config.target_cols,axis=1),test_set[self.config.target_cols]
        logger.info("define x,y for train and test subsets  ==> Done")           

        smt=SMOTEENN(random_state=42,sampling_strategy="minority")
        
        input_train_set_final,target_train_set_final=smt.fit_resample(input_train_set,target_train_set)
        input_test_set_final,target_test_set_final=smt.fit_resample(input_test_set,target_test_set)

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


        logger.info(input_train_set_final.shape) 
        logger.info(target_train_set_final.shape) 
        logger.info(input_test_set_final.shape) 
        logger.info(target_test_set_final.shape)

        print("=========================")

        logger.info(train_arr.shape) 
        logger.info(test_arr.shape)   
    
        
       