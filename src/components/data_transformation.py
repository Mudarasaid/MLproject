import os
import sys 
import numpy as np
from src.exception import CustomException
from src.logger import logging
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from src.utils import save_object

@dataclass
class DataTransfirmationConfig:
    preprocessor_obj_file_path: str=os.path.join('artifacts', "preprocessor.pkl")
    
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransfirmationConfig()
        
    def get_data_transformer_object(self):
        try:
            numerical_columns =["writing_score","reading_score"]
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'    
            ]
            
            numurical_culomns_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            
            categorical_Pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                    
                ]
                
            )
            
            logging.info("transforming numerical data and categorical data completed")
            
            preprocssor= ColumnTransformer(
                [
                
                ("numurical_culomns_pipeline", numurical_culomns_pipeline, numerical_columns),
                ("categorical_Pipeline", categorical_Pipeline, categorical_columns)
                
                ]
            )
            
            return preprocssor
                
        except Exception as e:
            raise CustomException(e,sys)    
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("read traing and test data completed")
            logging.info("obtaining preprocessing object")
            
            preprocessing_obj=self.get_data_transformer_object()
            
            target_column_name="math_score"
            numerical_columns =["writing_score","reading_score"]
            
            input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df= train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df= test_df[target_column_name]
            
            logging.info(
                f"applying preprocessing object on training and testing dataset"
            )
            
            input_feature_train_array= preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array= preprocessing_obj.transform(input_feature_test_df)
            """"
            We use fit_transform() on the training data because we need to learn the parameters required for transformation — 
            for example, the mean and standard deviation for scaling or the unique categories for encoding. 
            Once the transformer has learned these parameters during fit, it applies them to the training data using transform.

            On the other hand, we only use transform() on the test data to ensure consistency. 
            We must apply the same transformation logic learned from the training data. 
            Fitting on test data would introduce data leakage, 
            which leads to overly optimistic performance and invalidates the model evaluation.
            """
            train_arr= np.c_[
                input_feature_train_array, np.array(target_feature_train_df)]
            test_arr= np.c_[
                input_feature_test_array, np.array(target_feature_test_df)]
            """
            np.c_ is a handy NumPy shortcut to concatenate arrays column-wise.

            Basically, its like doing np.concatenate([...], axis=1) but simpler and cleaner.

            Makes merging features and targets super easy for feeding into models.
            """
            
            logging.info(f"saved processing object")
            
            save_object(
                file_path=  self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
                
            )
            
        except Exception as e: 
            raise CustomException(e,sys)