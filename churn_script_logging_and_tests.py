"""
This module contains loggind and testing for churn project

Author: Fabio
Date: Dec. 2021
"""


import os
import pandas as pd
import logging
import churn_library as cls
from pandas._testing import assert_frame_equal

  
logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(pth,import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        logging.info("INFO: Testing import_data asserts")

        assert os.path.exists(pth)==True, 'path does not exist'


    except FileNotFoundError as err:
        logging.error("ERROR: Testing import_data: The file wasn't found: {}".format(err))
 
    try:
        logging.info("INFO: Testing import_data read file")
        df= import_data(pth)    

        logging.info("SUCCESS: Testing import_data read file")

    except BaseException as err:
        logging.error("ERROR: Testing import_data error reading file: {}".format(err))

    try:
        assert df.shape[0] > 0,'zero records'
        assert df.shape[1] > 0,'zero records'
        logging.info("SUCCESS: Testing import_data file is not empty")

        return df

    except AssertionError as err:
        logging.error("ERROR: Testing import_data: The file doesn't appear to have rows and columns: {}".format(err))
    
        #return empty df
        return pd.DataFrame()   
    except BaseException as err:
        logging.error("ERROR: Testing import_data: dataframe does not exist: {}".format(err))

        #return empty df
        return pd.DataFrame()

def test_eda(df,perform_eda,pth_folder_plot):
    '''
    test perform eda function
    '''
    try:
        logging.info("INFO: Testing perform_eda")

        #check size of dataframe
        assert all([True if elem > 0 else False for elem in df.shape]),'not all columns in dataframe'

        logging.info("SUCCESS: Testing perform_eda: dataframe shape > 0")

        #check if exists any null values
        assert df.isnull().sum().sum()==0,'dataframe contains NULLS'
        logging.info("SUCCESS: Testing perform_eda: no NULL values in input dataframe")

        #check if folder to store plot exists
        assert os.path.exists(pth_folder_plot)==True, 'folder does not exist'
        logging.info("SUCCESS: Testing perform_eda: plot folder exists")


    except AssertionError as err:
        logging.error("ERROR: test_eda: {}".format(err)) 

    try:

        perform_eda(df,pth_folder_plot)

        logging.info("SUCCESS: Testing perform_eda: plot stored in folder")

    except BaseException as err:
        logging.error("ERROR: Testing test_eda: error storing plots: {}".format(err))    

def test_encoder_helper(encoder_helper,df, category_lst, response):
    '''
    test encoder helper
    '''
    try:
        logging.info("INFO: Testing test_encoder_helper")

        #check size of dataframe
        assert all([True if elem > 0 else False for elem in df.shape]),'empty dataframe'

        logging.info("SUCCESS: Testing test_encoder_helper: dataframe shape > 0")

        #check if exists any null values
        assert df.isnull().sum().sum()==0,'dataframe contains NULLS'
        logging.info("SUCCESS: Testing test_encoder_helper: no NULL values in input dataframe")

        #check if category_lst exist
        assert len(category_lst)>0,'category list empty'
        logging.info("SUCCESS: Testing test_encoder_helper: category list does not exist")

        #check if response is string
        assert isinstance(response,str),'response variable is not string'
        logging.info("SUCCESS: Testing test_encoder_helper: response is string")

        #check if category list are columns in dataframe
        assert all(elem in list(df.columns)  for elem in category_lst),'category list values are not in columns'
        logging.info("SUCCESS: Testing test_encoder_helper: category list are columns in dataframe")

        #check if response refers to a column in dataframe
        assert response in list(df.columns),'response variable is not in dataframe columns'
        logging.info("SUCCESS: Testing test_encoder_helper: response refers to a columns in dataframe")



    except AssertionError as err:
        logging.error("ERROR: Testing test_encoder_helper: error: {}".format(err))    

    try:
        logging.info("INFO: Testing test_encoder_helper method")

        # test method
        encoder_return= encoder_helper(df,category_lst,response)

        logging.info("SUCCESS: Testing test_encoder_helper: success performing new columns")    

        return encoder_return
    except BaseException as err :

        logging.error("ERROR: Testing test_encoder_helper: error performing new columns: {}".format(err))    
        #return empty df
        return pd.DataFrame()


    


def test_perform_feature_engineering(perform_feature_engineering,df,keep_cols,response):
    '''
    test perform_feature_engineering
    '''
    try:
        logging.info("INFO: Testing test_perform_feature_engineering")

        #check size of dataframe
        assert all([True if elem > 0 else False for elem in df.shape]), "dataframe shape>0"

        logging.info("SUCCESS: Testing test_perform_feature_engineering: dataframe shape > 0")

        #check if keep list are columns in dataframe
        assert all(elem in list(df.columns)  for elem in keep_cols), "keep cols are not in dataframe"
        logging.info("SUCCESS: Testing test_encoder_helper: keep_cols are columns in dataframe")


        #check if response is string
        assert isinstance(response,str), "response is not str"
        logging.info("SUCCESS: Testing test_perform_feature_engineering: response is string")

    except AssertionError as err:
        logging.error("ERROR: Testing test_perform_feature_engineering: error assertion: {}".format(err))    

    try:
        logging.info("INFO: Testing test_perform_feature_engineering splitting dataframe")

        X_train, X_test, y_train, y_test = perform_feature_engineering(df,keep_cols,response)

        logging.info("SUCCESS: Testing test_perform_feature_engineering splitting dataframe")
        return X_train, X_test, y_train, y_test
    except BaseException as err:
        logging.error("ERROR: Testing test_perform_feature_engineering: error splitting dataframe: {}".format(err))

        return pd.DataFrame(), pd.DataFrame(),pd.DataFrame(),pd.DataFrame()


def test_train_models(train_models,X_train, X_test, y_train, y_test):
    '''
	test train_models
	'''
    try:
        logging.info("INFO: Testing test_train_models asserts")

        #check size of dataframe
        assert all([True if elem > 0 else False for elem in X_train.shape]), "X_train shape>0"
        #check size of dataframe
        assert all([True if elem > 0 else False for elem in X_test.shape]), "X_test shape>0"
        #check size of dataframe
        assert all([True if elem > 0 else False for elem in y_train.shape]), "y_train shape>0"        
        #check size of dataframe
        assert all([True if elem > 0 else False for elem in y_test.shape]), "y_test shape>0"
    
    except AssertionError as err:
        logging.error("ERROR: Testing test_train_models: error assertion: {}".format(err))  

    try:

        logging.info("INFO: Testing test_train_models")
        train_models(X_train, X_test, y_train, y_test)  

    except BaseException as err:
        logging.error("ERROR: Testing test_train_models: error models: {}".format(err))  

    



if __name__ == "__main__":
    df=test_import('data/bank_data.csv',cls.import_data)
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    test_eda(df,cls.perform_eda,'images/eda/')
    
    category_lst=['Gender','Education_Level','Marital_Status','Income_Category','Card_Category']
    response='Churn'
    
    test_encoder_helper(cls.encoder_helper,df,['Gender','Education_Level','Marital_Status','Income_Category','Card_Category'],'Churn')

    
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']
    X_train, X_test, y_train, y_test=cls.perform_feature_engineering(df,keep_cols,response)
    #test_train_models(cls.train_models,X_train, X_test, y_train, y_test)





