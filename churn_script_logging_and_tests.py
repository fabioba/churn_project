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
        logging.info("INFO: Testing import_data")

        df=import_data(pth)    

            
        logging.info("SUCCESS: Testing import_data read file")


    except FileNotFoundError as err:
        logging.error("ERROR: Testing import_data: The file wasn't found: {}".format(err))
 

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0

    except AssertionError as err:
        logging.error("ERROR: Testing import_data: The file doesn't appear to have rows and columns: {}".format(err))
    except BaseException as err:
        logging.error("ERROR: Testing import_data: dataframe does not exist: {}".format(err))


def test_eda(df,perform_eda,pth_folder_plot):
    '''
    test perform eda function
    '''
    try:
        logging.info("INFO: Testing perform_eda")

        #check size of dataframe
        assert all([True if elem > 0 else False for elem in df.shape])

        logging.info("SUCCESS: Testing perform_eda: dataframe shape > 0")

        #check if exists any null values
        assert df.isnull().sum().sum()==0
        logging.info("SUCCESS: Testing perform_eda: no NULL values in input dataframe")

        #check if folder to store plot exists
        assert os.path.exists(pth_folder_plot)==True
        logging.info("SUCCESS: Testing perform_eda: plot folder exists")


        perform_eda(df,pth_folder_plot)

        


    except AssertionError as err:
        logging.error("ERROR: dataframe does not exist: {}".format(err))     
    

def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	'''


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''


def test_train_models(train_models):
	'''
	test train_models
	'''


if __name__ == "__main__":
	#test_import('data/bank_data.csv',cls.import_data)
    df=cls.import_data('data/bank_data.csv') 
    test_eda(df,cls.perform_eda,'images/eda/')







