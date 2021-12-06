"""
This module contains loggind and testing for churn project

Author: Fabio
Date: 4 Dec. 2021
"""


import os
import logging
import pandas as pd
import churn_library as cls


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data, pth):
    '''
    test data import - this example is completed for you to assist with the other test functions
    input:
          import_data: method to test
          pth: path file
    output:
          df_import: Dataframe
    '''
    try:
        logging.info("INFO: Testing import_data asserts")

        assert os.path.exists(pth) is True, 'path does not exist'

        logging.info("SUCCESS: Testing import_data: path exist")

        df_import = import_data(pth)

        logging.info("SUCCESS: Testing import_data read file")

    except FileNotFoundError as err:
        logging.error(
            "ERROR: Testing import_data: The file wasn't found: %s", err)
    except AssertionError as err:
        logging.error("ERROR: Testing import_data: %s", err)

    try:
        assert df_import.shape[0] > 0, 'zero records'
        assert df_import.shape[1] > 0, 'zero records'

        return df_import

    except AssertionError as err:
        logging.error("ERROR: Testing import_data: %s", err)

        # return empty df_import
        return pd.DataFrame()


def test_eda(perform_eda, df_import, pth_folder_plot):
    '''
    test perform eda function
    input:
              perform_eda: method to test
              df_import: dataframe
              pth_folder_plot: path folder
    output:
              None
    '''
    try:
        logging.info("INFO: Testing perform_eda")

        # check size of dataframe
        assert df_import.shape[0] > 0 and df_import.shape[1] > 0, 'empty df_import'
        logging.info("SUCCESS: Testing perform_eda: dataframe shape > 0")

        # check if exists any null values
        assert df_import.isnull().sum().sum() == 0, 'dataframe contains NULLS'
        logging.info(
            "SUCCESS: Testing perform_eda: no NULL values in input dataframe")

        # check if folder to store plot exists
        assert os.path.exists(pth_folder_plot) is True, 'folder does not exist'
        logging.info("SUCCESS: Testing perform_eda: plot folder exists")

        # since we want to create churn from 'Attrition_Flag' I need to check
        # it
        assert df_import['Attrition_Flag'].dtype == object, 'Attrition_Flag is not string'
        logging.info("SUCCESS: Testing perform_eda: Attrition_Flag is string")

    except AssertionError as err:
        logging.error("ERROR: test_eda: %s", err)

    try:
        perform_eda(df_import, pth_folder_plot)
        logging.info("SUCCESS: Testing perform_eda: plot stored in folder")

        # check if churn column is created
        assert 'Churn' in list(df_import.columns), 'CHURN columns not created'

    except AssertionError as err:
        logging.error("ERROR: test_eda: %s", err)


def test_encoder_helper(encoder_helper, df_import, category_lst, response):
    '''
    test encoder helper
    input:
              perform_eda: method to test
              df_import: dataframe
              category_lst: list variables to encode
              response: variable needed for encoding

    output:
              None
    '''
    try:

        logging.info("INFO: Testing test_encoder_helper")

        # check size of dataframe
        assert df_import.shape[0] > 0 and df_import.shape[1] > 0, 'empty dataframe'
        logging.info(
            "SUCCESS: Testing test_encoder_helper: dataframe shape > 0")

        # check if exists any null values
        assert df_import.isnull().sum().sum() == 0, 'dataframe contains NULLS'
        logging.info(
            "SUCCESS: Testing test_encoder_helper: no NULL values in input dataframe")

        # check if category_lst exist
        assert len(category_lst) > 0, 'category list empty'
        logging.info(
            "SUCCESS: Testing test_encoder_helper: category list does not exist")

        # check if response is string
        assert isinstance(response, object), 'response variable is not string'
        logging.info(
            "SUCCESS: Testing test_encoder_helper: response is string")

        # check if category list are columns in dataframe
        assert all(elem in list(df_import.columns)
                   for elem in category_lst), 'category list values are not in columns'
        logging.info(
            "SUCCESS: Testing test_encoder_helper: category list are columns in dataframe")

        # check if response refers to a column in dataframe
        assert response in list(
            df_import.columns), 'response variable is not in dataframe columns'
        logging.info(
            "SUCCESS: Testing test_encoder_helper: response refers to a columns in dataframe")

    except AssertionError as err:
        logging.error("ERROR: Testing test_encoder_helper: %s", err)

    try:
        # test method
        encoder_return = encoder_helper(df_import, category_lst, response)
        logging.info(
            "SUCCESS: Testing test_encoder_helper: success performing new columns")

        # check if encoder_helper is not empty
        assert encoder_return.shape[0] > 0, 'encoder_return not empty'
        assert encoder_return.shape[1] > 0, 'encoder_return not empty'
        logging.info(
            "SUCCESS: Testing test_encoder_helper: encoder_return is not empty")

        # check if encoder_helper method created Churn columns
        assert all(f'{elem}_{response}' in list(encoder_return.columns)
                   for elem in category_lst), 'churn columns not created'
        logging.info(
            "SUCCESS: Testing test_encoder_helper: all churn columns created correctly")

        return encoder_return
    except AssertionError as err:
        logging.error("ERROR: Testing test_encoder_helper: %s", err)
        # return empty df_import
        return pd.DataFrame()


def test_perform_feature_engineering(
        perform_feature_engineering, df_import, keep_cols, response):
    '''
    test perform_feature_engineering
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    try:
        logging.info("INFO: Testing test_perform_feature_engineering")

        # check size of dataframe
        assert df_import.shape[0] > 0 and df_import.shape[1] > 0, "dataframe shape = 0"
        logging.info(
            "SUCCESS: Testing test_perform_feature_engineering: dataframe shape > 0")

        # check if keep list are columns in dataframe
        assert all(elem in list(df_import.columns)
                   for elem in keep_cols), "keep cols are not in dataframe"
        logging.info(
            "SUCCESS: Testing test_encoder_helper: keep_cols are columns in dataframe")

        # check if response is string
        assert isinstance(response, str), "response is not str"
        logging.info(
            "SUCCESS: Testing test_perform_feature_engineering: response is string")

        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df_import, keep_cols, response)
        logging.info(
            "SUCCESS: Testing test_perform_feature_engineering splitting dataframe")
        print(y_train.shape)
        # check size of dataframe
        assert x_train.shape[0] > 0, "X_train shape = 0"
        assert x_train.shape[1] > 0, "X_train shape = 0"
        # check size of dataframe
        assert x_test.shape[0] > 0, "X_test shape = 0"
        assert x_test.shape[1] > 0, "X_test shape = 0"
        # check size of dataframe
        assert y_train.shape[0] > 0, "y_train shape = 0"
        # check size of dataframe
        assert y_test.shape[0] > 0, "y_test shape = 0"
        logging.info(
            "SUCCESS: Testing test_perform_feature_engineering alla splitted dataframe populated")

        return x_train, x_test, y_train, y_test

    except AssertionError as err:
        logging.error(
            "ERROR: Testing test_perform_feature_engineering: error assertion: %s", err)
        x_train = pd.DataFrame()
        x_test = pd.DataFrame()
        y_train = pd.DataFrame()
        y_test = pd.DataFrame()
        # return empty dataframe
        return x_train, x_test, y_train, y_test


def test_train_models(train_models, x_train, x_test, y_train, y_test):
    '''
    test train_models
    input:
              train_models: method to test
              x_train: x train data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    try:
        logging.info("INFO: Testing test_train_models asserts")

        # check size of dataframe
        assert x_train.shape[0] > 0, "X_train shape = 0"
        assert x_train.shape[1] > 0, "X_train shape = 0"
        # check size of dataframe
        assert x_test.shape[0] > 0, "X_test shape = 0"
        assert x_test.shape[1] > 0, "X_test shape = 0"
        # check size of dataframe
        assert y_train.shape[0] > 0, "y_train shape = 0"
        # check size of dataframe
        assert y_test.shape[0] > 0, "y_test shape = 0"
        logging.info("SUCCESS: Testing test_train_models asserts ok")

       # train models
        train_models(x_train, x_test, y_train, y_test)
        logging.info(
            "SUCCESS: Testing test_train_models model trained correctly")

        # check if images are stored
        assert os.path.exists(
            'images/results/lrc_plot.png') is True, 'path does not exist'
        assert os.path.exists(
            'images/results/gca.png') is True, 'path does not exist'
        assert os.path.exists(
            'images/results/rfc.png') is True, 'path does not exist'
        assert os.path.exists(
            'images/eda/shap_summary.png') is True, 'path does not exist'
        # check if models are stored
        assert os.path.exists(
            './models/rfc_model.pkl') is True, 'path does not exist'
        assert os.path.exists(
            './models/logistic_model.pkl') is True, 'path does not exist'

    except AssertionError as err:
        logging.error(
            "ERROR: Testing test_train_models: error assertion: %s", err)


if __name__ == "__main__":
    # test import
    DF_IMPORT = test_import(cls.import_data, 'data/bank_data.csv',)

    # test eda
    test_eda(cls.perform_eda, DF_IMPORT, 'images/eda/')

    # test encoder
    test_encoder_helper(cls.encoder_helper,
                        DF_IMPORT,
                        ['Gender',
                         'Education_Level',
                         'Marital_Status',
                         'Income_Category',
                         'Card_Category'],
                        'Churn')

    # test perform feature engineering
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cls.perform_feature_engineering,
        DF_IMPORT, ['Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn'], 'Churn')

    # test train model
    test_train_models(cls.train_models,
                      X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
