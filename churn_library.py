"""
This module contains churn project

Author: Fabio
Date: Dec. 2021
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    return pd.read_csv(pth)


def perform_eda(df,path_folder_plot):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    #create binary variable
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    #perform plots and store in folder
    plt.figure(figsize=(20,10)) 
    ax=df['Churn'].hist()
    path_plot=f'{path_folder_plot}/churn_hist.png'
    ax.figure.savefig(path_plot)

    plt.figure(figsize=(20,10))        
    ax=df.Marital_Status.value_counts('normalize').plot(kind='bar')
    path_plot=f'{path_folder_plot}/Marital_Status_bar.png'
    ax.figure.savefig(path_plot)

    plt.figure(figsize=(20,10)) 
    ax=sns.distplot(df['Total_Trans_Ct'])
    path_plot=f'{path_folder_plot}/Total_Trans_Ct_bar.png'
    ax.figure.savefig(path_plot)

    plt.figure(figsize=(20,10)) 
    ax=sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    path_plot=f'{path_folder_plot}/corr.png'
    ax.figure.savefig(path_plot)    


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    #iterate over list of category variables
    for cat in category_lst:
        cat_lst = []

        #group by current category and get mean of response variable (usually CHURN)
        cat_groups = df.groupby(cat).mean()[response]

        #create list of values
        for val in df[cat]:
                cat_lst.append(cat_groups.loc[val])

        #append current list as new column on input dataframe
        df[f'{cat}_{response}'] = cat_lst 

    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']

    X = df[keep_cols]
    y = df[response]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass

if __name__ == "__main__":
	import_data('ok')