"""
This module contains churn project

Author: Fabio
Date: Dec. 2021
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import shap
from sklearn.metrics import RocCurveDisplay
import logging


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    try:
        df=pd.read_csv(pth)

        return df
    except BaseException as err:
        print('ERROR: import_data: {}'.format(err))
        return pd.DataFrame()


def perform_eda(df,path_folder_plot):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    try:
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
    except BaseException as err:
        print('ERROR: perform_eda: {}'.format(err))



def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    try:   

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
    except BaseException as err:
        print('ERROR: encoder_helper: {}'.format(err))

        #return empty dataframe
        return pd.DataFrame()



def perform_feature_engineering(df, keep_cols, response):
    '''
    input:
              df: pandas dataframe
              keep_cols: list of columns to create X dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    try:
        X = df[keep_cols]
        y = df[response]

        #split input dataframe
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)  

        #return dataframes
        return X_train, X_test, y_train, y_test
    except BaseException as err:
        print('ERROR: perform_feature_engineering: {}'.format(err))
        return pd.DataFrame(), pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
        

    


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
    # scores
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))
    

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
    try:
        # Calculate feature importances
        importances = model.best_estimator_.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        
        # Rearrange feature names so they match the sorted feature importances
        names = [X_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20,5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(X_data.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(X_data.shape[1]), names, rotation=90)

        plt.savefig(output_pth)
    except BaseException as err:
        print(err)


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
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()
    param_grid = { 
        'n_estimators': [20, 50],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,10],
        'criterion' :['gini', 'entropy']
        }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    
    lrc.fit(X_train, y_train)
    
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # save report 
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    # 
    lrc_plot = RocCurveDisplay.from_estimator(lrc, X_test, y_test)
    lrc_plot.plot()

    plt.savefig('images/results/lrc_plot.png')

    # plots
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    ax.figure.savefig('images/results/gca.png')

    rfc_disp = RocCurveDisplay.from_estimator(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    rfc_disp.plot(ax=ax, alpha=0.8)
    plt.savefig('images/results/rfc.png')

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    #load models
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    # shape values
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    plt.savefig('images/eda/shap_summary.png')

    # feature importance
    feature_importance_plot(cv_rfc,X_train,f'images/results/cv_rfc.png')

    # iterate over list of models
    list_models=[{'name':'Logistic_Regressions','model':lr_model,
                        'y_test':y_test_preds_lr,
                        'y_train':y_train_preds_rf},
                {'name':'Random_Forest','model':rfc_model,
                        'y_test':y_test_preds_rf,
                        'y_train':y_train_preds_rf}]

    # iterate over list of models
    for model_ in list_models:
        print(model_)
        plt.rc('figure', figsize=(5, 5))
        #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
        plt.text(0.01, 1.25, str(f'{model_["name"]} Train'), {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.05, str(classification_report(y_test, model_['y_test'])), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str(f'{model_["name"]}Test'), {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.7, str(classification_report(y_train, model_['y_train'])), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.savefig(f'images/results/{model_["name"]}.png')


if __name__ == "__main__":
    df=import_data('data/bank_data.csv')
        
    #perform eda
    perform_eda(df,'images/eda')

    # encode   
    df_encode=encoder_helper(df,['Gender','Education_Level','Marital_Status','Income_Category','Card_Category'],'Churn')
