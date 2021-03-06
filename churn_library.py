"""
This module contains churn project

Author: Fabio
Date: 4Dec. 2021
"""
import os
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, RocCurveDisplay


# logging.basicConfig(
#    filename='./logs/churn_library_prod.log',
#    level=logging.INFO,
#    filemode='w',
#    format='%(name)s - %(levelname)s - %(message)s')


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df_read: pandas dataframe
    '''
    # create empty df
    df_read = pd.DataFrame()

    # check if paths exists
    if os.path.exists(pth):
        df_read = pd.read_csv(pth)

    return df_read


def perform_eda(df_input, path_folder_plot):
    '''
    perform eda on df_input and save figures to images folder
    input:
            df_input: pandas dataframe

    output:
            None
    '''
    # create binary variable
    df_input['Churn'] = df_input['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # perform plots and store in folder
    fig = plt.figure(figsize=(20, 10))
    ax_plot = df_input['Churn'].hist()
    path_plot = f'{path_folder_plot}/churn_hist.png'
    ax_plot.figure.savefig(path_plot)
    plt.close(fig)

    fig = plt.figure(figsize=(20, 10))
    ax_plot = df_input.Marital_Status.value_counts(
        'normalize').plot(kind='bar')
    path_plot = f'{path_folder_plot}/Marital_Status_bar.png'
    ax_plot.figure.savefig(path_plot)
    plt.close(fig)

    fig = plt.figure(figsize=(20, 10))
    ax_plot = sns.distplot(df_input['Total_Trans_Ct'])
    path_plot = f'{path_folder_plot}/Total_Trans_Ct_bar.png'
    ax_plot.figure.savefig(path_plot)
    plt.close(fig)

    fig = plt.figure(figsize=(20, 10))
    ax_plot = sns.heatmap(
        df_input.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    path_plot = f'{path_folder_plot}/corr.png'
    ax_plot.figure.savefig(path_plot)
    plt.close(fig)


def encoder_helper(df_input, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df_input: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional]

    output:
            df_input: pandas dataframe with new columns for
    '''

    # check if category list is not empty and response is not null
    if ((len(category_lst) > 0) and (isinstance(response, object))):
        # iterate over list of category variables
        for cat in category_lst:
            cat_lst = []

            # group by current category and get mean of response variable
            # (usually CHURN)
            cat_groups = df_input.groupby(cat).mean()[response]

            # create list of values
            for val in df_input[cat]:
                cat_lst.append(cat_groups.loc[val])

            # append current list as new column on input dataframe
            df_input[f'{cat}_{response}'] = cat_lst

    return df_input


def perform_feature_engineering(df_input, keep_cols, response):
    '''
    input:
              df_input: pandas dataframe
              keep_cols: list of columns to create X dataframe
              response: string of response name [optional]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # create empty dataframes
    x_train = pd.DataFrame()
    x_test = pd.DataFrame()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()

    # check if keep cols is not null, if response variable is string
    # and if df_input input is not empty
    if ((len(keep_cols) > 0) and (isinstance(response, object)) and
            (df_input.shape[0] > 0) and (df_input.shape[1] > 0)):

        x_df = df_input[keep_cols]
        y_df = df_input[response]

        # split input dataframe
        x_train, x_test, y_train, y_test = train_test_split(
            x_df, y_df, test_size=0.3, random_state=42)

    # return dataframes
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_model,
                                y_test_model):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_model: training predictions from model
            y_test_preds_model: test predictions from model

    output:
             None
    '''
    # scores
    print(classification_report(y_test, y_test_model))
    print(classification_report(y_train, y_train_model))


def feature_importance_plot(model, x_data, output_pth):
    '''
        creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
             '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    fig = plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-ax_plotis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth)
    plt.close(fig)


def train_models(x_train, x_test, y_train, y_test):
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
        'n_estimators': [20, 30],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [2, 3],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # iterate over list of models
    list_models = [{'name': 'Logistic_Regressions',
                    'y_test': y_test,
                    'y_train': y_train,
                   'y_test_model': y_test_preds_lr,
                    'y_train_model': y_train_preds_lr},
                   {'name': 'Random_Forest',
                    'y_test': y_test,
                    'y_train': y_train,
                    'y_test_model': y_test_preds_rf,
                    'y_train_model': y_train_preds_rf
                    }]
    for mod in list_models:
        # save report
        classification_report_image(mod['y_train'],
                                    mod['y_test'],
                                    mod['y_train_model'],
                                    mod['y_test_model'])
    # plot roc
    fig = plt.figure(figsize=(20, 10))
    lrc_plot = RocCurveDisplay.from_estimator(lrc, x_test, y_test)
    lrc_plot.plot()
    plt.savefig('images/results/lrc_plot.png')
    plt.close(fig)

    # plots
    fig = plt.figure(figsize=(20, 10))
    ax_plot = plt.gca()
    ax_plot.figure.savefig('images/results/gca.png')
    plt.close(fig)

    fig = plt.figure(figsize=(20, 10))
    rfc_disp = RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_, x_test, y_test, ax=ax_plot, alpha=0.8)
    rfc_disp.plot(ax=ax_plot, alpha=0.8)
    plt.savefig('images/results/rfc.png')
    plt.close(fig)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # load models
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    # shape values
    fig = plt.figure(figsize=(20, 10))
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar")
    plt.savefig('images/eda/shap_summary.png')
    plt.close(fig)

    # feature importance
    feature_importance_plot(cv_rfc, x_train, 'images/results/cv_rfc.png')

    # iterate over list of models
    list_models = [{'name': 'Logistic_Regressions', 'model': lr_model,
                   'y_test': y_test_preds_lr,
                    'y_train': y_train_preds_lr},
                   {'name': 'Random_Forest', 'model': rfc_model,
                    'y_test': y_test_preds_rf,
                    'y_train': y_train_preds_rf}]

    # iterate over list of models
    for model_ in list_models:

        fig = plt.rc('figure', figsize=(5, 5))
        # plt.text(0.01, 0.05, str(model.summary()),
        # {'fontsize': 12}) old
        # approach
        plt.text(0.01, 1.25, str(f'{model_["name"]} Train'), {
                 'fontsize': 10}, fontproperties='monospace')
        # approach improved by OP -> monospace!
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    y_test, model_['y_test'])), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, str(f'{model_["name"]}Test'), {
                 'fontsize': 10}, fontproperties='monospace')
        # approach improved by OP -> monospace!
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    y_train, model_['y_train'])), {
                'fontsize': 10}, fontproperties='monospace')
        plt.savefig(f'images/results/{model_["name"]}.png')
        plt.close(fig)


if __name__ == "__main__":
    # read data
    DF_INPUT = import_data('data/bank_data.csv')

    # perform eda
    perform_eda(DF_INPUT, 'images/eda')

    # encode
    DF_ENCODE = encoder_helper(DF_INPUT,
                               ['Gender',
                                'Education_Level',
                                'Marital_Status',
                                'Income_Category',
                                'Card_Category'],
                               'Churn')

    # feature engineering
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        DF_ENCODE, ['Customer_Age', 'Dependent_count', 'Months_on_book',
                    'Total_Relationship_Count', 'Months_Inactive_12_mon',
                    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                    'Income_Category_Churn', 'Card_Category_Churn'], 'Churn')
    # train model
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
