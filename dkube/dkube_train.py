import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import requests, os
import argparse

import mlflow
from sklearn import metrics
from dkube.sdk import *
import joblib

inp_path = "/opt/dkube/in"
out_path = "/opt/dkube/out"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", dest="url", default=None, type=str, help="setup URL")
    parser.add_argument("--fs", dest="fs", required=True, type=str, help="featureset")

    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    dkubeURL = FLAGS.url
    fs = FLAGS.fs

    ########--- Read features from input FeatureSet ---########

    # Featureset API
    authToken = os.getenv("DKUBE_USER_ACCESS_TOKEN")
    # Get client handle
    api = DkubeApi(URL=dkubeURL, token=authToken)

    # Read features
    feature_df = api.read_featureset(name = fs)  # output: data

    ########--- Train ---########
    # feature_df = feature_df.drop(['case_id', 'patientid', 'Hospital_region_code', 'Ward_Facility_Code'], axis =1)
    los_input = feature_df.drop('Stay', axis =1)
    los_target = feature_df['Stay']

    # Splitting train data
    x_train, x_test, y_train, y_test = train_test_split(los_input, los_target, test_size =0.20, random_state =100) 

    #fit linear model to the train set data
    classifier_nb = GaussianNB()
    model_nb = classifier_nb.fit(x_train, y_train)

    y_pred = model_nb.predict(x_test)
    # y_pred_train = linReg.predict(x_train)    # Predict on train data.
    # y_pred_train[y_pred_train < 0] = y_pred_train.mean()
    # y_pred = linReg.predict(x_test)   # Predict on test data.
    # y_pred[y_pred < 0] = y_pred.mean()
    
    #######--- Calculating metrics ---############
    acc_score_nb = accuracy_score(y_pred, y_test)
    print("Accuracy:", acc_score_nb)
    # mlflow.log_metric("accuracyNB", acc_score_nb*100)
    # mlflow.sklearn.log_model(model_nb, "model_nb")
    # mlflow.log_param("param", "param")

    # mae = metrics.mean_absolute_error(y_test, y_pred)
    # mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
    # mse = metrics.mean_squared_error(y_test, y_pred)
    # rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
    # r2 = metrics.r2_score(y_test, y_pred)

    # print('Mean Absolute Error:', mae)  
    # print('Mean Squared Error:', mse)  
    # print('Root Mean Squared Error:', rmse)
    # print('R2 score:', r2)
    # print("MAPE", mape)

    ########--- Logging metrics into Dkube via mlflow ---############
    mlflow.log_metric("Accuracy", acc_score_nb)
    # mlflow.log_metric("MAPE", mape)
    # mlflow.log_metric("MSE", mse)
    # mlflow.log_metric("RMSE", rmse)
    # mlflow.log_metric("R2", r2)

    # Exporting model
    filename = os.path.join(out_path, "model.joblib")
    joblib.dump(model_nb, filename)