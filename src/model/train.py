import pathlib
import numpy as np
import pandas as pd 
import joblib
from sklearn.metrics import accuracy_score
from processing.data_manager import load_test_dataset, load_train_dataset, test_train_split
from processing.features import featureEng
from processing.model import naive_bayes_model
import time
from datetime import datetime

# Mlflow imports
import mlflow
import mlflow.sklearn

def run_training() -> None:
    """Train the model."""
    mlflow.set_tracking_uri("./mlruns")
    experiment_id = datetime.now().strftime('%Y%m-%d%H-%M%S-')
    experiment_name = "HealtCareAnalytics"
    experiment_id = "experiment_"+ experiment_id+ experiment_name
    mlflow.create_experiment(experiment_id)

    mlflow.set_experiment(experiment_id)
    #ML Flow Tracking starts
    mlflow.end_run()
    mlflow.start_run()

    #Read traning data
    train = load_train_dataset()
    test = load_test_dataset()

    #Feature Engineering
    df = featureEng(train, test)

    # Divide into train & test (I was here)
    X_train, X_test, y_train, y_test = test_train_split(df)

    # Fit the model
    model_nb = naive_bayes_model(X_train, y_train)
    
    mlflow.log_param("param","param")

    # Predict & Evaluate the Model
    prediction_nb = model_nb.predict(X_test)
    acc_score_nb = accuracy_score(prediction_nb,y_test)
    print("Acurracy:", acc_score_nb*100)
    mlflow.log_metric("accuracyNB",acc_score_nb*100)

    # Save the model & test files
    model_save_path = pathlib.WindowsPath(__file__).parent.joinpath('trained_pkl/').joinpath('naive_bayes_model')
    data_save_path = pathlib.WindowsPath(__file__).parents[1].joinpath('data/')
    X_test.to_csv(data_save_path.joinpath('X_test.csv'))
    y_test.to_csv(data_save_path.joinpath('y_test.csv'))

    joblib.dump(model_nb , model_save_path)
    print("Saved the model..!!!")

    mlflow.end_run()

if __name__ == "__main__":
    run_training()