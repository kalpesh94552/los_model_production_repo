
# Mlflow imports
import mlflow
import mlflow.sklearn
import time
from datetime import datetime

def mlflow_exp_create():
    mlflow.set_tracking_uri("./mlruns")
    experiment_id = "LOS_Mlflow"
    experiment_name = "HealtCareAnalytics"
    mlflow.create_experiment(experiment_id)

    return experiment_id
