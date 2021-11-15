# Mlflow imports
import mlflow
import mlflow.sklearn

def mlflow_exp_create():
    mlflow.set_tracking_uri("./mlruns")
    experiment_id = "LOS_Mlflow"
    mlflow.create_experiment(experiment_id)
    return experiment_id
