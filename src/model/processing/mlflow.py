
# Mlflow imports
import mlflow
import mlflow.sklearn

def mlflow_exp_create():
    mlflow.set_tracking_uri("./mlruns")
    experiment_id = datetime.now().strftime('%Y%m-%d%H-%M%S-')
    experiment_name = "HealtCareAnalytics"
    experiment_id = "experiment_"+ experiment_id+ experiment_name
    mlflow.create_experiment(experiment_id)

    return experiment_id
