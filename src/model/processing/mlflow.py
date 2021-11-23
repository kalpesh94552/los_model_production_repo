# Mlflow imports
import mlflow
import mlflow.sklearn

def mlflow_exp_create():
    mlflow.set_tracking_uri("./mlruns")
    experiment_id = "LOS_Mlflow_demo"
    Server='CM1VA204\SQLEXPRESS2019'
    Database='mlflow'
    Driver='ODBC Driver 17 for SQL Server'
    db_uri= f'mssql://@{Server}/{Database}?driver={Driver}'
    print("db_uri:", db_uri)
    mlflow.create_experiment(experiment_id, artifact_location=db_uri)
    return experiment_id
