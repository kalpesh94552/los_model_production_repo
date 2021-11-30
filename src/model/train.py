import pathlib
import joblib
from sklearn.metrics import accuracy_score
from processing.data_manager import load_test_dataset, load_train_dataset, test_train_split
from processing.features import featureEng
from processing.model import naive_bayes_model
# from processing.mlflow import mlflow_exp_create
# from config.config import save_mlflow_expID, load_mlflow_expID


# Mlflow imports
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def run_training() -> None:
    """Train the model."""
    # expID = load_mlflow_expID()
    # if expID == "temp":
    #     expID = mlflow_exp_create()
    #     save_mlflow_expID(expID)
    #
    # mlflow.set_experiment(expID)
    artifact_repository = 's3://testmlflows3bucket/'
    Server = 'testmlflowrds.caomvyotl9eq.us-east-2.rds.amazonaws.com'
    Port = '3306'
    Database = 'testmlflow'
    Username = 'admin'
    Password = 'admin123'
    # --backend-store-uri mysql+pymysql://${USERNAME}:${PASSWORD}@${HOST}:${PORT}/${DATABASE}
    db_uri = f'mysql+pymysql://{Username}:{Password}@{Server}:{Port}/{Database}'
    print("db_uri:", db_uri)

    # Provide uri and connect to your tracking server
    mlflow.set_tracking_uri(db_uri)

    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # mlflow.set_tracking_uri("file:///Users/ParidhiA/OneDrive/Medtronic/los/los_model_production_repo/mlruns")

    # Initialize MLflow client
    client = MlflowClient()
    try:
        # Create experiment
        experiment_id = client.create_experiment("LOS_Mlflow_experiment", artifact_location=artifact_repository)
    except:
        # Get the experiment id if it already exists
        experiment_id = client.get_experiment_by_name("LOS_Mlflow_experiment").experiment_id

    # ML Flow Tracking starts
    mlflow.end_run()
    with mlflow.start_run(experiment_id=experiment_id) as run:
        # Read traning data
        train = load_train_dataset()
        test = load_test_dataset()

        # Feature Engineering
        df = featureEng(train, test)

        # Divide into train & test (I was here)
        X_train, X_test, y_train, y_test = test_train_split(df)

        # Fit the model
        model_nb = naive_bayes_model(X_train, y_train)

        # Predict & Evaluate the Model
        prediction_nb = model_nb.predict(X_test)
        acc_score_nb = accuracy_score(prediction_nb, y_test)
        print("Acurracy:", acc_score_nb*100)

        mlflow.log_metric("accuracyNB", acc_score_nb*100)
        mlflow.sklearn.log_model(model_nb, "model_nb")
        mlflow.log_param("param", "param")

        # Save the model & test files
        model_save_path = pathlib.WindowsPath(__file__).parent.joinpath(
            'trained_pkl/').joinpath('naive_bayes_model')
        data_save_path = pathlib.WindowsPath(
            __file__).parents[1].joinpath('data/')
        X_test.to_csv(data_save_path.joinpath('X_test.csv'))
        y_test.to_csv(data_save_path.joinpath('y_test.csv'))

        joblib.dump(model_nb, model_save_path)
        print("Saved the model..!!!")

    # mlflow.end_run()


if __name__ == "__main__":
    run_training()
