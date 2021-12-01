import pandas as pd
import pathlib
import joblib
from flask import Flask, render_template
from sklearn.metrics import accuracy_score

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/predict")
def make_prediction():
    model_save_path = pathlib.WindowsPath(
        __file__).parent.joinpath('src/model/trained_pkl/naive_bayes_model')
    model_nb = joblib.load(model_save_path)

    print("Trained model: " + str(model_nb.get_params()))

    data_save_path = pathlib.WindowsPath(__file__).parent.joinpath('src/data')
    X_test = pd.read_csv(data_save_path.joinpath('X_test.csv'))
    X_test.drop(X_test.columns[0], axis=1, inplace=True)
    y_test = pd.read_csv(data_save_path.joinpath('y_test.csv'))
    y_test.drop(y_test.columns[0], axis=1, inplace=True)

    prediction_nb = model_nb.predict(X_test)
    acc_score_nb = round(accuracy_score(prediction_nb, y_test)*100, 2)
    print("Prediction")
    print("Acurracy:", acc_score_nb)
    # return "<p>Prediction!</p>"

    return render_template("index.html", accuracy=acc_score_nb, expid=1, runid=2)


if __name__ == "__main__":
    app.run(debug=True)
