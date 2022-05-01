import pandas as pd
import pathlib
import joblib
import pytest
from sklearn.metrics import accuracy_score


@pytest.fixture
def test_acc_score_nb():
    return 0.3


def test_accuracy(test_acc_score_nb):
    # acc_score_nb = nb_prediction_acc()
    # assert acc_score_nb >= test_acc_score_nb
    model_save_path = pathlib.WindowsPath(__file__).parents[1].joinpath(
        'src/model/trained_pkl/naive_bayes_model')
    model_nb = joblib.load(model_save_path)

    data_save_path = pathlib.WindowsPath(
        __file__).parents[1].joinpath('src/data/')
    X_test = pd.read_csv(data_save_path.joinpath('X_test.csv'))
    X_test.drop(X_test.columns[0], axis=1, inplace=True)
    y_test = pd.read_csv(data_save_path.joinpath('y_test.csv'))
    y_test.drop(y_test.columns[0], axis=1, inplace=True)

    prediction_nb = model_nb.predict(X_test)
    acc_score_nb = accuracy_score(prediction_nb, y_test)
    print("Prediction")
    print("Acurracy:", acc_score_nb*100)

    assert acc_score_nb >= test_acc_score_nb
