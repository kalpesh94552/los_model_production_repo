import numpy as np
import pandas as pd 
import pathlib
import joblib
from sklearn.metrics import accuracy_score

def make_prediction():
    model_save_path = pathlib.WindowsPath(__file__).parent.joinpath('trained_pkl/').joinpath('naive_bayes_model')
    model_nb = joblib.load(model_save_path)

    data_save_path = pathlib.WindowsPath(__file__).parents[1].joinpath('data/')
    X_test= pd.read_csv(data_save_path.joinpath('X_test.csv'))
    X_test.drop(X_test.columns[0], axis=1,inplace=True)
    y_test= pd.read_csv(data_save_path.joinpath('y_test.csv'))
    y_test.drop(y_test.columns[0], axis=1,inplace=True)

    prediction_nb = model_nb.predict(X_test)
    acc_score_nb = accuracy_score(prediction_nb,y_test)
    print("Prediction")
    print("Acurracy:", acc_score_nb*100)

    # Naive Bayes
    # print("Prediction")
    # pred_nb = classifier_nb.predict(test1.iloc[:,1:])
    # result_nb = pd.DataFrame(pred_nb, columns=['Stay'])
    # result_nb['case_id'] = test1['case_id']
    # result_nb = result_nb[['case_id', 'Stay']]

    # result_nb['Stay'] = result_nb['Stay'].replace({0:'0-10', 1: '11-20', 2: '21-30', 3:'31-40', 4: '41-50', 5: '51-60', 6: '61-70', 7: '71-80', 8: '81-90', 9: '91-100', 10: 'More than 100 Days'})


if __name__ == "__main__":
    make_prediction()