import numpy as np
import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split

file_path = pathlib.WindowsPath(__file__).parent.parent.parent.joinpath('data/')
test_path = file_path.joinpath('test.csv')
train_path = file_path.joinpath('train.csv')

# Importing datasets
def load_test_dataset() -> pd.DataFrame:
    _data = pd.read_csv(test_path)
    return _data

# Importing datasets
def load_train_dataset() -> pd.DataFrame:
    _data = pd.read_csv(train_path)
    return _data   

def get_countid_enocde(train, test, cols, name):
    temp = train.groupby(cols)['case_id'].count().reset_index().rename(columns = {'case_id': name})
    temp2 = test.groupby(cols)['case_id'].count().reset_index().rename(columns = {'case_id': name})
    train = pd.merge(train, temp, how='left', on= cols)
    test = pd.merge(test,temp2, how='left', on= cols)
    train[name] = train[name].astype('float')
    test[name] = test[name].astype('float')
    train[name].fillna(np.median(temp[name]), inplace = True)
    test[name].fillna(np.median(temp2[name]), inplace = True)
    return train, test

def test_train_split(df):  
    #Spearating Train and Test Datasets
    train = df[df['Stay']!=-1]
    test = df[df['Stay']==-1]

    train, test = get_countid_enocde(train, test, ['patientid'], name = 'count_id_patient')
    train, test = get_countid_enocde(train, test, 
                                    ['patientid', 'Hospital_region_code'], name = 'count_id_patient_hospitalCode')
    train, test = get_countid_enocde(train, test, 
                                    ['patientid', 'Ward_Facility_Code'], name = 'count_id_patient_wardfacilityCode')

    # Droping duplicate columns
    test1 = test.drop(['Stay', 'patientid', 'Hospital_region_code', 'Ward_Facility_Code'], axis =1)
    train1 = train.drop(['case_id', 'patientid', 'Hospital_region_code', 'Ward_Facility_Code'], axis =1)

    # Splitting train data for Naive Bayes and XGBoost
    X1 = train1.drop('Stay', axis =1)
    y1 = train1['Stay']
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size =0.20, random_state =100)

    return X_train, X_test, y_train, y_test