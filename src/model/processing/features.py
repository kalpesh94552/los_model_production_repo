from sklearn.preprocessing import LabelEncoder
import pandas as pd

def featureEng(train, test):
    #Replacing NA values in Bed Grade Column for both Train and Test datssets
    train['Bed Grade'].fillna(train['Bed Grade'].mode()[0], inplace = True)
    test['Bed Grade'].fillna(test['Bed Grade'].mode()[0], inplace = True)

    #Replacing NA values in  Column for both Train and Test datssets
    train['City_Code_Patient'].fillna(train['City_Code_Patient'].mode()[0], inplace = True)
    test['City_Code_Patient'].fillna(test['City_Code_Patient'].mode()[0], inplace = True)

    # Label Encoding Stay column in train dataset
    le = LabelEncoder()
    train["Stay"] = le.fit_transform(train["Stay"].astype('str'))

    #Imputing dummy Stay column in test datset to concatenate with train dataset
    test["Stay"] = -1
    df = pd.concat([train, test])
    df.shape

    #Label Encoding all the columns in Train and test datasets
    for i in ['Hospital_type_code', 'Hospital_region_code', 'Department',
            'Ward_Type', 'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness', 'Age']:
        le = LabelEncoder()
        df[i] = le.fit_transform(df[i].astype(str))

    return df
