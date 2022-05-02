import numpy as np
import pandas as pd
import pathlib
import argparse
import pandas
from sklearn.preprocessing import LabelEncoder

from dkube.sdk import *
inp_dir = "/opt/dkube/in"
out_path = "/opt/dkube/out"


def feature_eng(train):
    #Replacing NA values in Bed Grade Column for both Train and Test datssets
    train['Bed Grade'].fillna(train['Bed Grade'].mode()[0], inplace = True)

    #Replacing NA values in  Column for both Train and Test datssets
    train['City_Code_Patient'].fillna(train['City_Code_Patient'].mode()[0], inplace = True)

    # Label Encoding Stay column in train dataset
    le = LabelEncoder()
    # train["Stay"] = le.fit_transform(train["Stay"].astype('str'))

    #Label Encoding all the columns in Train and test datasets
    for i in ['Hospital_type_code', 'Hospital_region_code', 'Department',
            'Ward_Type', 'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness', 'Age']:
        le = LabelEncoder()
        train[i] = le.fit_transform(train[i].astype(str))

    return train


if __name__ == "__main__":
    ########--- Parse for parameters ---########
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", dest="url", default=None, type=str, help="setup URL")
    parser.add_argument("--fs", dest="fs", required=True, type=str, help="featureset")

    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    fs = FLAGS.fs

    ########--- Get DKube client handle ---########
    dkubeURL = FLAGS.url
    # Dkube user access token for API authentication
    authToken = os.getenv("DKUBE_USER_ACCESS_TOKEN")
    # Get client handle
    api = DkubeApi(URL=dkubeURL, token=authToken)

    ########--- Extract and load data  ---######
    df_los = pd.read_csv(os.path.join(inp_dir, "los.csv"))


    ########--- Feature Engineering ---#######
    df_los = feature_eng(df_los)

    # Commit featureset
    resp = api.commit_featureset(name=fs, df=df_los)
    print("featureset commit response:", resp)