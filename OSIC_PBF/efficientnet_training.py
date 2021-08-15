# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import os
import pandas as pd

# #### Load data

path = 'data/'
train_path = path + 'train/'

# FVC values
fvc_train = pd.read_csv(train_path + 'train.csv')
fvc_train = fvc_train.sort_values(by=['Patient', 'Weeks'])

fvc_train['SmokingStatus'] = fvc_train['SmokingStatus'].astype('category')

# DCM files
dcm_patients = sorted(os.listdir(train_path + 'DCM/'))
dcm_patients.remove('.DS_Store')

vector = [(df.Age.values[0] - 30) / 30]


def get_tab(df):
    vector = [(df.Age.values[0] - 30) / 30] 
    print("vector 0 ", vector)
    
    if df.Sex.values[0].lower() == 'male':
       vector.append(0)
    else:
       vector.append(1)
    print("vector 1 ", vector)
    
    if df.SmokingStatus.values[0] == 'Never smoked':
        vector.extend([0,0])
    elif df.SmokingStatus.values[0] == 'Ex-smoker':
        vector.extend([1,1])
    elif df.SmokingStatus.values[0] == 'Currently smokes':
        vector.extend([0,1])
    else:
        vector.extend([1,0])
    print("vector 2 ", vector)

    print("vector 3 ", np.array(vector))
    return np.array(vector)


patients = fvc_train["Patient"].unique()

patients[0]

for p in patients[0:1]:
    sub = fvc_train[fvc_train['Patient'] == p]
    get_tab(sub)


