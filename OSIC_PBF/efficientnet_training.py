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

# +
fvc_train['SmokingStatus'] = fvc_train['SmokingStatus'].astype('category')
fvc_train['SmokingStatusCat'] = fvc_train['SmokingStatus'].cat.codes

fvc_train['Sex'] = fvc_train['Sex'].astype('category')
fvc_train['SexCat'] = fvc_train['Sex'].cat.codes
# -

# DCM files
dcm_patients = sorted(os.listdir(train_path + 'DCM/'))
dcm_patients.remove('.DS_Store')

patients = fvc_train["Patient"].unique()

patients[0]

# +
A = {}
P = []
# Check TAB!!

for p in patients:
    sub = fvc_train[fvc_train['Patient'] == p]
    fvc = sub.FVC.values
    weeks = sub.Weeks.values
    c = np.vstack([weeks, np.ones(len(weeks))]).T
    
    # Linear equation, least-squares solution
    # Why fitting a straight line to fvc trend?
    a, b = np.linalg.lstsq(c, fvc, rcond=None)[0]
    
    A[p] = a
    P = p
# -


