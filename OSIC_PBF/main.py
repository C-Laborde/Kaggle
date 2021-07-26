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

# +
import matplotlib.pyplot as plt
import os
import pandas as pd
from pydicom import dcmread

import efficientnet.tfkeras as efn
from tensorflow.keras import models
# -

path = 'data/'
train_path = path + 'train/'

# FVC values
fvc = pd.read_csv(train_path + 'train.csv')
fvc = fvc.sort_values(by=['Patient', 'Weeks'])

fvc.head()

# DCM files
dcm_patients = sorted(os.listdir(train_path + 'DCM/'))
dcm_patients.remove('.DS_Store')

# I want to check if all patients in fvc data has dcm data and viceversa
fvc_patients = list(sorted(fvc.Patient.unique()))
dcm_patients == fvc_patients

ds = dcmread(train_path + 'DCM/' + 'ID00184637202242062969203/1.dcm')

plt.imshow(ds.pixel_array, cmap=plt.cm.gray)

# Efficientnet B5 (following the competition winner solution https://towardsdatascience.com/how-i-achieved-the-1st-place-in-kaggle-osic-pulmonary-fibrosis-progression-competition-e410962c4edc)
# Why using B5 and not other Bs?
conv_base = efn.EfficientNetB5(weights='imagenet', include_top=False,)

model = models.Sequential()
model.add(conv_base)
