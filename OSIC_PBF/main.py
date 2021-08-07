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

# ## OSIC Pulmonary Fibrosis Progression
#
# ### Predict lung function decline
#
# Link to Kaggle competition: https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression
#
# The work on this notebook primarily follows the work done here https://www.kaggle.com/artkulak/inference-45-55-600-epochs-tuned-effnet-b5-30-ep

# +
import cv2
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

ds.pixel_array.max()

2**11

ds.pixel_array.shape

plt.imshow(ds.pixel_array, cmap=plt.cm.plasma)

# Efficientnet B5 (following the competition winner solution https://towardsdatascience.com/how-i-achieved-the-1st-place-in-kaggle-osic-pulmonary-fibrosis-progression-competition-e410962c4edc)
# Why using B5 and not other Bs?
conv_base = efn.EfficientNetB5(weights='imagenet', include_top=False,)

model = models.Sequential()
model.add(conv_base)
