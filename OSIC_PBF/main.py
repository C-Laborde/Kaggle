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
# The work on this notebook primarily follows the work done here https://www.kaggle.com/larsran/pulmonaryfibrosis-environment/notebook?select=pfutils.py

# +
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pydicom import dcmread
# from tqdm import tqdm

from sklearn.model_selection import train_test_split

from classes import ImageGenerator
from utils import build_model
# -

# #### Load data

path = 'data/'
train_path = path + 'train/'

# FVC values
fvc_train = pd.read_csv(train_path + 'train.csv')
fvc_train = fvc_train.sort_values(by=['Patient', 'Weeks'])

fvc_train.head()

# DCM files
dcm_patients = sorted(os.listdir(train_path + 'DCM/'))
dcm_patients.remove('.DS_Store')

# #### Parameters

# +
# Predicting the slope is making the assumption that the decrease is linear
PREDICT_SLOPE = False

# Image flags
USE_IMAGES = True
DIM = 224
IMG_FEATURES = 22
EFFNET = 'b5'

OPTIMAL_SIGMA_LOSS = False

# Dropout rate
DROP_OUT_RATE = 0
DROP_OUT_LAYERS = [1, 2]

# L2-Regularization
L2_REGULARIZATION = True
REGULARIZATION_CONSTANT = 0.0001

# Amount of features inputted in NN
NUMBER_FEATURES = 10

# Hidden layers
HIDDEN_LAYERS = [32, 32]

# Activation functions (relu, swish, leakyrely)
ACTIVATION_FUNCTION = 'relu'

# Batch normalization
BATCH_NORMALIZATION = False
PRE_BATCH_NORMALIZATION = False
BATCH_RENORMALIZATION = False

# Train length
EPOCHS = 500

config = dict(PREDICT_SLOPE=PREDICT_SLOPE, USE_IMAGES=USE_IMAGES, DIM=DIM, IMG_FEATURES=IMG_FEATURES,
              EFFNET=EFFNET, OPTIMAL_SIGMA_LOSS=OPTIMAL_SIGMA_LOSS, DROP_OUT_RATE=DROP_OUT_RATE,
              DROP_OUT_LAYERS=DROP_OUT_LAYERS, L2_REGULARIZATION=L2_REGULARIZATION,
              REGULARIZATION_CONSTANT=REGULARIZATION_CONSTANT, NUMBER_FEATURES=NUMBER_FEATURES,
              HIDDEN_LAYERS=HIDDEN_LAYERS, ACTIVATION_FUNCTION=ACTIVATION_FUNCTION,
              BATCH_NORMALIZATION=BATCH_NORMALIZATION, PRE_BATCH_NORMALIZATION=PRE_BATCH_NORMALIZATION
              BATCH_RENORMALIZATION=BATCH_RENORMALIZATION, EPOCHS=EPOCHS)
# -

# #### Build model

# +
b_effnet = 'b5'

model = build_model(b_effnet)
# -

# #### Prepare data

train, val = train_test_split(fvc_train.Patient.unique(), shuffle=True, train_size=0.8)

# We have to match rows fvc data for a specific patient with his/her corresponding DCM data
x = []
rows = []
for p in val:
    



def get_tab(df):
    vector = [(df.Age.values[0] - 30) / 30] 
    
    if df.Sex.values[0] == 'male':
       vector.append(0)
    else:
       vector.append(1)
    
    if df.SmokingStatus.values[0] == 'Never smoked':
        vector.extend([0,0])
    elif df.SmokingStatus.values[0] == 'Ex-smoker':
        vector.extend([1,1])
    elif df.SmokingStatus.values[0] == 'Currently smokes':
        vector.extend([0,1])
    else:
        vector.extend([1,0])
    return np.array(vector) 


get_tab(fvc_train[fvc_train.Patient == "ID00422637202311677017371"])


