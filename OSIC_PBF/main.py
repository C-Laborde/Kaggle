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

import matplotlib.pyplot as plt
import pandas as pd
from pydicom import dcmread

path = 'data/'
train_path = path + 'train/'

# FVC values
fvc = pd.read_csv(train_path + 'train.csv')

fvc.head()

ds = dcmread(train_path + 'ID00184637202242062969203/1.dcm')

plt.imshow(ds.pixel_array, cmap=plt.cm.gray)

type(ds)

ds.PatientName


