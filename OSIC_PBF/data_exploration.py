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

# ### Description
#
# Notebook to explore how the data looks like

import matplotlib.pyplot as plt
import pandas as pd

path = 'data/'
train_path = path + 'train/'

# FVC values
fvc = pd.read_csv(train_path + 'train.csv')
fvc = fvc.sort_values(by=['Patient', 'Weeks'])

fvc.head()

print(fvc['Sex'].unique())
print(fvc['SmokingStatus'].unique())
print("Number of patients: ", fvc.Patient.unique().shape)

fvc.isnull().sum()

# +
# We plot FVC for some patients to have an idea of the behaviour

for p in fvc.Patient.unique()[10:30]:
    sub = fvc[fvc["Patient"] == p]
    plt.plot(sub.Weeks, sub.FVC, '.-')
# -


