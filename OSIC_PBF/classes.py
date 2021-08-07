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

from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    
    def __init__(self, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle


