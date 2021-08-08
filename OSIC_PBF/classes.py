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

# Data generators with Keras: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly 

# +
import numpy as np

from tensorflow.keras.utils import Sequence
# -

path = 'data/'
train_path = path + 'train/'


class ImageGenerator(Sequence):
    
    def __init__(self, ids, dim=(512, 512), batch_size=32, shuffle=True):
        self.ids = ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dim = dim
        self.on_epoch_end()
    
    def on_epoch_end(self):
        "Update indexes after each epoch"
        # TODO may be this needs to go in __len__ if not called properly, or in a callback, see: https://stackoverflow.com/questions/59645556/on-epoch-end-not-called-in-keras-fit-generator
        self.indexes = np.arange(len(self.idx))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
    def __data_generation(self, idx_temp):
        "Generates data containing batch_size samples"
        x = np.empty((self.batch_size, *self.dim))
        
        for i, ID in enumerate(idx_temps):
            x[i,] = dcmread(train_path + 'DCM/' + str(idx_temp) + '.dcm')
        
        return x

    def __len__(self):
        """Each call requests a batch index between 0 and the total number of batches, the latter is speficied in this method
        Common practice is to use the value below so that the model sees each sample at most once
        """   
        return int(np.floor(len(self.idx) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        
        idx_temp = [self.ids[k] for k in indexes]
        
        x = self.__data_generation(idx_temp)
        
        return x






