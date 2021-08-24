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
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

# from classes import classes
# from utils import build_model

# -

def get_efficientnet(model, shape):
    models = {'b0': efn.EfficientNetB0(input_shape=shape, weights=None, include_top=False),
              'b1': efn.EfficientNetB1(input_shape=shape, weights=None, include_top=False),
              'b2': efn.EfficientNetB2(input_shape=shape, weights=None, include_top=False),
              'b3': efn.EfficientNetB3(input_shape=shape, weights=None, include_top=False),
              'b4': efn.EfficientNetB4(input_shape=shape, weights=None, include_top=False),
              'b5': efn.EfficientNetB5(input_shape=shape, weights=None, include_top=False),
              'b6': efn.EfficientNetB6(input_shape=shape, weights=None, include_top=False),
              'b7': efn.EfficientNetB7(input_shape=shape, weights=None, include_top=False)}

    return models[model]


def build_model(config):
    inp = Input(shape=shape)
    base = get_efficientnet(b_model, shape)
    x = base(inp)
    x = GlobalAveragePooling2D()(x)
    
    inp2 = Input(shape=(4,))
    x2 = GaussianNoise(0.2)(inp2)
    x = Concatenate()([x, x2])
    x = Dropout(0.5)(x)
    x = Dense(1)(x)
    
    model = Model([inp, inp2], x)
    
    return model


from tensorflow.keras.utils import Sequence
class ImageDataGenerator(Sequence):
    
    def __init__(self, ids, dim=(512, 512, 1), batch_size=32, shuffle=True):
        self.ids = ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dim = dim
        self.on_epoch_end()
    
    def on_epoch_end(self):
        "Update indexes after each epoch"
        # TODO may be this needs to go in __len__ if not called properly, or in a callback, see: https://stackoverflow.com/questions/59645556/on-epoch-end-not-called-in-keras-fit-generator
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
    def __data_generation(self, idx_temp):
        "Generates data containing batch_size samples"
        x = np.empty((self.batch_size, *self.dim))
        
        for i, ID in enumerate(idx_temps):
            # Should we resize and normalize here?
            x[i,] = dcmread(train_path + 'DCM/' + str(idx_temp) + '.dcm').pixel_array
        
        return x

    def __len__(self):
        """Each call requests a batch index between 0 and the total number of batches, the latter is speficied in this method
        Common practice is to use the value below so that the model sees each sample at most once
        """   
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        
        idx_temp = [self.ids[k] for k in indexes]
        
        x = self.__data_generation(idx_temp)
        
        return x


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

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

patients = fvc_train["Patient"].unique()[:10]

# </br>
# Linear fit

# +
A = {}
P = []
# Check TAB!!

for p in patients[:10]:
    sub = fvc_train[fvc_train['Patient'] == p]
    fvc = sub.FVC.values
    weeks = sub.Weeks.values
    c = np.vstack([weeks, np.ones(len(weeks))]).T
    
    # Linear equation, least-squares solution
    # Why fitting a straight line to fvc trend?
    a, b = np.linalg.lstsq(c, fvc, rcond=None)[0]
    
    A[p] = a
    P.append(p)
# -

# </br>
# Training

# EPOCHS = 50
EPOCHS = 5
SAVE_BEST = True
B_MODEL = 'b5'
PATIENCE = 5

# +
es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=PATIENCE,
        mode='auto',
        baseline=None,
        restore_best_weights=True)

mcp = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'models/effnet_{EPOCHS}.h5',
        monitor='val_loss',
        save_best_only=SAVE_BEST,
        mode='auto')

rlp = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=PATIENCE,
        mode='auto',
        cooldown=0,
        min_lr=0.0001)

# +
model = build_model(b_model=B_MODEL)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.MeanAbsoluteError())
# history = model.fit_generator(ImageDataGenerator(P),
#                     steps_per_epoch=32,
#                     # Why validation_data = training data??
#                     validation_data=ImageDataGenerator(P),
#                     callbacks=[es, mcp, rlp],
#                     epochs=EPOCHS)

history = model.fit()
folds_history = []
folds_history.append(history.history)
# -

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])


