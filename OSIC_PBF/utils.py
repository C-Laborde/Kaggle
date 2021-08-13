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

import efficientnet.tfkeras as efn
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Dense, Dropout, GaussianNoise, GlobalAveragePooling2D, Input


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


def build_model(b_model=None, shape=(512, 512, 1)):
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


build_model(b_model='b5')


def score(y_pred, y_true, sigma):
    sigma_clipped = max(sigma, 70)
    delta = min(abs(y_pred, y_true), 1000)
    metric = - 2**0.5 * delta / sigma_clipped - np.log(2**0.5 * sigma_clipped)
    # Shou
    return metric


