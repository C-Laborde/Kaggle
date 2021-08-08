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

import efficientnet.tfkeras as efn


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
