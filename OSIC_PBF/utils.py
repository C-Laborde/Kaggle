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

path = 'data/'
train_path = path + 'train/'


# +

def get_train_data(df, train_on_backward_weeks=False, use_images=True):
    index = 0
    train = pd.DataFrame()

    for patient in df.Patient.unique()[:2]:
        sub = df[df.Patient == patient]
        weeks = sub['Weeks']
        for weektarget in weeks:
            dftemp = sub[(sub.Weeks != weektarget) & ((sub.Weeks<weektarget) | (train_on_backward_weeks))]
            dftemp = dftemp.assign(WeekTarget = weektarget)
            dftemp = dftemp.assign(TargetFVC = sub[sub.Weeks == weektarget]['FVC'].values[0])
            dftemp = dftemp.assign(PatientIndex = index)
            train = train.append(dftemp, ignore_index = True)
        index += 1
    
        if use_images:


# +
import pandas as pd
df = pd.read_csv(train_path + 'train.csv')

get_train_data(df)


# -

def get_efficientnet(effnet):
    models = {'b0': efn.EfficientNetB0,
              'b1': efn.EfficientNetB1,
              'b2': efn.EfficientNetB2,
              'b3': efn.EfficientNetB3,
              'b4': efn.EfficientNetB4,
              'b5': efn.EfficientNetB5,
              'b6': efn.EfficientNetB6,
              'b7': efn.EfficientNetB7}
    return models[model]


def get_activation_function(name):

    actfunc_mapping = {'relu': tf.keras.activations.relu,
                       'swish': tf.keras.activations.swish,
                       'leakyrelu': tf.keras.activations.leakyrelu}
    
    return actfunc_mapping[name]


def build_model(config):

    # Parameteres
    actfunc = get_activation_function[config['ACTIVATION_FUNCTION']]
    drop_out_layers = config['DROP_OUT_LAYERS']
    drop_out_rate = config['DROP_OUT_RATE']
    effnet = config['EFFNET']
    hidden_layers = config['HIDDEN_LAYERS']
    img_features = config['IMAGE_FEATURES']
    l2_regularization = config['L2_REGULARIZATION']
    optimal_sigma_loss = config['OPTIMAL_SIGMA_LOSS']
    optimal_sigma_loss_function = config['OPTIMAL_SIGMA_LOSS_FUNCTION']
    output_normalization = config['OUTPUT_NORMALIZATION']
    pre_batch_normalization = config['PRE_BATCH_NORMALIZATION']
    predict_slope = config['PREDICT_SLOPE']
    size = config['NUMBER_FEATURES']
    use_imgs = config['USE_IMAGES']

    # Keras Tensor instantiation
    metadata = tf.keras.layers.Input(shape=(size), name='meta_data')
    inputs = [metadata]
    
    x = metadata

    if use_imags:
        image = tf.keras.layers.Input(shape=(dim, dim, 3), name='image')
        inputs.append(image)
        # TOFO When to use include_top true or false?
        # TOFO When to use each pooling type?
        # TOFO What are these weights? They won't change when training?
        base = get_efficientnet(effnet)(input_shape=(dim, dim, 3), weights='imagenet', include_top=False, pooling='avg')
        y = base(image)
        y = tf.keras.layers.Dense(img_features)(y)
        y = actfunc(y)
        
        x = tf.keras.layers.concatenate([x, y])
        
    if pre_batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)
    
    for j, neurons in enumerate(hidden_layers):
        
        if l2_regularization:
            x = tf.keras.layers.Dense(neurons, kernel_regularizer = tf.keras.regularizers.l2(regularization_constant))(x)
        else:
            x = tf.keras.layers.Dense(neurons)(x)
        
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization(x)
        
        x = actfunc(x)
        if j in drop_out_layers:
            x = tf.keras.layers.Dropout(drop_out_rate)(x)
    
    FVC_output = tf.keras.layers.Dense(1, name = 'FVC_output')(x)
    sigma_output = tf.keras.layers.Dense(1, name = 'sigma_output')(x)
    
    # TOFO How is this normalizing?
    if output_normalization:
        FVC_output = tf.math.scalar_mul(tf.constant(50, dtype='float32'), FVC_output)
        sigma_output = tf.math.scalar_mul(tf.constant(50, dtype='float32'), sigma_output)
        
        if not predict_slope:
            FVC_output = tf.math.scalar_mul(tf.constant(100, dtype='float32'), FVC_output)
            sigma_output = tf.math.scalar_mul(tf.constant(100, dtype='float32'), sigma_out)
    
    if predict_slope:
        WeekDiff = tf.keras.layers.Input(shape = (1), name = 'WeekDiff')
        InitFVC = tf.keras.layers.Input(shape = (1), name = 'InitFVC')
        inputs.extend([WeekDiff, InitFVC])
        FVC_output = tf.add(tf.keras.layers.multiply([FVC_output, WeekDiff]), InitFVC)
        sigma_output = tf.keras.layers.multiply([sigma_output, WeekDiff])
        
    outputs = tf.keras.layers.concatenate([FVC_output, sigma_output])
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    opt = tf.keras.optimizers.Adam()
    
    if optimal_sigma_loss:
        loss = optimal_sigma_loss_function
    else:
        loss=(lambda x,y:(laplace_log_likelihood(x, y) + absolute_delta_error(x, y) * loss_modification*SQRT2/70))
    
    model.compile(optimizer=opt, loss=loss,
                  metrics = [laplace_metric, sigma_cost, delta_over_sigma, absolute_delta_error])
    
    return model



build_model(b_model='b5')


def score(y_pred, y_true, sigma):
    sigma_clipped = max(sigma, 70)
    delta = min(abs(y_pred, y_true), 1000)
    metric = - 2**0.5 * delta / sigma_clipped - np.log(2**0.5 * sigma_clipped)
    # Should I return the mean of metric?
    return metric


# +
# def encode_feature(row):
#    vector = [(row.Age - 30) / 30]
# -

def get_img(path):
    ds = dcmread(train_path + 'DCM/' + path)
    # TODO there is some resizing and scaling in the original code here, understand why
    return ds


