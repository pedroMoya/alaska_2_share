import numpy as np
import pandas as pd
import cv2
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as kb
from tensorflow.keras import preprocessing
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import callbacks as cb


class custom_image_normalizer():

    def normalize(self, local_image_rgb):
        local_y_channel = cv2.cvtColor(local_image_rgb, cv2.COLOR_GRAY2RGB)
        local_y_channel = cv2.cvtColor(local_y_channel, cv2.COLOR_RGB2YCR_CB)
        local_y_channel = local_y_channel[:, :, 0: 1]
        local_max = np.amax(local_y_channel, axis=(0, 1))
        local_min = np.amin(local_y_channel, axis=(0, 1))
        local_denom_diff = np.add(local_max, -local_min)
        local_denom_diff[local_denom_diff == 0] = 1
        local_num_diff = np.add(local_y_channel, -local_min)
        return tf.math.divide_no_nan(local_num_diff, local_denom_diff)


class customized_loss(losses.Loss):
    @tf.function
    def call(self, local_true, local_pred):
        return tf.math.abs(tf.math.add(tf.nn.log_softmax(local_true), -tf.nn.log_softmax(local_pred)))


with open('../input/metadata-groups-folds-for-images/model_hyperparameters.json') as json_file:
    model_hyperparameters = json.load(json_file)
    json_file.close()

kb.clear_session()
np.random.seed(11)
tf.random.set_seed(2)
type_of_model = '_custom'
# load hyperparameters
units_layer_1 = model_hyperparameters['units_layer_1']
units_layer_2 = model_hyperparameters['units_layer_2']
units_layer_3 = model_hyperparameters['units_layer_3']
units_layer_4 = model_hyperparameters['units_layer_4']
units_dense_layer_4 = model_hyperparameters['units_dense_layer_4']
units_final_layer = model_hyperparameters['units_final_layer']
activation_1 = model_hyperparameters['activation_1']
activation_2 = model_hyperparameters['activation_2']
activation_3 = model_hyperparameters['activation_3']
activation_4 = model_hyperparameters['activation_4']
activation_dense_layer_4 = model_hyperparameters['activation_dense_layer_4']
activation_final_layer = model_hyperparameters['activation_final_layer']
dropout_layer_1 = model_hyperparameters['dropout_layer_1']
dropout_layer_2 = model_hyperparameters['dropout_layer_2']
dropout_layer_3 = model_hyperparameters['dropout_layer_3']
dropout_layer_4 = model_hyperparameters['dropout_layer_4']
dropout_dense_layer_4 = model_hyperparameters['dropout_dense_layer_4']
input_shape_y = model_hyperparameters['input_shape_y']
input_shape_x = model_hyperparameters['input_shape_x']
nof_channels = model_hyperparameters['nof_channels']
stride_y_1 = model_hyperparameters['stride_y_1']
stride_x_1 = model_hyperparameters['stride_x_1']
kernel_size_y_1 = model_hyperparameters['kernel_size_y_1']
kernel_size_x_1 = model_hyperparameters['kernel_size_x_1']
kernel_size_y_2 = model_hyperparameters['kernel_size_y_2']
kernel_size_x_2 = model_hyperparameters['kernel_size_x_2']
kernel_size_y_3 = model_hyperparameters['kernel_size_y_3']
kernel_size_x_3 = model_hyperparameters['kernel_size_x_3']
kernel_size_y_4 = model_hyperparameters['kernel_size_y_4']
kernel_size_x_4 = model_hyperparameters['kernel_size_x_4']
pool_size_y_1 = model_hyperparameters['pool_size_y_1']
pool_size_x_1 = model_hyperparameters['pool_size_x_1']
pool_size_y_2 = model_hyperparameters['pool_size_y_2']
pool_size_x_2 = model_hyperparameters['pool_size_x_2']
pool_size_y_3 = model_hyperparameters['pool_size_y_3']
pool_size_x_3 = model_hyperparameters['pool_size_x_3']
pool_size_y_4 = model_hyperparameters['pool_size_y_4']
pool_size_x_4 = model_hyperparameters['pool_size_x_4']
optimizer_function = model_hyperparameters['optimizer']
optimizer_learning_rate = model_hyperparameters['learning_rate']
if optimizer_function == 'adam':
    optimizer_function = optimizers.Adam(optimizer_learning_rate)
    optimizer_function = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer_function)
elif optimizer_function == 'ftrl':
    optimizer_function = optimizers.Ftrl(optimizer_learning_rate)
elif optimizer_function == 'sgd':
    optimizer_function = optimizers.SGD(optimizer_learning_rate)
losses_list = []
loss_1 = model_hyperparameters['loss_1']
loss_2 = model_hyperparameters['loss_2']
loss_3 = model_hyperparameters['loss_3']
union_settings_losses = [loss_1, loss_2, loss_3]
if 'CategoricalCrossentropy' in union_settings_losses:
    losses_list.append(losses.CategoricalCrossentropy())
if 'CategoricalHinge' in union_settings_losses:
    losses_list.append(losses.CategoricalHinge())
if 'LogCosh' in union_settings_losses:
    losses_list.append(losses.LogCosh)
if 'customized_loss_function' in union_settings_losses:
    losses_list.append(customized_loss())
metrics_list = []
metric1 = model_hyperparameters['metrics1']
metric2 = model_hyperparameters['metrics2']
union_settings_metrics = [metric1, metric2]
if 'auc_roc' in union_settings_metrics:
    metrics_list.append(metrics.AUC())
if 'CategoricalAccuracy' in union_settings_metrics:
    metrics_list.append(metrics.CategoricalAccuracy())
if 'CategoricalHinge' in union_settings_metrics:
    metrics_list.append(metrics.CategoricalHinge())
if 'BinaryAccuracy' in union_settings_metrics:
    metrics_list.append(metrics.BinaryAccuracy())
if model_hyperparameters['regularizers_l1_l2_1'] == 'True':
    l1_1 = model_hyperparameters['l1_1']
    l2_1 = model_hyperparameters['l2_1']
    activation_regularizer_1 = regularizers.l1_l2(l1=l1_1, l2=l2_1)
else:
    activation_regularizer_1 = None
if model_hyperparameters['regularizers_l1_l2_2'] == 'True':
    l1_2 = model_hyperparameters['l1_2']
    l2_2 = model_hyperparameters['l2_2']
    activation_regularizer_2 = regularizers.l1_l2(l1=l1_2, l2=l2_2)
else:
    activation_regularizer_2 = None
if model_hyperparameters['regularizers_l1_l2_3'] == 'True':
    l1_3 = model_hyperparameters['l1_3']
    l2_3 = model_hyperparameters['l2_3']
    activation_regularizer_3 = regularizers.l1_l2(l1=l1_3, l2=l2_3)
else:
    activation_regularizer_3 = None
if model_hyperparameters['regularizers_l1_l2_4'] == 'True':
    l1_4 = model_hyperparameters['l1_4']
    l2_4 = model_hyperparameters['l2_4']
    activation_regularizer_4 = regularizers.l1_l2(l1=l1_4, l2=l2_4)
else:
    activation_regularizer_4 = None
if model_hyperparameters['regularizers_l1_l2_dense_4'] == 'True':
    l1_dense_4 = model_hyperparameters['l1_dense_4']
    l2_dense_4 = model_hyperparameters['l2_dense_4']
    activation_regularizer_dense_layer_4 = regularizers.l1_l2(l1=l1_dense_4, l2=l2_dense_4)
else:
    activation_regularizer_dense_layer_4 = None

classifier_ = tf.keras.models.Sequential()
# first layer
classifier_.add(layers.Conv2D(units_layer_1, kernel_size=(kernel_size_y_1, kernel_size_x_1),
                              input_shape=(input_shape_y, input_shape_x, nof_channels),
                              strides=(stride_y_1, stride_x_1),
                              activity_regularizer=activation_regularizer_1,
                              activation=activation_1,
                              kernel_initializer=tf.keras.initializers.VarianceScaling(
                                  scale=2., mode='fan_out', distribution='truncated_normal')))
classifier_.add(layers.BatchNormalization(axis=-1))
classifier_.add(layers.MaxPooling2D(pool_size=(pool_size_y_1, pool_size_x_1)))
classifier_.add(layers.Dropout(dropout_layer_1))
# second layer
classifier_.add(layers.Conv2D(units_layer_2, kernel_size=(kernel_size_y_2, kernel_size_x_2),
                              activity_regularizer=activation_regularizer_2,
                              activation=activation_2,
                              kernel_initializer=tf.keras.initializers.VarianceScaling(
                                  scale=2., mode='fan_out', distribution='truncated_normal')))
classifier_.add(layers.BatchNormalization(axis=-1))
classifier_.add(layers.MaxPooling2D(pool_size=(pool_size_y_2, pool_size_x_2)))
classifier_.add(layers.Dropout(dropout_layer_2))
# third layer
classifier_.add(layers.Conv2D(units_layer_3,
                              kernel_size=(kernel_size_y_3, kernel_size_x_3),
                              activity_regularizer=activation_regularizer_3,
                              activation=activation_3,
                              kernel_initializer=tf.keras.initializers.VarianceScaling(
                                  scale=2., mode='fan_out', distribution='truncated_normal')))
classifier_.add(layers.BatchNormalization(axis=-1))
classifier_.add(layers.MaxPooling2D(pool_size=(pool_size_y_3, pool_size_x_3)))
classifier_.add(layers.Dropout(dropout_layer_3))
# fourth layer
classifier_.add(layers.Conv2D(units_layer_4,
                              kernel_size=(kernel_size_y_4, kernel_size_x_4),
                              activity_regularizer=activation_regularizer_4,
                              activation=activation_4,
                              kernel_initializer=tf.keras.initializers.VarianceScaling(
                                  scale=2., mode='fan_out', distribution='truncated_normal')))
classifier_.add(layers.BatchNormalization(axis=-1))
classifier_.add(layers.Activation(tf.keras.activations.swish))
classifier_.add(layers.GlobalAveragePooling2D())
# classifier_.add(layers.MaxPooling2D(pool_size=(pool_size_y_4, pool_size_x_4)))
classifier_.add(layers.Dropout(dropout_layer_4))
# Flattening
classifier_.add(layers.Flatten())
# Full connection and final layer
classifier_.add(layers.Dense(units_dense_layer_4, activation=activation_dense_layer_4,
                             activity_regularizer=activation_regularizer_dense_layer_4,
                             kernel_initializer=tf.keras.initializers.VarianceScaling(
                                 scale=0.333333333, mode='fan_out', distribution='uniform')))
classifier_.add(layers.Dropout(dropout_dense_layer_4))
classifier_.add(layers.Dense(units_final_layer, activation=activation_final_layer,
                             kernel_initializer=tf.keras.initializers.VarianceScaling(
                                 scale=0.333333333, mode='fan_out', distribution='uniform')))

classifier_.compile(optimizer=optimizer_function, loss=losses_list, metrics=metrics_list)
classifier_.summary()

classifier_json = classifier_.to_json()
model_name = 'kaggle_notebook_script_alaska2_model'
with open(''.join([model_name, str(type_of_model), '_classifier_.json']), 'w') \
        as json_file:
    json_file.write(classifier_json)
    json_file.close()
classifier_.save(''.join([model_name, str(type_of_model), '_classifier_.h5']))
