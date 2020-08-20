# build CNN model
import os
import sys
import datetime
import logging
import logging.handlers as handlers
import json
import numpy as np
import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
tf.keras.backend.set_floatx('float32')
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import losses, models
from tensorflow.keras import metrics
from tensorflow.keras import callbacks as cb
from tensorflow.keras import backend as kb
from sklearn.metrics import mean_squared_error
from tensorflow.keras.utils import plot_model, model_to_dot


# open local settings
with open('./settings.json') as local_json_file:
    local_submodule_settings = json.loads(local_json_file.read())
    local_json_file.close()

# log setup
current_script_name = os.path.basename(__file__).split('.')[0]
log_path_filename = ''.join([local_submodule_settings['log_path'], current_script_name, '.log'])
logging.basicConfig(filename=log_path_filename, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
logHandler = handlers.RotatingFileHandler(log_path_filename, maxBytes=10485760, backupCount=5)
logger.addHandler(logHandler)

# load custom libraries
sys.path.insert(1, local_submodule_settings['custom_library_path'])
from model_analyzer import model_structure


# function definitions


@tf.function
def customized_loss_t2(y_true_local, y_pred_local):
    # y_pred_local = tf.clip_by_value(y_pred_local, 0., 1.)
    y_true_idx = tf.clip_by_value(tf.math.argmax(y_true_local, axis=1), 0, 1)
    y_true = tf.one_hot(y_true_idx, depth=2)
    y_pred = tf.expand_dims(tf.reduce_sum(y_pred_local[:, 1: 4], axis=1), axis=1)
    y_pred = tf.concat([y_pred_local[:, 0: 1], y_pred], axis=1)
    cat_crossent = losses.CategoricalCrossentropy(label_smoothing=0.05, reduction=losses.Reduction.SUM)
    return cat_crossent(y_true, y_pred)


# class definitions


# class customized_loss_t2(losses.Loss):
#     @tf.function
#     def call(self, y_true_local, y_pred_local):
#         y_true_idx = tf.clip_by_value(tf.math.argmax(y_true_local, axis=1), 0, 1)
#         y_true = tf.one_hot(y_true_idx, depth=2)
#         y_pred = tf.concat([y_pred_local[:, 0: 1],
#                             tf.expand_dims(tf.reduce_sum(y_pred_local[:, 1: 4], axis=1), axis=1)], axis=1)
#         cat_crossent = losses.CategoricalCrossentropy(label_smoothing=0.05, reduction=losses.Reduction.SUM)
#         return cat_crossent(y_true, y_pred)


class customized_loss_t(losses.Loss):
    @tf.function
    def call(self, y_true_local, y_pred_local):
        return tf.reduce_max(tf.nn.softmax_cross_entropy_with_logits(labels=y_true_local, logits=y_pred_local))



class customized_loss(losses.Loss):
    @tf.function
    def call(self, local_true, local_pred):
        log_softmax_diff = tf.reduce_sum(tf.math.abs(tf.math.add(local_true, -local_pred)))
        return log_softmax_diff


class customized_metric_auc_roc(metrics.AUC):
    @tf.function
    def call(self, local_true, local_pred):
        y_true_idx = tf.clip_by_value(tf.math.argmax(local_true, axis=1), 0, 1)
        y_true = tf.one_hot(y_true_idx, depth=2)
        y_pred = tf.expand_dims(tf.reduce_sum(local_pred[:, 1: 4], axis=1), axis=1)
        y_pred = tf.concat([local_pred[:, 0: 1], y_pred], axis=1)
        self.update_state(y_true, y_pred)
        return self.result()


class customized_loss2(losses.Loss):
    @tf.function
    def call(self, local_true, local_pred):
        local_true = tf.convert_to_tensor(local_true, dtype=tf.float32)
        local_pred = tf.convert_to_tensor(local_pred, dtype=tf.float32)
        factor_difference = tf.reduce_mean(tf.abs(tf.add(local_pred, -local_true)))
        factor_true = tf.reduce_mean(tf.add(tf.convert_to_tensor(1., dtype=tf.float32), local_true))
        return tf.math.multiply_no_nan(factor_difference, factor_true)


class model_classifier_:

    def build_and_compile(self, local_model_name, local_settings, local_hyperparameters):
        try:
            # keras,tf session/random seed reset/fix
            # kb.clear_session()
            # tf.compat.v1.reset_default_graph()
            np.random.seed(11)
            tf.random.set_seed(2)

            # load hyperparameters
            units_layer_1 = local_hyperparameters['units_layer_1']
            units_layer_2 = local_hyperparameters['units_layer_2']
            units_layer_3 = local_hyperparameters['units_layer_3']
            units_layer_4 = local_hyperparameters['units_layer_4']
            units_dense_layer_4 = local_hyperparameters['units_dense_layer_4']
            units_final_layer = local_hyperparameters['units_final_layer']
            activation_1 = local_hyperparameters['activation_1']
            activation_2 = local_hyperparameters['activation_2']
            activation_3 = local_hyperparameters['activation_3']
            activation_4 = local_hyperparameters['activation_4']
            activation_dense_layer_4 = local_hyperparameters['activation_dense_layer_4']
            activation_final_layer = local_hyperparameters['activation_final_layer']
            dropout_layer_1 = local_hyperparameters['dropout_layer_1']
            dropout_layer_2 = local_hyperparameters['dropout_layer_2']
            dropout_layer_3 = local_hyperparameters['dropout_layer_3']
            dropout_layer_4 = local_hyperparameters['dropout_layer_4']
            dropout_dense_layer_4 = local_hyperparameters['dropout_dense_layer_4']
            input_shape_y = local_hyperparameters['input_shape_y']
            input_shape_x = local_hyperparameters['input_shape_x']
            nof_channels = local_hyperparameters['nof_channels']
            stride_y_1 = local_hyperparameters['stride_y_1']
            stride_x_1 = local_hyperparameters['stride_x_1']
            kernel_size_y_1 = local_hyperparameters['kernel_size_y_1']
            kernel_size_x_1 = local_hyperparameters['kernel_size_x_1']
            kernel_size_y_2 = local_hyperparameters['kernel_size_y_2']
            kernel_size_x_2 = local_hyperparameters['kernel_size_x_2']
            kernel_size_y_3 = local_hyperparameters['kernel_size_y_3']
            kernel_size_x_3 = local_hyperparameters['kernel_size_x_3']
            kernel_size_y_4 = local_hyperparameters['kernel_size_y_4']
            kernel_size_x_4 = local_hyperparameters['kernel_size_x_4']
            pool_size_y_1 = local_hyperparameters['pool_size_y_1']
            pool_size_x_1 = local_hyperparameters['pool_size_x_1']
            pool_size_y_2 = local_hyperparameters['pool_size_y_2']
            pool_size_x_2 = local_hyperparameters['pool_size_x_2']
            pool_size_y_3 = local_hyperparameters['pool_size_y_3']
            pool_size_x_3 = local_hyperparameters['pool_size_x_3']
            pool_size_y_4 = local_hyperparameters['pool_size_y_4']
            pool_size_x_4 = local_hyperparameters['pool_size_x_4']
            optimizer_function = local_hyperparameters['optimizer']
            optimizer_learning_rate = local_hyperparameters['learning_rate']
            epsilon_adam = local_hyperparameters['epsilon_adam']
            if optimizer_function == 'adam':
                optimizer_function = optimizers.Adam(learning_rate=optimizer_learning_rate, epsilon=epsilon_adam)
            elif optimizer_function == 'ftrl':
                optimizer_function = optimizers.Ftrl(optimizer_learning_rate)
            elif optimizer_function == 'sgd':
                optimizer_function = optimizers.SGD(optimizer_learning_rate)
            elif optimizer_function == 'rmsp':
                optimizer_function = optimizers.RMSprop(optimizer_learning_rate, epsilon=epsilon_adam)
            optimizer_function = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer_function)
            loss_1 = local_hyperparameters['loss_1']
            loss_2 = local_hyperparameters['loss_2']
            loss_3 = local_hyperparameters['loss_3']
            label_smoothing = local_hyperparameters['label_smoothing']
            losses_list = []
            union_settings_losses = [loss_1, loss_2, loss_3]
            if 'CategoricalCrossentropy' in union_settings_losses:
                losses_list.append(losses.CategoricalCrossentropy(label_smoothing=label_smoothing))
            if 'BinaryCrossentropy' in union_settings_losses:
                losses_list.append(losses.BinaryCrossentropy())
            if 'CategoricalHinge' in union_settings_losses:
                losses_list.append(losses.CategoricalHinge())
            if 'KLD' in union_settings_losses:
                losses_list.append(losses.KLDivergence())
            if 'customized_loss_function' in union_settings_losses:
                losses_list.append(customized_loss())
            if 'customized_loss_t2' in union_settings_losses:
                losses_list.append(customized_loss_t2)
            if "Huber" in union_settings_losses:
                losses_list.append(losses.Huber())
            metrics_list = []
            metric1 = local_hyperparameters['metrics1']
            metric2 = local_hyperparameters['metrics2']
            union_settings_metrics = [metric1, metric2]
            if 'auc_roc' in union_settings_metrics:
                metrics_list.append(metrics.AUC())
            if 'customized_metric_auc_roc' in union_settings_metrics:
                metrics_list.append(customized_metric_auc_roc())
            if 'CategoricalAccuracy' in union_settings_metrics:
                metrics_list.append(metrics.CategoricalAccuracy())
            if 'CategoricalHinge' in union_settings_metrics:
                metrics_list.append(metrics.CategoricalHinge())
            if 'BinaryAccuracy' in union_settings_metrics:
                metrics_list.append(metrics.BinaryAccuracy())
            if local_settings['use_efficientNetB2'] == 'False':
                type_of_model = '_custom'
                if local_hyperparameters['regularizers_l1_l2_1'] == 'True':
                    l1_1 = local_hyperparameters['l1_1']
                    l2_1 = local_hyperparameters['l2_1']
                    activation_regularizer_1 = regularizers.l1_l2(l1=l1_1, l2=l2_1)
                else:
                    activation_regularizer_1 = None
                if local_hyperparameters['regularizers_l1_l2_2'] == 'True':
                    l1_2 = local_hyperparameters['l1_2']
                    l2_2 = local_hyperparameters['l2_2']
                    activation_regularizer_2 = regularizers.l1_l2(l1=l1_2, l2=l2_2)
                else:
                    activation_regularizer_2 = None
                if local_hyperparameters['regularizers_l1_l2_3'] == 'True':
                    l1_3 = local_hyperparameters['l1_3']
                    l2_3 = local_hyperparameters['l2_3']
                    activation_regularizer_3 = regularizers.l1_l2(l1=l1_3, l2=l2_3)
                else:
                    activation_regularizer_3 = None
                if local_hyperparameters['regularizers_l1_l2_4'] == 'True':
                    l1_4 = local_hyperparameters['l1_4']
                    l2_4 = local_hyperparameters['l2_4']
                    activation_regularizer_4 = regularizers.l1_l2(l1=l1_4, l2=l2_4)
                else:
                    activation_regularizer_4 = None
                if local_hyperparameters['regularizers_l1_l2_dense_4'] == 'True':
                    l1_dense_4 = local_hyperparameters['l1_dense_4']
                    l2_dense_4 = local_hyperparameters['l2_dense_4']
                    activation_regularizer_dense_layer_4 = regularizers.l1_l2(l1=l1_dense_4, l2=l2_dense_4)
                else:
                    activation_regularizer_dense_layer_4 = None

                # building model
                classifier_ = tf.keras.models.Sequential()
                # first layer
                classifier_.add(layers.Input(shape=(input_shape_y, input_shape_x, nof_channels)))
                # classifier_.add(layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
                classifier_.add(layers.Conv2D(units_layer_1, kernel_size=(kernel_size_y_1, kernel_size_x_1),
                                              strides=(stride_y_1, stride_x_1),
                                              activity_regularizer=activation_regularizer_1,
                                              activation=activation_1,
                                              padding='same',
                                              kernel_initializer=tf.keras.initializers.VarianceScaling(
                                                  scale=2., mode='fan_out', distribution='truncated_normal')))
                classifier_.add(layers.BatchNormalization(axis=-1))
                classifier_.add(layers.Activation(tf.keras.activations.swish))
                classifier_.add(layers.GlobalAveragePooling2D())
                classifier_.add(layers.Dropout(dropout_layer_1))
                # LAYER 1.5
                classifier_.add(layers.Conv2D(units_layer_1, kernel_size=(kernel_size_y_1, kernel_size_x_1),
                                              input_shape=(input_shape_y, input_shape_x, nof_channels),
                                              strides=(stride_y_1, stride_x_1),
                                              activity_regularizer=activation_regularizer_1,
                                              activation=activation_1,
                                              padding='same',
                                              kernel_initializer=tf.keras.initializers.VarianceScaling(
                                                  scale=2., mode='fan_out', distribution='truncated_normal')))
                classifier_.add(layers.BatchNormalization(axis=-1))
                classifier_.add(layers.Activation(tf.keras.activations.swish))
                classifier_.add(layers.GlobalAveragePooling2D())
                classifier_.add(layers.Dropout(dropout_layer_1))
                # second layer
                classifier_.add(layers.Conv2D(units_layer_2, kernel_size=(kernel_size_y_2, kernel_size_x_2),
                                              activity_regularizer=activation_regularizer_2,
                                              activation=activation_2,
                                              padding='same',
                                              kernel_initializer=tf.keras.initializers.VarianceScaling(
                                                  scale=2., mode='fan_out', distribution='truncated_normal')))
                classifier_.add(layers.BatchNormalization(axis=-1))
                classifier_.add(layers.Activation(tf.keras.activations.swish))
                classifier_.add(layers.GlobalAveragePooling2D())
                classifier_.add(layers.Dropout(dropout_layer_2))
                # LAYER 2.5
                classifier_.add(layers.Conv2D(units_layer_2, kernel_size=(kernel_size_y_2, kernel_size_x_2),
                                              activity_regularizer=activation_regularizer_2,
                                              activation=activation_2,
                                              padding='same',
                                              kernel_initializer=tf.keras.initializers.VarianceScaling(
                                                  scale=2., mode='fan_out', distribution='truncated_normal')))
                classifier_.add(layers.BatchNormalization(axis=-1))
                classifier_.add(layers.Activation(tf.keras.activations.swish))
                classifier_.add(layers.GlobalAveragePooling2D())
                classifier_.add(layers.Dropout(dropout_layer_2))
                # third layer
                classifier_.add(layers.Conv2D(units_layer_3,
                                              kernel_size=(kernel_size_y_3, kernel_size_x_3),
                                              activity_regularizer=activation_regularizer_3,
                                              activation=activation_3,
                                              padding='same',
                                              kernel_initializer=tf.keras.initializers.VarianceScaling(
                                                  scale=2., mode='fan_out', distribution='truncated_normal')))
                classifier_.add(layers.BatchNormalization(axis=-1))
                classifier_.add(layers.Activation(tf.keras.activations.swish))
                classifier_.add(layers.GlobalAveragePooling2D())
                classifier_.add(layers.Dropout(dropout_layer_3))
                # LAYER 3.5
                classifier_.add(layers.Conv2D(units_layer_3,
                                              kernel_size=(kernel_size_y_3, kernel_size_x_3),
                                              activity_regularizer=activation_regularizer_3,
                                              activation=activation_3,
                                              padding='same',
                                              kernel_initializer=tf.keras.initializers.VarianceScaling(
                                                  scale=2., mode='fan_out', distribution='truncated_normal')))
                classifier_.add(layers.BatchNormalization(axis=-1))
                classifier_.add(layers.Activation(tf.keras.activations.swish))
                classifier_.add(layers.GlobalAveragePooling2D())
                classifier_.add(layers.Dropout(dropout_layer_3))
                # fourth layer
                classifier_.add(layers.Conv2D(units_layer_4,
                                              kernel_size=(kernel_size_y_4, kernel_size_x_4),
                                              activity_regularizer=activation_regularizer_4,
                                              activation=activation_4,
                                              padding='same',
                                              kernel_initializer=tf.keras.initializers.VarianceScaling(
                                                  scale=2., mode='fan_out', distribution='truncated_normal')))
                classifier_.add(layers.BatchNormalization(axis=-1))
                classifier_.add(layers.Activation(tf.keras.activations.swish))
                classifier_.add(layers.GlobalAveragePooling2D())
                classifier_.add(layers.Dropout(dropout_layer_4))
                # Full connection and final layer
                classifier_.add(layers.Dense(units=units_final_layer, activation=activation_final_layer))
                # Compile model
                classifier_.compile(optimizer=optimizer_function, loss=losses_list, metrics=metrics_list)

            elif local_settings['use_efficientNetB2'] == 'True':
                type_of_model = '_EfficientNetB2'
                # pretrained_weights = ''.join([local_settings['models_path'],
                #                               local_hyperparameters['weights_for_training_efficientnetb2']])
                classifier_pretrained = tf.keras.applications.EfficientNetB2(include_top=False, weights='imagenet',
                                                                             input_tensor=None,
                                                                             input_shape=(input_shape_y,
                                                                                          input_shape_x, 3),
                                                                             pooling=None,
                                                                             classifier_activation=None)
                # classifier_pretrained.save_weights(''.join([local_settings['models_path'],
                #                                             'pretrained_efficientnetb2_weights.h5']))
                #
                # classifier_receptor = tf.keras.applications.EfficientNetB2(include_top=False, weights=None,
                #                                                              input_tensor=None,
                #                                                              input_shape=(input_shape_y,
                #                                                                           input_shape_x, 1),
                #                                                              pooling=None,
                #                                                              classifier_activation=None)
                #
                # classifier_receptor.load_weights(''.join([local_settings['models_path'],
                #                                             'pretrained_efficientnetb2_weights.h5']), by_name=True)
                #
                # classifier_pretrained = classifier_receptor

                if local_settings['nof_classes'] == 2 or local_hyperparameters['use_bias_always'] == 'True':
                    # if two classes, log(pos/neg) = log(0.75/0.25) = 0.477121254719
                    bias_initializer = tf.keras.initializers.Constant(local_hyperparameters['bias_initializer'])
                else:
                    # assuming balanced classes...
                    bias_initializer = tf.keras.initializers.Constant(0)

                effnb2_model = models.Sequential(classifier_pretrained)
                effnb2_model.add(layers.GlobalAveragePooling2D())
                effnb2_model.add(layers.Dropout(dropout_dense_layer_4))
                # effnb2_model.add(layers.Dense(units=units_dense_layer_4, activation=activation_dense_layer_4,
                #                  kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.333333333,
                #                                                                           mode='fan_out',
                #                                                                           distribution='uniform'),
                #                               bias_initializer=bias_initializer))
                # effnb2_model.add(layers.Dropout(dropout_dense_layer_4))
                effnb2_model.add(layers.Dense(units_final_layer, activation=activation_final_layer,
                                 kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.333333333,
                                                                                          mode='fan_out',
                                                                                          distribution='uniform'),
                                              bias_initializer=bias_initializer))
                classifier_ = effnb2_model

                if local_settings['use_local_pretrained_weights_for_retraining'] != 'False':
                    classifier_.load_weights(''.join([local_settings['models_path'],
                                                      local_settings['use_local_pretrained_weights_for_retraining']]))
                    for layer in classifier_.layers[0].layers:
                        layer.trainable = True
                        # if 'excite' in layer.name:
                        #     layer.trainable = True
                        # if 'top_conv' in layer.name:
                        #     layer.trainable = True
                        # if 'project_conv' in layer.name:
                        #     layer.trainable = True

                classifier_.build(input_shape=(input_shape_y, input_shape_x, nof_channels))
                classifier_.compile(optimizer=optimizer_function, loss=losses_list, metrics=metrics_list)

            # Summary of model
            classifier_.summary()

            # save_model
            classifier_json = classifier_.to_json()
            with open(''.join([local_settings['models_path'], local_model_name, type_of_model,
                               '_classifier_.json']), 'w') \
                    as json_file:
                json_file.write(classifier_json)
                json_file.close()
            classifier_.save(''.join([local_settings['models_path'], local_model_name, type_of_model,
                                      '_classifier_.h5']))
            classifier_.save(''.join([local_settings['models_path'], local_model_name, type_of_model,
                                      '/']), save_format='tf')
            print('model architecture saved')

            # output png and pdf with model, additionally saves a json file model_name_analyzed.json
            if local_settings['model_analyzer'] == 'True':
                model_architecture = model_structure()
                model_architecture_review = model_architecture.analize(''.join([local_model_name, type_of_model,
                                                                                '_classifier_.h5']),
                                                                       local_settings, local_hyperparameters)
        except Exception as e:
            print('error in build or compile of customized model')
            print(e)
            classifier_ = None
            logger.error(str(e), exc_info=True)
        return classifier_
