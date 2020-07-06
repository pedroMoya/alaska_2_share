# build CNN model
import os
import sys
import logging
import logging.handlers as handlers
import json
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
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
logging.basicConfig(filename=log_path_filename, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
logHandler = handlers.RotatingFileHandler(log_path_filename, maxBytes=10485760, backupCount=5)
logger.addHandler(logHandler)

# load custom libraries
sys.path.insert(1, local_submodule_settings['custom_library_path'])
from customized_metrics import auc_roc
from model_analyzer import model_structure

# class definitions


class model_classifier_:

    def build(self, local_model_name, local_settings, local_hyperparameters):
        try:
            # keras session/random seed reset/fix
            kb.clear_session()
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
            epochs = int(local_hyperparameters['epochs'])
            batch_size = int(local_hyperparameters['batch_size'])
            workers = int(local_hyperparameters['workers'])
            if optimizer_function == 'adam':
                optimizer_function = optimizers.Adam(optimizer_learning_rate)
                optimizer_function = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer_function)
            elif optimizer_function == 'ftrl':
                optimizer_function = optimizers.Ftrl(optimizer_learning_rate)
            elif optimizer_function == 'sgd':
                optimizer_function = optimizers.SGD(optimizer_learning_rate)
            losses_list = []
            loss_1 = local_hyperparameters['loss_1']
            loss_2 = local_hyperparameters['loss_2']
            loss_3 = local_hyperparameters['loss_3']
            union_settings_losses = [loss_1, loss_2, loss_3]
            if 'categorical_crossentropy' in union_settings_losses:
                losses_list.append(losses.CategoricalCrossentropy())
            # if 'customized_loss_function' in union_settings_losses:
            #     losses_list.append(customized_loss())
            metrics_list = []
            metric1 = local_hyperparameters['metrics1']
            metric2 = local_hyperparameters['metrics2']
            union_settings_metrics = [metric1, metric2]
            if 'auc_roc' in union_settings_metrics:
                metrics_list.append(auc_roc())
            if 'categorical_accuracy' in union_settings_metrics:
                metrics_list.append(metrics.CategoricalAccuracy())
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
            if local_hyperparameters['regularizers_l1_l2_2'] == 'True':
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
            classifier_.add(layers.Conv2D(units_layer_1, kernel_size=(kernel_size_y_1, kernel_size_x_1),
                                          stride=(),
                                          input_shape=(input_shape_y, input_shape_x, nof_channels),
                                          activity_regularizer=activation_regularizer_1,
                                          activation=activation_1))
            classifier_.add(layers.MaxPooling2D(pool_size=(pool_size_y_1, pool_size_x_1)))
            classifier_.add(layers.Dropout(dropout_layer_1))
            # second layer
            classifier_.add(layers.Conv2D(units_layer_2,
                                          kernel_size=(kernel_size_y_2, kernel_size_x_2),
                                          activity_regularizer=activation_regularizer_2,
                                          activation=activation_2))
            classifier_.add(layers.MaxPooling2D(pool_size=(pool_size_y_2, pool_size_x_2)))
            classifier_.add(layers.Dropout(dropout_layer_2))
            # third layer
            classifier_.add(layers.Conv2D(units_layer_3,
                                          kernel_size=(kernel_size_y_3, kernel_size_x_3),
                                          activity_regularizer=activation_regularizer_3,
                                          activation=activation_3))
            classifier_.add(layers.MaxPooling2D(pool_size=(pool_size_y_3, pool_size_x_3)))
            classifier_.add(layers.Dropout(dropout_layer_3))
            # fourth layer
            classifier_.add(layers.Conv2D(units_layer_4,
                                          kernel_size=(kernel_size_y_4, kernel_size_x_4),
                                          activity_regularizer=activation_regularizer_4,
                                          activation=activation_4))
            classifier_.add(layers.MaxPooling2D(pool_size=(pool_size_y_4, pool_size_x_4)))
            classifier_.add(layers.Dropout(dropout_layer_4))
            # Flattening
            classifier_.add(layers.Flatten())
            # Full connection
            classifier_.add(layers.Dense(units_dense_layer_4, activation=activation_dense_layer_4),
                            activity_regularizer_dense_layer_4=activation_regularizer_dense_layer_4)
            classifier_.add(layers.Dropout(dropout_dense_layer_4))
            classifier_.add(layers.Dense(units_final_layer, activation=activation_final_layer))

            # Compile model
            classifier_.compile(optimizer=optimizer_function, loss=losses_list, metrics=metrics_list)

            # Summary and metrics
            classifier_.summary()
            print('metrics: ', classifier_.metrics)

            # save_model
            classifier_json = classifier_.to_json()
            with open(''.join([local_settings['models_path'], 'custom_classifier_.json']), 'w') \
                    as json_file:
                json_file.write(classifier_json)
                json_file.close()
            classifier_.save(''.join([local_settings['models_path'], 'custom_classifier_.h5']))
            print('model architecture saved')

            # output png and pdf with model
            model_architecture = model_structure()
            model_architecture_review = model_architecture.analize('custom_classifier_.h5', local_settings)

        except Exception as e:
            print('error in build or compile of customized model')
            print(e)
            classifier_ = None
            logger.error(str(e), exc_info=True)
        return classifier_
