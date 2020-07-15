# submodule for alternative trainining using EfficientNetB2
import os
import sys
import datetime
import logging
import logging.handlers as handlers
import json
import itertools as it
import numpy as np
import pandas as pd
import cv2
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

# import custom libraries
sys.path.insert(1, local_submodule_settings['custom_library_path'])
from alternative_custom_normalizer import alternative_custom_image_normalizer


# class definitions


class alternative_training():

    def train_model(self, local_model, local_at_hyperparameters, local_at_settings):
        # this training creates a model for multiclass classification
        try:
            # open hyperparameters
            local_nof_methods = local_at_settings['nof_methods']
            local_nof_groups = local_at_settings['nof_K_fold_groups']

            # columns: id_number, method, quality_factor, group, filename, filepath
            image_metadata = pd.read_csv(''.join([local_at_settings['clean_data_path'],
                                                 'training_metadata_for_local.csv']))

            # open each image and obtain 16 more frequent prediction
            custom_image_normalizer_instance = alternative_custom_image_normalizer()
            # stored data: 300000_id_numbers, 1000_greater_predictions, 1000_classes, 4_methods, 4_groups
            train_dataset_predictions = []
            for local_method, local_group in it.product(range(local_nof_methods), range(local_nof_groups)):
                method_group = \
                    image_metadata[(image_metadata['method'] == local_method)
                                   & (image_metadata['group'] == local_group)]
                local_id_numbers = method_group.iloc[:, 0]
                local_filepaths = method_group.iloc[:, 5]
                for local_id_number, local_filepath in zip(local_id_numbers, local_filepaths):
                    image = cv2.imread(local_filepath)
                    image = cv2.resize(image, (260, 260))
                    image_normalized = custom_image_normalizer_instance.normalize(image) / 255.
                    image_normalized = image_normalized.reshape(1, image_normalized.shape[0], image_normalized.shape[1],
                                                                image_normalized.shape[2])
                    prediction = local_model.predict(image_normalized)
                    # clearly don't stored 16 greater but 1000 (all data), unknown why [:16] don't select 16
                    # but this is completely serendipity
                    greater_16_predictions_classes = (-prediction).argsort()[:16]
                    greater_16_predictions = prediction[0, greater_16_predictions_classes]
                    greater_16_predictions = greater_16_predictions.reshape(greater_16_predictions.shape[1])
                    greater_16_predictions_classes = \
                        greater_16_predictions_classes.reshape(greater_16_predictions_classes.shape[1])
                    train_dataset_predictions.append([local_id_number, greater_16_predictions,
                                                     greater_16_predictions_classes,
                                                      local_method, local_group])
            train_dataset_predictions = np.array(train_dataset_predictions)
            np.save(''.join([local_at_settings['clean_data_path'], 'train_dataset_predictions_efficientnetb2']),
                    train_dataset_predictions)

        except Exception as e2:
            print('error in alternative training auxiliary submodule')
            print(e2)
            logger.error(str(e2), exc_info=True)
            return False
        return True
