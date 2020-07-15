# calculate the most probable and best prediction based in a historic classification journal
# submodule that think...
# o beyond the line, that makes things better than thinking
import os
import logging
from logging import handlers
import json
import itertools as it
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2

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


class core_alternative_training():

    def imagine(self, question, local_cat_settings):
        try:
            answer = question
            # first check if experience_memory exist
            if os.path.isfile(''.join([local_cat_settings['clean_data_path'], 'experience_memory.npy'])):
                a = 0
            else:
                print('creating first memory at born of data seeds')
                local_cat_nof_methods = local_cat_settings['nof_methods']
                local_cat_nof_groups = local_cat_settings['nof_K_fold_groups']
                local_cat_nof_classes = local_cat_nof_methods * local_cat_nof_groups
                local_cat_nof_weights = local_cat_nof_classes
                # method, group, classes, weights (a form for unknown_data_representation based in previous data)
                experience_memory = np.zeros(shape=(local_cat_nof_methods, local_cat_nof_groups, local_cat_nof_classes,
                                             local_cat_nof_weights), dtype=np.float32)
                # loading cleaned data
                # 300000_id_numbers, 1000_predictions, 1000_classes, 4_methods, 4_groups
                cleaned_data = np.load(''.join([local_cat_settings['clean_data_path'],
                                                'train_dataset_predictions_efficientnetb2.npy']),
                                       allow_pickle=True)
                cleaned_data = np.delete(cleaned_data, 0, 1)
                # kernel of core
                local_nof_images = cleaned_data.shape[0]

                for local_cat_method, local_cat_group in it.product(range(local_cat_nof_methods),
                                                                    range(local_cat_nof_groups)):
                    pred_classes_by_method_group = cleaned_data[(cleaned_data[:, 2] == local_cat_method) &
                                                                (cleaned_data[:, 3] == local_cat_group)]
                    pred_classes_by_method_group = pred_classes_by_method_group[:, 0: 2]
                    print(pred_classes_by_method_group.shape)
                    while True:
                        a = 0
        except Exception as e3:
            print('error in brain core of alternative training auxiliary submodule')
            print(e3)
            logger.error(str(e3), exc_info=True)
            return False
        return answer
