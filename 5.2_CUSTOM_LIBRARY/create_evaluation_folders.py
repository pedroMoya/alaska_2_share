# Randomly select images and creates evaluation folders
import os
import sys
import shutil
import random
import itertools as it
import logging
import logging.handlers as handlers
import json
import numpy as np
import tensorflow as tf

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

# keras session/random seed reset/fix
np.random.seed(1)
random.seed(42)
tf.random.set_seed(2)

# functions definitions


def erase_previous_images(local_folder):
    for local_filename in os.listdir(local_folder):
        local_file_path = os.path.join(local_folder, local_filename)
        try:
            if os.path.isfile(local_file_path) or os.path.islink(local_file_path):
                os.unlink(local_file_path)
        except Exception as e:
            print('Failed at deleting file ', local_file_path)
            print(e)
            return False
    print('previous images deleted from evaluation folder ')
    return True

# classes definitions
class select_evaluation_images:

    def select_images(self, local_settings, local_model_evaluation_folder):
        try:
            local_nof_methods = local_settings['nof_methods']
            local_nof_quality_factors = local_settings['nof_quality_factors']
            nof_samples_by_group = local_settings['nof_evaluation_samples_by_group']
            if local_settings['repeat_select_images_for_evaluation'] == "False":
                print('settings indicates maintain the evaluation dataset')
                return True
            # clean files previously selected
            rng = np.random.default_rng()
            for method in range(local_nof_methods):
                # clean files previously selected
                local_dest_subfolder = ''.join([local_model_evaluation_folder, 'method_', str(method), '/'])
                print('erasing files in folder of method ', method)
                erase_previous_images(local_dest_subfolder)
                # select randomly
                local_src_subfolder = ''.join([local_settings['train_data_path'], 'method_', str(method), '/'])
                nof_samples = 0
                while nof_samples < nof_samples_by_group:
                    file_selected = random.choice([image_file for image_file in os.listdir(local_src_subfolder)
                                                   if os.path.isfile(os.path.join(local_src_subfolder, image_file))])
                    file_selected_path = ''.join([local_dest_subfolder, file_selected])
                    if not os.path.isfile(file_selected_path):
                        shutil.copyfile(''.join([local_src_subfolder, file_selected]), ''.join([local_dest_subfolder,
                                                                                               file_selected]))
                        nof_samples += 1
            print('select_evaluation_images submodule had finished')
        except Exception as e1:
            print('Error at select_evaluation_images submodule')
            print(e1)
            logger.error(str(e1), exc_info=True)
            return False
        return True
