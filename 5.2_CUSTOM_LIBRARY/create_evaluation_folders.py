# Randomly select images and creates evaluation folders
import os
import sys
import shutil
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
tf.random.set_seed(2)


class select_evaluation_images:

    def select_images(self, local_settings):
        try:
            if local_settings['repeat_select_images_for_evaluation'] == "False":
                print('settings indicates maintain the evaluation dataset')
                return True
            # clean files previously selected


        except Exception as e1:
            print('Error at select_evaluation_images submodule')
            print(e1)
            logger.error(str(e1), exc_info=True)
            return False
        return True
