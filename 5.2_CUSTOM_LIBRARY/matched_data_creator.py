# match the images and creates evaluation folders (stored in 3_CLEAN_DATA_DIR)
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
from sklearn.model_selection import train_test_split

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


# classes definitions
class matched_data_builder:

    def create_dataset(self):
        try:
            # stack so we can split on the same quartet of images
            # (idea inspired from the work of Radu Enucă
            #  https://medium.com/datadriveninvestor/dual-input-cnn-with-keras-1e6d458cd979)
            x_train_comp = np.stack((x_train_method_0, x_train_method_1, x_train_method_2, x_train_method_3), axis=4)
            x_train, x_test, y_train, y_test = train_test_split(x_train_comp, labels, test_size=0.3, random_state=666)
            print('matched_images dataset creation submodule had finished')
        except Exception as e1:
            print('Error at matched_images dataset creation submodule')
            print(e1)
            logger.error(str(e1), exc_info=True)
            return False
        return True
