# evaluation module
# open predictions done, ground truth data and applies metrics for evaluate models
# save results

# importing python libraries and opening settings
try:
    import os
    import sys
    import logging
    import logging.handlers as handlers
    import json
    import datetime
    import numpy as np
    import pandas as pd
    import itertools as it
    import tensorflow as tf
    from tensorflow.keras import backend as kb
    from tensorflow.keras import losses, models
    from tensorflow.keras.metrics import mean_absolute_percentage_error
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    with open('./settings.json') as local_json_file:
        local_script_settings = json.loads(local_json_file.read())
        local_json_file.close()
    sys.path.insert(1, local_script_settings['custom_library_path'])
    from model_analyzer import model_structure

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    tf.keras.backend.set_floatx('float32')
except Exception as ee1:
    print('Error importing libraries or opening settings (evaluation module)')
    print(ee1)

# log setup
current_script_name = os.path.basename(__file__).split('.')[0]
log_path_filename = ''.join([local_script_settings['log_path'], current_script_name, '.log'])
logging.basicConfig(filename=log_path_filename, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
logHandler = handlers.RotatingFileHandler(log_path_filename, maxBytes=10485760, backupCount=5)
logger.addHandler(logHandler)

# keras session/random seed reset/fix
kb.clear_session()
np.random.seed(1)
tf.random.set_seed(2)

# classes definitions


# functions definitions


def evaluate():
    try:
        print('\n~evaluation module~')

        # generate diagram of neural network model
        if local_script_settings['model_analyzer'] == 'on':
            analyzer = model_structure()
            model_name = '_acc_freq_in_block_nn_model_.h5'
            analysis_result = analyzer.analize(model_name, local_script_settings)
            if analysis_result:
                print('model_analysis successfully, json file saved')
            else:
                print('error at model_analysis submodule')


        # finalizing the last module
        print('model evaluation subprocess ended successfully')
    except Exception as e1:
        print('Error in evaluator module')
        print(e1)
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' evaluator module error']))
        logger.error(str(e1), exc_info=True)
        return False
    return True
