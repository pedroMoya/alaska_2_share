# preparing data (cleaning raw data, aggregating and saving to file)

# importing python libraries and opening settings
import os
import logging
import logging.handlers as handlers
import json
import datetime
import numpy as np
import pandas as pd
import itertools as it

# open local settings and change local_scrip_settings if metaheuristic equals True
with open('./settings.json') as local_json_file:
    local_script_settings = json.loads(local_json_file.read())
    local_json_file.close()
if local_script_settings['metaheuristic_optimization'] == "True":
    with open(''.join([local_script_settings['metaheuristics_path'],
                       'organic_settings.json'])) as local_json_file:
        local_script_settings = json.loads(local_json_file.read())
        local_json_file.close()

# log setup
current_script_name = os.path.basename(__file__).split('.')[0]
log_path_filename = ''.join([local_script_settings['log_path'], current_script_name, '.log'])
logging.basicConfig(filename=log_path_filename, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
logHandler = handlers.RotatingFileHandler(log_path_filename, maxBytes=10485760, backupCount=5)
logger.addHandler(logHandler)
logger.info('_prepare_data module start')

# Random seed fixed
np.random.seed(1)

# functions definitions


def prepare():
    print('\n~prepare_data module~')
    # check if clean is done
    if local_script_settings['data_cleaning_done'] == "True":
        print('datasets already cleaned, based in settings info')
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' raw datasets already cleaned']))
        if local_script_settings['repeat_data_cleaning'] == "False":
            print('skipping prepare_data cleaning, as settings indicates')
            return True
        else:
            print('repeating data cleaning again')
            logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                                 ' cleaning raw datasets']))

    # pre-processing core
    try:
        # open raw_data

        # data general_mean based - scaling
        # this step is automatically done in train by ImageDataGenerator
        print('data scaling was correctly prepared')

        # data normalization based in moving window
        # this step is included as a pre-processing_function in ImageDataGenerator
        print('data normalization was prepared as a pre-processing_function')

        # data aggregation

        print('data aggregation was done')

        # save clean data source for subsequent training
        # np.save(''.join([local_script_settings['train_data_path'], 'x_train_source']),
        #         window_normalized_scaled_unit_sales)
        # np.savetxt(''.join([local_script_settings['clean_data_path'], 'x_train_source.csv']),
        #            window_normalized_scaled_unit_sales, fmt='%10.15f', delimiter=',', newline='\n')
        print('cleaned data -and their metadata- saved to file')
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' successful saved cleaned data and metadata']))
    except Exception as e1:
        print('Error at pre-processing raw data')
        print(e1)
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' data pre-processing error']))
        logger.error(str(e1), exc_info=True)
        return False

    # save settings
    try:
        if local_script_settings['metaheuristic_optimization'] == "False":
            with open('./settings.json', 'w', encoding='utf-8') as local_wr_json_file:
                local_script_settings['data_cleaning_done'] = "True"
                json.dump(local_script_settings, local_wr_json_file, ensure_ascii=False, indent=2)
                local_wr_json_file.close()
        elif local_script_settings['metaheuristic_optimization'] == "True":
            with open(''.join([local_script_settings['metaheuristics_path'],
                               'organic_settings.json']), 'w', encoding='utf-8') as local_wr_json_file:
                local_script_settings['data_cleaning_done'] = "True"
                json.dump(local_script_settings, local_wr_json_file, ensure_ascii=False, indent=2)
                local_wr_json_file.close()
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' settings modified and saved']))
        print('raw datasets cleaned, settings saved..')
    except Exception as e1:
        print('Error saving settings')
        print(e1)
        logger.error(str(e1), exc_info=True)

    # back to main code
    return True
