# preparing data (cleaning raw data, aggregating and saving to file)

# importing python libraries and opening settings
import os
import sys
import shutil
import logging
import logging.handlers as handlers
import json
import datetime
import numpy as np
import pandas as pd
import itertools as it
import tensorflow as tf
from tensorflow.keras import preprocessing

# open local settings and change local_scrip_settings if metaheuristic equals True
with open('./settings.json') as local_json_file:
    local_script_settings = json.loads(local_json_file.read())
    local_json_file.close()

# import custom libraries
sys.path.insert(1, local_script_settings['custom_library_path'])
from quality_factor_detector import quality_factor

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
        # define filepaths
        raw_data_path = local_script_settings['raw_data_path']
        method_0_raw_data_path = ''.join([raw_data_path, local_script_settings['method_0_folder']])
        method_1_raw_data_path = ''.join([raw_data_path, local_script_settings['method_1_folder']])
        method_2_raw_data_path = ''.join([raw_data_path, local_script_settings['method_2_folder']])
        method_3_raw_data_path = ''.join([raw_data_path, local_script_settings['method_3_folder']])

        # extract files
        if not os.path.isfile(''.join([local_script_settings['raw_data_path'], 'images_localization.txt'])):
            images_method_0 = [''.join([method_0_raw_data_path, filename])
                               for filename in os.listdir(method_0_raw_data_path)]
            images_method_1 = [''.join([method_1_raw_data_path, filename])
                               for filename in os.listdir(method_1_raw_data_path)]
            images_method_2 = [''.join([method_2_raw_data_path, filename])
                               for filename in os.listdir(method_2_raw_data_path)]
            images_method_3 = [''.join([method_3_raw_data_path, filename])
                               for filename in os.listdir(method_3_raw_data_path)]
            images_loc = ','.join(images_method_0 + images_method_1 + images_method_2 + images_method_3)
            # save
            with open(''.join([local_script_settings['raw_data_path'], 'images_localization.txt']), 'w') as f:
                f.write(images_loc)
                f.close()
            images_loc = images_loc.split(',')
        else:
            with open(''.join([local_script_settings['raw_data_path'], 'images_localization.txt'])) as f:
                chain = f.read()
                images_loc = chain.split(',')
                f.close()
        nof_images = len(images_loc)
        print('total jpg images found:', nof_images)

        # open raw_data
        print('first pre-processing step: disaggregation')
        if local_script_settings['disaggregation_done'] == "False":
            for image_path in images_loc:
                # detecting the steganographic-method by folder
                train_data_path_template = local_script_settings['train_data_path']
                if 'Cover' in image_path:
                    train_data_path = ''.join([train_data_path_template, 'method_0_compression_'])
                elif 'JMiPOD' in image_path:
                    train_data_path = ''.join([train_data_path_template, 'method_1_compression_'])
                elif 'JUNIWARD' in image_path:
                    train_data_path = ''.join([train_data_path_template, 'method_2_compression_'])
                elif 'UERD' in image_path:
                    train_data_path = ''.join([train_data_path_template, 'method_3_compression_'])
                else:
                    print('steganographic-method not understood')
                    return False
                # detecting the compression or quality_factor
                quality_factor_instance = quality_factor()
                quality_factor_detected = quality_factor_instance.detect(image_path)
                filename = image_path.split('/')[-1]
                train_data_path_filename = ''.join([train_data_path, quality_factor_detected, '/', filename])
                # storing the file in the correct folder
                shutil.copyfile(image_path, train_data_path_filename)
            with open('./settings.json', 'w', encoding='utf-8') as local_wr_json_file:
                local_script_settings['disaggregation_done'] = "True"
                json.dump(local_script_settings, local_wr_json_file, ensure_ascii=False, indent=2)
                local_wr_json_file.close()
            print('data aggregation was done')
        elif local_script_settings['disaggregation_done'] == "True":
            print('data disaggregation was done previously')
        else:
            print('settings disaggregation not understood')
            return False

        # data general_mean based - scaling
        # this step is automatically done in train by ImageDataGenerator
        print('data scaling was correctly prepared')

        # data normalization based in moving window
        # this step is included as a pre-processing_function in ImageDataGenerator
        print('data normalization was prepared as a pre-processing_function')

        # save clean data source for subsequent training
        # np.save(''.join([local_script_settings['train_data_path'], 'x_train_source']),
        #         window_normalized_scaled_unit_sales)
        # np.savetxt(''.join([local_script_settings['clean_data_path'], 'x_train_source.csv']),
        #            window_normalized_scaled_unit_sales, fmt='%10.15f', delimiter=',', newline='\n')
        # print('cleaned data -and their metadata- saved to file')
        # logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
        #                      ' successful saved cleaned data and metadata']))
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
