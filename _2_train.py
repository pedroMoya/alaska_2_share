# open clean data, conform data structures
# training and saving models

# importing python libraries and opening settings
try:
    import os
    import sys
    import logging
    import logging.handlers as handlers
    import json
    import datetime
    import PIL
    import cv2
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    tf.keras.backend.set_floatx('float32')
    from tensorflow.keras import layers, models
    from tensorflow.keras import backend as kb
    from tensorflow.keras import preprocessing
    from tensorflow.keras import regularizers
    from tensorflow.keras import optimizers
    from tensorflow.keras import losses
    from tensorflow.keras import metrics
    from tensorflow.keras import callbacks as cb

    # open local settings
    with open('./settings.json') as local_json_file:
        local_settings = json.loads(local_json_file.read())
        local_json_file.close()

    # import custom libraries
    sys.path.insert(1, local_settings['custom_library_path'])
    from custom_model_builder import model_classifier_
    # from custom_normalizer import image_normalizer

except Exception as ee1:
    print('Error importing libraries or opening settings (train module)')
    print(ee1)

# log setup
current_script_name = os.path.basename(__file__).split('.')[0]
log_path_filename = ''.join([local_settings['log_path'], current_script_name, '.log'])
logging.basicConfig(filename=log_path_filename, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
logHandler = handlers.RotatingFileHandler(log_path_filename, maxBytes=10485760, backupCount=5)
logger.addHandler(logHandler)


# classes definitions


class customized_loss(losses.Loss):
    @tf.function
    def call(self, local_true, local_pred):
        return tf.math.abs(tf.math.add(tf.nn.log_softmax(local_true), -tf.nn.log_softmax(local_pred)))


class customized_loss3(losses.Loss):
    @tf.function
    def call(self, local_true, local_pred):
        return tf.math.add(1., -metrics.categorical_accuracy(local_true, local_pred))



class customized_loss2(losses.Loss):
    @tf.function
    def call(self, local_true, local_pred):
        local_true = tf.convert_to_tensor(local_true, dtype=tf.float32)
        local_pred = tf.convert_to_tensor(local_pred, dtype=tf.float32)
        factor_difference = tf.reduce_mean(tf.abs(tf.add(local_pred, -local_true)))
        factor_true = tf.reduce_mean(tf.add(tf.convert_to_tensor(1., dtype=tf.float32), local_true))
        return tf.math.multiply_no_nan(factor_difference, factor_true)


# functions definitions


def image_normalizer(local_image_rgb):
    # local_y_channel = cv2.cvtColor(local_image_rgb, cv2.COLOR_RGB2YCR_CB)
    # local_y_channel = local_y_channel[:, :, 0: 1]
    local_max = np.amax(local_image_rgb, axis=(0, 1))
    local_min = np.amin(local_image_rgb, axis=(0, 1))
    local_denom_diff = np.add(local_max, -local_min)
    local_denom_diff[local_denom_diff == 0] = 1
    local_num_diff = np.add(local_image_rgb, -local_min)
    return tf.math.divide_no_nan(local_num_diff, local_denom_diff)


def train():
    print('\n~train_model module~')

    # keras session/random seed reset/fix
    kb.clear_session()
    np.random.seed(11)
    tf.random.set_seed(2)
    # tf.compat.v1.disable_eager_execution()

    # load model hyperparameters
    try:
        with open('./settings.json') as local_r_json_file:
            local_script_settings = json.loads(local_r_json_file.read())
            local_r_json_file.close()
        if local_script_settings['metaheuristic_optimization'] == "True":
            print('changing settings control to metaheuristic optimization')
            with open(''.join(
                    [local_script_settings['metaheuristics_path'], 'organic_settings.json'])) as local_r_json_file:
                local_script_settings = json.loads(local_r_json_file.read())
                local_r_json_file.close()

        # opening hyperparameters
        with open(''.join([local_script_settings['hyperparameters_path'], 'model_hyperparameters.json'])) \
                as local_r_json_file:
            model_hyperparameters = json.loads(local_r_json_file.read())
            local_r_json_file.close()

        if local_script_settings['data_cleaning_done'] == 'False':
            print('data was not cleaning')
            print('first prepare_data module have to run')
            return False

    except Exception as e1:
        print('Error loading model hyperparameters (train module)')
        print(e1)
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' error at loading model hyperparameters']))
        logger.error(str(e1), exc_info=True)

    # register model hyperparameters settings in log
    logging.info("\nexecuting train module program..\ncurrent models hyperparameters settings:%s",
                 ''.join(['\n', str(model_hyperparameters).replace(',', '\n')]))
    print('-current models hyperparameters registered in log')

    # starting training
    try:
        print('\n~train_model module~')
        # check settings for previous training and then repeat or not this phase
        if local_script_settings['training_done'] == "True":
            print('training of neural_network previously done')
            if local_script_settings['repeat_training'] == "True":
                print('repeating training')
            else:
                print("settings indicates don't repeat training")
                return True
        else:
            print('model training start')

        # model training hyperparameters
        nof_groups = local_script_settings['nof_K_fold_groups']
        input_shape_y = model_hyperparameters['input_shape_y']
        input_shape_x = model_hyperparameters['input_shape_x']
        epochs = model_hyperparameters['epochs']
        batch_size = model_hyperparameters['batch_size']
        workers = model_hyperparameters['workers']
        validation_split = model_hyperparameters['validation_split']
        early_stopping_patience = model_hyperparameters['early_stopping_patience']
        reduce_lr_on_plateau_factor = model_hyperparameters['ReduceLROnPlateau_factor']
        reduce_lr_on_plateau_patience = model_hyperparameters['ReduceLROnPlateau_patience']
        reduce_lr_on_plateau_min_lr = model_hyperparameters['ReduceLROnPlateau_min_lr']
        validation_freq = model_hyperparameters['validation_freq']
        training_set_folder = model_hyperparameters['training_set_folder']

        # load raw_data and cleaned_data
        training_set_folder_group = ''.join([training_set_folder])
        train_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                               horizontal_flip=True,
                                                               vertical_flip=True,
                                                               preprocessing_function=image_normalizer,
                                                               validation_split=validation_split)
        train_generator = train_datagen.flow_from_directory(training_set_folder_group,
                                                            target_size=(input_shape_y, input_shape_x),
                                                            batch_size=batch_size,
                                                            class_mode='categorical',
                                                            color_mode='rgb',
                                                            shuffle=True,
                                                            subset='training')
        print('labels and indices')
        print(train_generator.class_indices)
        validation_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                    preprocessing_function=image_normalizer,
                                                                    validation_split=validation_split)
        validation_generator = validation_datagen.flow_from_directory(training_set_folder_group,
                                                                      target_size=(input_shape_y, input_shape_x),
                                                                      batch_size=batch_size,
                                                                      class_mode='categorical',
                                                                      color_mode='rgb',
                                                                      shuffle=True,
                                                                      subset='validation')

        # build, compile and save model
        model_name = model_hyperparameters['current_model_name']
        classifier_constructor = model_classifier_()
        classifier = classifier_constructor.build_and_compile(model_name, local_script_settings,
                                                              model_hyperparameters)

        # define callbacks, checkpoints namepaths
        model_weights = ''.join([local_settings['checkpoints_path'],
                                 'check_point_', model_name, "_loss_-{loss:.4f}-.hdf5"])
        callback1 = cb.EarlyStopping(monitor='loss', patience=early_stopping_patience)
        callback2 = [cb.ModelCheckpoint(model_weights, monitor='loss', verbose=1,
                                       save_best_only=True, mode='min')]
        callback3 = cb.ReduceLROnPlateau(monitor='loss', factor=reduce_lr_on_plateau_factor,
                                         patience=reduce_lr_on_plateau_patience,
                                         min_lr=reduce_lr_on_plateau_min_lr)
        callbacks = [callback1, callback2, callback3]

        # training model
        model_train_history = classifier.fit(x=train_generator, batch_size=batch_size, epochs=epochs,
                                             steps_per_epoch=train_generator.samples // batch_size,
                                             callbacks=callbacks, shuffle=True, workers=workers,
                                             validation_data=validation_generator,
                                             validation_freq=validation_freq,
                                             validation_steps=validation_generator.samples // batch_size)

        # save weights (model was saved previously at model build and compile in h5 and json formats)
        date = datetime.date.today()
        classifier.save_weights(''.join([local_script_settings['models_path'], model_name, '_', str(date),
                                         '_weights.h5']))

        # closing train module
        print('full training of each group module ended')
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' correct model training, correct saving of model and weights']))
        local_script_settings['training_done'] = "True"
        if local_script_settings['metaheuristic_optimization'] == "False":
            with open('./settings.json', 'w', encoding='utf-8') as local_wr_json_file:
                json.dump(local_script_settings, local_wr_json_file, ensure_ascii=False, indent=2)
                local_wr_json_file.close()
        elif local_script_settings['metaheuristic_optimization'] == "True":
            with open(''.join([local_script_settings['metaheuristics_path'],
                               'organic_settings.json']), 'w', encoding='utf-8') as local_wr_json_file:
                json.dump(local_script_settings, local_wr_json_file, ensure_ascii=False, indent=2)
                local_wr_json_file.close()
                # metaheuristic_train = tuning_metaheuristic()
                # metaheuristic_hyperparameters = metaheuristic_train.stochastic_brain(local_script_settings)
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' settings modified and saved']))
    except Exception as e1:
        print('Error training model')
        print(e1)
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' model training error']))
        logger.error(str(e1), exc_info=True)
        return False
    return True
