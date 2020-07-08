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
    from tensorflow.keras import preprocessing
    from sklearn.metrics import confusion_matrix, classification_report, log_loss, roc_auc_score
    from sklearn.preprocessing import minmax_scale

    with open('./settings.json') as local_json_file:
        local_script_settings = json.loads(local_json_file.read())
        local_json_file.close()
    sys.path.insert(1, local_script_settings['custom_library_path'])
    from create_evaluation_folders import select_evaluation_images

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    tf.keras.backend.set_floatx('float32')
except Exception as ee1:
    print('Error importing libraries or opening settings (evaluation module)')
    print(ee1)

# log setup
current_script_name = os.path.basename(__file__).split('.')[0]
log_path_filename = ''.join([local_script_settings['log_path'], current_script_name, '.log'])
logging.basicConfig(filename=log_path_filename, level=logging.INFO,
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
        # evaluate current model, store and display results
        print('\n~evaluation module~')

        # opening hyperparameters
        with open(''.join([local_script_settings['hyperparameters_path'], 'model_hyperparameters.json'])) \
                as local_r_json_file:
            model_hyperparameters = json.loads(local_r_json_file.read())
            local_r_json_file.close()

        # open model and weights
        current_model_name = model_hyperparameters['current_model_name']
        classifier = models.load_model(''.join([local_script_settings['models_path'], current_model_name]))
        weights_file_name = local_script_settings['weights_loaded_in evaluation']
        if weights_file_name == 'from_today':
            print('model loaded and by default the currents weights saved today will be loaded')
            date = datetime.date.today()
            classifier.loads_weights(''.join([local_script_settings['models_path'], current_model_name, '_', str(date),
                                              '_weights.h5']))
        else:
            print('model loaded and by setting this weights will be loaded:', weights_file_name)
            classifier.loads_weights(''.join([local_script_settings['models_path'], weights_file_name]))

        # define from scratch random evaluation folder
        model_evaluation_folder = ''.join([local_script_settings['models_evaluation_path'], 'images_for_evaluation/'])
        create_evaluation_folder = select_evaluation_images()
        create_evaluation_folder_review = create_evaluation_folder.select_images(local_script_settings)
        if create_evaluation_folder_review:
            print('new randomly selected images for evaluation generated successfully')
        else:
            print('error at generating folder with images for evaluation')
            logger.info('error at folder_for_evaluation generation')
            return False

        # Outcomes: accuracy - confusion_matrix
        try:
            batch_size = model_hyperparameters['batch_size']
            input_shape_y = model_hyperparameters['input_shape_y']
            input_shape_x = model_hyperparameters['input_shape_x']
            test_datagen = preprocessing.image.ImageDataGenerator(rescale=1. / 255)
            test_set = test_datagen.flow_from_directory(model_evaluation_folder,
                                                        shuffle=False,
                                                        target_size=(input_shape_y, input_shape_x),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
            y_predictions = np.array(classifier.predict(test_set))
            print('Confusion Matrix')
            print(confusion_matrix(test_set.classes, y_predictions))
            print('Classification Report')
            target_names = ['method_0_compression_0', 'method_0_compression_1', 'method_0_compression_2',
                            'method_1_compression_0', 'method_1_compression_1', 'method_1_compression_2',
                            'method_2_compression_0', 'method_2_compression_1', 'method_2_compression_2',
                            'method_3_compression_0', 'method_3_compression_1', 'method_3_compression_2']
            print(classification_report(test_set.classes, y_predictions, target_names=target_names))
            print(tf.math.confusion_matrix(labels=test_set.classes, predictions=y_predictions))
            print(classifier.metrics_names)
            print(classifier.evaluate(test_set, verbose=0))
            print("log_loss(sklearn.metrics):", log_loss(np.asarray(test_set.classes), y_predictions, eps=1e-15))
        except Exception as e:
            print('Error at making predictions or with model evaluation from sci-kit learn or tf confusion_matrix')
            print(e)
            logger.error(str(e), exc_info=True)

        # finalizing this module
        print('model evaluation subprocess ended successfully')
    except Exception as e1:
        print('Error in evaluator module')
        print(e1)
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' evaluator module error']))
        logger.error(str(e1), exc_info=True)
        return False
    return True
