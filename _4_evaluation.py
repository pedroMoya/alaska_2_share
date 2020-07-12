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
    import cv2
    import tensorflow as tf
    from tensorflow.keras import backend as kb
    from tensorflow.keras import losses, models, metrics
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
np.random.seed(11)
tf.random.set_seed(2)

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
        custom_obj = {'customized_loss': customized_loss}
        classifier = models.load_model(''.join([local_script_settings['models_path'], current_model_name,
                                                '_custom_classifier_.h5']), custom_objects=custom_obj)
        classifier.summary()
        weights_file_name = local_script_settings['weights_loaded_in evaluation']
        if weights_file_name == 'from_today':
            print('model loaded and by default the currents weights saved today will be loaded')
            date = datetime.date.today()
            classifier.load_weights(''.join([local_script_settings['models_path'], current_model_name, '_', str(date),
                                             '_weights.h5']))
        else:
            print('model correctly loaded and by setting this weights will be loaded:', weights_file_name)
            classifier.load_weights(''.join([local_script_settings['models_path'], weights_file_name]))
        if local_script_settings['custom_model_evaluation_set_not_trainable'] == 'True':
            for layer in classifier.layers:
                layer.trainable = False
            print('current model loaded and layers were set to not_trainable')

        # define from scratch random evaluation folder
        model_evaluation_folder = ''.join([local_script_settings['models_evaluation_path'], 'images_for_evaluation/'])
        create_evaluation_folder = select_evaluation_images()
        create_evaluation_folder_review = create_evaluation_folder.select_images(local_script_settings,
                                                                                 model_evaluation_folder)
        if create_evaluation_folder_review:
            print('new or maintained randomly selected images for evaluation subprocess ended successfully')
        else:
            print('error at generating folder with images for evaluation')
            logger.info('error at folder_for_evaluation generation')
            return False

        # Outcomes: accuracy - confusion_matrix
        try:
            nof_methods = local_script_settings['nof_methods']
            nof_K_fold_groups = local_script_settings['nof_K_fold_groups']
            nof_evaluation_samples_by_group = local_script_settings['nof_evaluation_samples_by_group']
            batch_size = model_hyperparameters['batch_size']
            input_shape_y = model_hyperparameters['input_shape_y']
            input_shape_x = model_hyperparameters['input_shape_x']
            test_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                  preprocessing_function=image_normalizer)
            test_set = test_datagen.flow_from_directory(model_evaluation_folder,
                                                        shuffle=False,
                                                        target_size=(input_shape_y, input_shape_x),
                                                        batch_size=batch_size,
                                                        color_mode='rgb',
                                                        class_mode='categorical')
            y_predictions_raw = classifier.predict(test_set)
            y_predictions = y_predictions_raw.argmax(axis=1)
            print('Confusion Matrix for all categories')
            print(confusion_matrix(test_set.classes, y_predictions))
            print('Classification Report')
            target_names = ['cat',
                            'dog']
            print(classification_report(test_set.classes, y_predictions, target_names=target_names))
            print('\nevaluation of classifier by tf.keras.models.evaluate:')
            print(classifier.evaluate(x=test_set, verbose=0, return_dict=True))
            print("\nlog_loss(sklearn.metrics):", log_loss(np.asarray(test_set.classes),
                                                         y_predictions_raw, eps=1e-15))
            print('number of classes:', test_set.num_classes, '\n')
            confusion_matrix_tf = tf.math.confusion_matrix(labels=test_set.classes,
                                                           predictions=y_predictions)
            print(confusion_matrix_tf)

            # calculating if stenographic method was used or not 0: no_hidden_message 1: hidden_message
            print('\nadjusting evaluation to ~no-hidden or hidden message in image~ binary classification')
            print('Confusion Matrix for binary classification')
            hidden_message_prob = np.sum(y_predictions_raw[:, nof_K_fold_groups: nof_methods * nof_K_fold_groups],
                                         axis=1)
            # no_hidden_message_prob = np.round(np.add(1., -hidden_message_prob))
            print('prob hidden_message:\n', hidden_message_prob, '\n')
            labels = np.zeros(shape=hidden_message_prob.shape, dtype=np.dtype('int32'))
            labels[nof_evaluation_samples_by_group * nof_K_fold_groups:] = 1
            binary_predictions = np.round(hidden_message_prob).astype('int')
            print('\nground_truth:', labels)
            print('\nconfusion_matrix_tf_binary')
            confusion_matrix_tf_binary = tf.math.confusion_matrix(labels=labels,
                                                                  predictions=binary_predictions)
            print(confusion_matrix_tf_binary, '\n')
            print('confusion_matrix_sklearn_binary')
            print(confusion_matrix(labels, binary_predictions), '\n')
            target_names = ['no_hidden_message', 'hidden_message']
            print(classification_report(labels, binary_predictions, target_names=target_names))
            print('\n AUC_ROC score:', roc_auc_score(labels,  y_predictions_raw[:, 1]))
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
