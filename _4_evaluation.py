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
    from tensorflow.keras import losses, models, metrics, optimizers
    from tensorflow.keras import preprocessing
    from sklearn.metrics import confusion_matrix, classification_report, log_loss, roc_auc_score
    from sklearn.preprocessing import minmax_scale

    with open('./settings.json') as local_json_file:
        local_script_settings = json.loads(local_json_file.read())
        local_json_file.close()
    sys.path.insert(1, local_script_settings['custom_library_path'])
    from create_evaluation_folders import select_evaluation_images
    from custom_normalizer import custom_image_normalizer
    from lsbit_custom_normalizer import lsbit_custom_image_normalizer
    from alternative_evaluator import alternative_evaluation
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


# classes definitions



class customized_loss(losses.Loss):
    @tf.function
    def call(self, local_true, local_pred):
        log_softmax_diff = tf.math.abs(tf.math.add(tf.nn.log_softmax(local_true), -tf.nn.log_softmax(local_pred)))
        return log_softmax_diff


class customized_metric_auc_roc(metrics.AUC):
    @tf.function
    def call(self, local_true, local_pred):
        y_true_idx = tf.clip_by_value(tf.math.argmax(local_true, axis=1), 0, 1)
        y_true = tf.one_hot(y_true_idx, depth=2)
        y_pred = tf.concat([local_pred[:, 0: 1],
                            tf.expand_dims(tf.reduce_sum(local_pred[:, 1: 4], axis=1), axis=1)], axis=1)
        self.update_state(y_true, y_pred)
        return self.result()


# functions definitions


@tf.function
def customized_loss_t2(y_true_local, y_pred_local):
    # y_pred_local = tf.clip_by_value(y_pred_local, 0., 1.)
    y_true_idx = tf.clip_by_value(tf.math.argmax(y_true_local, axis=1), 0, 1)
    y_true = tf.one_hot(y_true_idx, depth=2)
    y_pred = tf.concat([y_pred_local[:, 0: 1],
                        tf.expand_dims(tf.reduce_sum(y_pred_local[:, 1: 4], axis=1), axis=1)], axis=1)
    cat_crossent = losses.CategoricalCrossentropy(label_smoothing=0.05, reduction=losses.Reduction.SUM)
    return cat_crossent(y_true, y_pred)


def image_normalizer(image_rgb):
    custom_image_normalizer_instance = custom_image_normalizer()
    channel_y = custom_image_normalizer_instance.normalize(image_rgb)
    return channel_y


def evaluate():
    # keras,tf session/random seed reset/fix
    # kb.clear_session()
    # tf.compat.v1.reset_default_graph()
    np.random.seed(11)
    tf.random.set_seed(2)

    try:
        # evaluate current model, store and display results
        print('\n~evaluation module~')

        # opening hyperparameters
        with open(''.join([local_script_settings['hyperparameters_path'], 'model_hyperparameters.json'])) \
                as local_r_json_file:
            model_hyperparameters = json.loads(local_r_json_file.read())
            local_r_json_file.close()

        if local_script_settings['alternative_training'] == 'True':
            print('as configuration indicate, alternative classifier will be evaluated')
            alternative_evaluation_instance = alternative_evaluation()
            alternative_evaluation_instance_review = alternative_evaluation_instance.run(local_script_settings)
            if alternative_evaluation_instance_review:
                print('alternative evaluation was successful')
            else:
                print('error at alternative evaluation')
        else:
            group = 0
            # open model and weights
            current_model_name = ''.join([model_hyperparameters['current_model_name'], '_group_', str(group)])
            # model_filepath = ''.join([local_script_settings['models_path'], current_model_name,
            #                          '_custom_classifier_.json'])
            # with open(model_filepath) as local_file:
            #     model_json = local_file.read()
            #     local_file.close()
            # classifier = models.model_from_json(model_json)
            custom_obj = {'customized_loss_t2': customized_loss_t2,
                          'customized_metric_auc_roc': customized_metric_auc_roc}
            if local_script_settings['use_efficientNetB2'] == "False":
                type_of_model = '_custom'
                classifier = models.load_model(''.join([local_script_settings['models_path'], current_model_name,
                                                        type_of_model, '_classifier_.h5']), custom_objects=custom_obj)
            elif local_script_settings['use_efficientNetB2'] == "True":
                type_of_model = '_EfficientNetB2'
                classifier = models.load_model(''.join([local_script_settings['models_path'], current_model_name,
                                                        type_of_model, '_classifier_.h5']), custom_objects=custom_obj)
                date = datetime.date.today()
                # classifier = models.load_model(''.join([local_script_settings['models_path'], current_model_name,
                #                                         type_of_model, '_trained_/']))
            else:
                print('type of model not understood')
                return False

            classifier.summary()
            weights_file_name = local_script_settings['weights_loaded_in evaluation']
            if weights_file_name == 'from_today':
                print('model loaded and by default the currents weights saved today will be loaded')
                date = datetime.date.today()
                classifier.load_weights(''.join([local_script_settings['models_path'], current_model_name, '_',
                                                 type_of_model, '_', str(date), '_weights.h5']))
            else:
                print('model correctly loaded and by settings this weights will be loaded:', weights_file_name)
                classifier.load_weights(''.join([local_script_settings['models_path'], weights_file_name]))

            if local_script_settings['custom_model_evaluation_set_not_trainable'] == 'True':
                for layer in classifier.layers:
                    layer.trainable = False
                print('current model loaded and layers were set to not_trainable')
            # classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics='categorical_accuracy')

            # # define from scratch random evaluation folder
            # model_evaluation_folder = ''.join([local_script_settings['models_evaluation_path'], 'images_for_evaluation/'])
            # create_evaluation_folder = select_evaluation_images()
            # create_evaluation_folder_review = create_evaluation_folder.select_images(local_script_settings,
            #                                                                          model_evaluation_folder)
            # if create_evaluation_folder_review:
            #     print('new or maintained randomly selected images for evaluation subprocess ended successfully')
            # else:
            #     print('error at generating folder with images for evaluation')
            #     logger.info('error at folder_for_evaluation generation')
            #     return False

            # Outcomes: accuracy - confusion_matrix
            try:
                nof_group_for_evaluation = local_script_settings['nof_group_for_evaluation']
                nof_classes = local_script_settings['nof_classes']
                batch_size = model_hyperparameters['batch_size']
                input_shape_y = model_hyperparameters['input_shape_y']
                input_shape_x = model_hyperparameters['input_shape_x']
                test_datagen = \
                    preprocessing.image.ImageDataGenerator(rescale=None,
                                                           preprocessing_function=image_normalizer)
                column_names = ['id_number', 'id_class', 'group', 'filename', 'filepath']
                x_col = 'filepath'
                y_col = 'id_class'
                group = '0'
                metadata_train_images = \
                    pd.read_csv(''.join([local_script_settings['train_data_path'], 'training_metadata.csv']),
                                dtype=str, names=column_names, header=None)
                metadata_test_images = metadata_train_images.loc[metadata_train_images['group'] == group]
                test_set = test_datagen.flow_from_dataframe(dataframe=metadata_test_images,
                                                            directory=None,
                                                            x_col=x_col,
                                                            y_col=y_col,
                                                            shuffle=False,
                                                            target_size=(input_shape_y, input_shape_x),
                                                            batch_size=batch_size,
                                                            color_mode='rgb',
                                                            class_mode='categorical')
                y_predictions_raw = classifier.predict(test_set, workers=1)
                print(y_predictions_raw)
                y_predictions = y_predictions_raw.argmax(axis=1)

                print('Confusion Matrix for all categories')
                test_set_labels = test_set.labels
                print(test_set_labels)
                test_set_classes = [id_class for name, id_class in test_set.class_indices.items()]
                print(confusion_matrix(test_set_labels, y_predictions))
                print('Classification Report')
                print(classification_report(test_set_labels, y_predictions, labels=test_set_classes))
                print('\nevaluation of classifier by tf.keras.models.evaluate:')
                print('number of classes:', len(test_set_classes), '\n')
                confusion_matrix_tf = tf.math.confusion_matrix(labels=test_set_labels,
                                                               predictions=y_predictions)
                print(confusion_matrix_tf)
                print(classifier.evaluate(x=test_set, verbose=1, steps=1, return_dict=True))

                # calculating if stenographic method was used or not 0: no_hidden_message 1: hidden_message
                print('\nadjusting evaluation to ~no-hidden or hidden message in image~ binary classification')
                print('Confusion Matrix for binary classification')
                hidden_message_prob = np.argmax(y_predictions_raw[:,
                              nof_group_for_evaluation: nof_classes * nof_group_for_evaluation],
                              axis=1).clip(0, 1)
                # no_hidden_message_prob = np.round(np.add(1., -hidden_message_prob))
                print('prob hidden_message:\n', hidden_message_prob, '\n')
                labels = [label if label < 1 else 1 for label in test_set_labels]
                # binary_predictions = np.round(hidden_message_prob).astype('int')
                binary_predictions = np.sum(y_predictions_raw[:, 1:], axis=1)
                # binary_predictions = \
                #     np.divide(binary_predictions, np.add(binary_predictions, y_predictions_raw[:, 0]))
                print('\nground_truth:', labels)
                print('\nbinary_pred:', binary_predictions)
                print('\nconfusion_matrix_tf_binary')
                confusion_matrix_tf_binary = tf.math.confusion_matrix(labels=labels,
                                                                      predictions=np.round(binary_predictions))
                print(confusion_matrix_tf_binary, '\n')
                print('confusion_matrix_sklearn_binary')
                print(confusion_matrix(labels, np.round(binary_predictions)), '\n')
                target_names = ['no_hidden_message', 'hidden_message']
                print(classification_report(labels, np.round(binary_predictions), target_names=target_names))
                print('\n AUC_ROC score:', roc_auc_score(labels,  binary_predictions))
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
