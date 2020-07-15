# the classifier derived from alternative training
import numpy as np
import tensorflow as tf
import cv2


class alternative_classifier():

    def think(self, local_image_rgb, local_aclass_settings):
        type_of_model = '_EfficientNetB2'
        alt_classifier_ = tf.keras.applications.EfficientNetB2(include_top=True, weights='imagenet',
                                                           input_tensor=None, input_shape=None,
                                                           pooling=None, classes=1000,
                                                           classifier_activation='softmax')
        local_image = cv2.resize(local_image_rgb, (260, 260))
        local_image = local_image.reshape(1, local_image.shape[0], local_image.shape[1], local_image.shape[2])
        pred = alt_classifier_.predict(local_image)
        local_weights_hidden_message = np.load(''.join([local_aclass_settings['clean_data_path'],
                                                        'weights_hidden_message']))
        local_weights_no_hidden_message = np.load(''.join([local_aclass_settings['clean_data_path'],
                                                        'weights_no_hidden_message']))
        pred_hidden = np.multiply(pred, local_weights_hidden_message)
        pred_no_hidden = np.multiply(pred, local_weights_no_hidden_message)
        pred_hidden = np.divide(pred_hidden, np.amax(pred_hidden)).sum()
        pred_no_hidden = np.divide(pred_no_hidden, np.amax(pred_no_hidden)).sum()
        pred = pred_hidden / ((pred_hidden + pred_no_hidden) if (pred_hidden + pred_no_hidden) != 0 else 1.)
        return pred
