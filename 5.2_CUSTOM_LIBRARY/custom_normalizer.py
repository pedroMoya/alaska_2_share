# function that makes last pre-process step to images, preparing data for training the CNNs
import numpy as np
import tensorflow as tf
import cv2


class custom_image_normalizer():

    def normalize(self, local_image_rgb):
        local_y_channel = cv2.cvtColor(local_image_rgb, cv2.COLOR_GRAY2RGB)
        local_y_channel = cv2.cvtColor(local_y_channel, cv2.COLOR_RGB2YCR_CB)
        local_y_channel = local_y_channel[:, :, 0: 1]
        local_max = np.amax(local_y_channel, axis=(0, 1))
        local_min = np.amin(local_y_channel, axis=(0, 1))
        local_denom_diff = np.add(local_max, -local_min)
        local_denom_diff[local_denom_diff == 0] = 1
        local_num_diff = np.add(local_y_channel, -local_min)
        return tf.math.divide_no_nan(local_num_diff, local_denom_diff)
