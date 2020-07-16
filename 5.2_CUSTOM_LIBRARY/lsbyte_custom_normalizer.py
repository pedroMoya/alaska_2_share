# function that makes last pre-process step to images, preparing data for training the CNNs
# use vectorize and LSByte
import numpy as np
import itertools as it
import tensorflow as tf
import cv2


class lsbyte_custom_image_normalizer():

    def normalize(self, local_image_rgb):
        local_image_rgb = cv2.cvtColor(local_image_rgb, cv2.COLOR_RGB2YCR_CB)
        channel_y = local_image_rgb[:, :, 0: 1]
        local_image_rgb[:, :, 0:1] = channel_y
        channel_y = local_image_rgb[:, :, 0: 1].astype(np.int32)
        local_image_rgb[:, :, 2: 3] = np.bitwise_and(channel_y, 1)
        local_image_rgb[:, :, 1: 2] = np.bitwise_and(channel_y, -channel_y)
        local_image_rgb = local_image_rgb.astype(np.float32)
        local_max = np.amax(local_image_rgb, axis=(0, 1))
        local_min = np.amin(local_image_rgb, axis=(0, 1))
        local_denom_diff = np.add(local_max, -local_min)
        local_denom_diff[local_denom_diff == 0] = 1
        local_num_diff = np.add(local_image_rgb, -local_min)
        return np.divide(local_num_diff, 255 * local_denom_diff)
