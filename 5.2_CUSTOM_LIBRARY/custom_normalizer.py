# function that makes last pre-process step to images, preparing data for training the CNNs
import numpy as np
import tensorflow as tf
import cv2
from scipy.fftpack import dct


class custom_image_normalizer():

    def normalize(self, local_image_rgb):
        local_image_rgb = cv2.cvtColor(local_image_rgb, cv2.COLOR_RGB2YCR_CB)
        channel_y = local_image_rgb[:, :, 0: 1]
        local_image_rgb[:, :, 1: 2] = np.add(channel_y, 128).clip(0, 255)
        local_image_rgb[:, :, 2: 3] = np.add(channel_y, -128).clip(0)
        return local_image_rgb
