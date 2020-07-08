# metric for training CNN
import numpy as np


class image_normalizer:

    def call(self, local_image):
        local_max = np.amax(local_image, axis=(0, 1))
        local_min = np.amin(local_image, axis=(0, 1))
        local_denom_diff = np.add(local_max, -local_min)
        local_denom_diff[local_denom_diff == 0] = 1.
        local_min[local_denom_diff == 1] = 0.
        local_num_diff = np.add(local_image, -local_min)
        local_image = np.divide(local_num_diff, local_denom_diff)
        return local_image
