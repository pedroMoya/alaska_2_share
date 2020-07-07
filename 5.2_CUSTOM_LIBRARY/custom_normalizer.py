# metric for training CNN
import numpy as np


class image_normalizer:

    def normalize(self, local_image):
        local_max = np.amax(local_image, axis=(0, 1))
        local_min = np.amin(local_image, axis=(0, 1))
        local_diff = np.add(local_max, -local_min)
        local_diff[local_diff == 0] = 1.
        local_image = np.divide(np.add(local_image, -local_min), local_diff)
        return local_image
