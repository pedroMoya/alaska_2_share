# metric for training CNN
import tensorflow as tf
from sklearn.metrics import roc_auc_score

class customized_metrics:

    def auc_roc(self, y_true_local, y_pred_local):
        return tf.py_function(roc_auc_score, (y_true_local, y_pred_local), tf.double)
