# metric for training CNN
import tensorflow as tf
from tensorflow.keras import metrics
from sklearn.metrics import roc_auc_score

class customized_metrics_auc_roc(metrics.Metric):
    @tf.function
    def call(self, y_true_local, y_pred_local):
        return tf.py_function(roc_auc_score, (y_true_local, y_pred_local), tf.double)
