import tensorflow as tf

# class MeanIOU(tf.keras.metrics.MeanIoU):
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         # y_true = tf.squeeze(y_true, -1)
#         y_pred = tf.nn.softmax(y_pred)
#         y_true = tf.argmax(y_true, axis=-1)
#
#         return super().update_state(y_true, y_pred, sample_weight)

# class Precision(tf.keras.metrics.Precision):
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         # y_true = tf.squeeze(y_true, -1)
#         y_pred = tf.nn.softmax(y_pred)
#         y_true = tf.argmax(y_true, axis=-1)
#
#         return super().update_state(y_true, y_pred)