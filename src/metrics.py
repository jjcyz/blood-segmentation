import tensorflow as tf
from tensorflow.keras import backend as K

class MetricsCollection:
  @staticmethod
  def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3,4])
    union = K.sum(y_true, axis=[1,2,3,4]) + K.sum(y_pred, axis=[1,2,3,4])
    return K.mean((2 * intersection + smooth) / (union + smooth))

