import tensorflow as tf

def conv_block(x, filters):
  x = tf.keras.layers.Conv3D(filters, 3, padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.ReLU()(x)
  return x
