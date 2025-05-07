import tensorflow as tf

def conv_block(x, filters):
  # Double Convolution
  x = tf.keras.layers.Conv3D(filters, 3, padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.ReLU()(x)

  x = tf.keras.layers.Conv3D(filter, 3, padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.ReLU()(x)

  return x

def create_memory_efficient_unet(input_shape):
  inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32)

  # Encoder
  conv1 = conv_block(inputs, 32)
  pool1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)

  conv2 = conv_block(pool1, 64)
  pool2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)

  conv3 = conv_block(pool2, 128)

  # Decoder
  up2 = tf.keras.layers.Conv3DTranspose(64, 2, strides=(2, 2, 2), padding='same')(conv3)
  up2 = tf.keras.layers.concatenate([conv2, up2])
  conv4 = conv_block(up2, 64)

  up1 = tf.keras.layers.Conv3DTranspose(32, 2, strides=(2, 2, 2), padding='same')(conv4)
  up1 = tf.keras.layers.concatenate([conv1, up1])
  conv5 = conv_block(up1, 32)

  outputs = tf.keras.layers.Conv3D(1, 1, activation='sigmoid')(conv5)

  return tf.keras.Model(input=[inputs], outputs=[outputs])





