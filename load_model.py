import tensorflow as tf

loaded_model = tf.keras.models.load_model('crop_simple.keras')

loaded_model.summary()