import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
keras.utils.plot_model(model, show_shapes=True)\


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


bertLayer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1', trainable=True)

