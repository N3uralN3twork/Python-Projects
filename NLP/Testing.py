



import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
print(tf.__version__)

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

encoder = info.features["text"].encoder

print(f"Vocabulary size: {encoder.vocab_size}")

BUFFER_SIZE = 10000
BATCH_SIZE = 4

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE)

test_dataset = test_dataset.padded_batch(BATCH_SIZE)




