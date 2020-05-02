# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:06:14 2019

@author: MatthiasQ
"""

# Import the packages needed
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
print(tf.VERSION)

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test,y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)

model = tf.keras.models.Sequential([ #must include the brackets for 3+ layers
        tf.keras.layers.Flatten(input_shape =(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation = 'softmax')
        ])

model.compile(optimizer = "adam",
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy']) #Need to include brackets around metrics

model.fit(x=x_train, y = y_train,
          epochs = 10,
          verbose = True)

model.evaluate(x = x_test, y = y_test,
               verbose = True)
