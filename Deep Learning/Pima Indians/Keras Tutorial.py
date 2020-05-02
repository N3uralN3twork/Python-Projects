# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:39:09 2019

@author: MatthiasQ

Updated on 31st October to use Tensorflow 2.0
"""


Keras Tutorial

import os
abspath = os.path.abspath('C:/Users/MatthiasQ.MATTQ/Desktop/Python Projects/KerasTutorials/Pima Indians')
os.chdir(abspath)

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.losses
import tensorflow.keras.metrics as metrics

dataset = np.loadtxt("pima.csv", delimiter=",")
dataset.shape

data = dataset[:,0:8]
labels = dataset[:,8]


#Define the keras model
#A sequential model adds layers one at a time
'relu' - 'rectified linear unit activation function'

#Defining the model
model = keras.Sequential()

#We have 8 input variables so the input dimension is 8
#The first hidden layer has 12 nodes and uses the relu function
model.add(Dense(12, input_dim=8, activation='relu'))

#The second hidden layer has 8 nodes and still uses the relu function
model.add(Dense(8, activation='relu'))

#The output layer has one node (1 variable) and uses the sigmoid function
#The sigmoid function is either a 0 or a 1, so it works for binary classifiers
model.add(Dense(1, activation='sigmoid'))


"Compiling the model"
#We must specify a loss function to evaluate our weights and an optimizer to
#search through different weights and finally some metrics

#Loss = cross_entropy
#Since we have a binary label, our classifier will be "binary_crossentropy"
#We will also be using the adam optimizer

model.compile(optimizer = "Adam",
              loss = keras.losses.binary_crossentropy,
              metrics = ['accuracy', metrics.mape])


"Fitting the model"
#It's now time to run the model
#Definitions:#
#Epoch = 1 pass throught all of the rows in train
#Batch = 1 or more samples considered by the model within an epoch before weights are updated

model.fit(x=data, y=labels, batch_size=768,
          epochs = 150, verbose=True)   


"Evaluating the model"

results = model.evaluate(x = data, y = labels)

for sample_i in range(1):
    print("Loss is:", results[0])
    print("Accuracy is:", results[1]*100)
    print("MAPE is:", results[2])









