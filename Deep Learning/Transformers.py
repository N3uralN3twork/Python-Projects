"""A Simple Interface to Text Classification via Hugging Face Transformers.
Author: Matt Quinn
Date: 16 February 2020
Source: https://colab.research.google.com/drive/1YxcceZxsNlvK35pRURgbwvkgejXwFxUt
Goal: Build a model that can predict the newsgroup category for a given post.
"""
###############################################################################
###                     1.  Define Working Directory                        ###
###############################################################################
import os
abspath = os.path.abspath('C:/Users/MatthiasQ.MATTQ/Desktop/Python Projects/Deep Learning/Yelp Reviews')
os.chdir(abspath)
###############################################################################
###                    2. Import Libraries and Models                       ###
###############################################################################
import pandas as pd
import tensorflow as tf
from sklearn.datasets import fetch_20newsgroups
import ktrain
from ktrain import text
from ktrain.text.models import


tf.__version__


# Load the dataset
categories = ["alt.atheism", "soc.religion.christian",
              "comp.graphics", "sci.med"]
train_df = fetch_20newsgroups(subset="train",
                              categories = categories,
                              shuffle=True, random_state=123)
test_df = fetch_20newsgroups(subset='test',
                             categories=categories, shuffle=True,
                             random_state=123)

x_train = train_df.data
y_train = train_df.target
x_test = test_df.data
y_test = test_df.target

######################################
# Choose your model to use here
MODEL_NAME = "distilbert-base-uncased"


t = text.Transformer(MODEL_NAME, maxlen=500, classes=train_df.target_names) #Initialize the specific model
trn = t.preprocess_train(x_train, y_train)
val = t.preprocess_test(x_test, y_test)
model = t.get_classifier() #Get the model
learner = ktrain.get_learner(model, batch_size = 4, #Decrease batch size if you run into an OOM error
                             train_data=trn, val_data=val)

# Training the Model:
learner.model.summary() #67 Million parameters, my poor GPU

learner.fit_onecycle(lr = 0.00005, epochs=2, verbose=1)

# Evaluating the Model:
learner.validate(val_data=(x_test, y_test))


# Making Predictions:
predictor = ktrain.get_predictor(model = learner.model, preproc=t)

predictor.predict("Jesus Christ is awesome")

learner.plot()

