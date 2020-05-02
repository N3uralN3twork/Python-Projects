# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 12:58:51 2019

@author: MatthiasQ

Introduction to Sci-kit Learn
"""
import os
abspath = os.path.abspath('D:/Python Projects/Sci-kit Learn')
os.chdir(abspath)

"Scikit-Learn is THE machine learning library in Python.
"Also nice for beginners to learn too.
"This isn't a tutorial about EDA, although that's important
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

"1. Importing model families and other stuff"
#A 'family' of models are like Random Forests, NN, SVM's.
#Within each family is a set of models that you actually use
from sklearn.ensemble.forest import RandomForestRegressor

#For cross-validation
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

#For metrics
from sklearn.metrics import mean_squared_error, r2_score

#For saving models
from sklearn.externals import joblib





"2. Load in the dataset"
#You can read in data from a variety of sources via pandas
#Our dataset will be on wine sommeliers

url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
wine = pd.read_csv(url)

wine
#Notice how the separator is not a comma, but rather a semicolon
#To change the separator:
wine = pd.read_csv(url, sep = ';')
wine.info()
#1,599 obs and 12 variables
#We will be predicting the quality of the wine (numeric)


#Don't forget to standardize the numeric data

"3. Training and Test Splitting"
#First, set your x and y
y = wine.quality
x = wine.drop('quality', axis=1)

#Simple as that

#After that, you define the train-test split

x_train, x_test, y_train, y_test = train_test_split(
                                        x, y,
                                        test_size = 0.2,
                                        random_state = 123,
                                        stratify = y)

#That easy

"4. Preprocessing"
#First, standardize the numeric data
#Subtracting means and dividing be std. dev. for each numeric variable
#1. fit the transformer on the training set
#2. Apply transformer on training set
#3. Apply transformer on testing set

pipeline = make_pipeline(
        preprocessing.StandardScaler(),
        RandomForestRegressor(n_estimators = 100))
#Transform the data, then use your modelto fit the data



"5. Tuning Hyperparameters"
#Hyperparameters are higher-level information about the model and are usually
#set before training the model.

#To get the hyperparameters of a model type:
"pipeline.get_params()"
'https://scikit-learn.org'

pipeline.get_params()
#There are a lot of things you can tune with a random forest.
#19, actually

#If you remember in the full tutorial, early on we laid out the hyperparameters 
#that our models would use. This is the same exact thing

#What you see below is called dictionary format
#Keys are the names of the hyperparameter and values are the list of things to try
rfhyperparameters =  {  "randomforestregressor__max_features": ["auto", "sqrt", "log2"],
                        "randomforestregressor__max_depth" : [None, 5, 3, 1],
                        "randomforestregressor__n_jobs": [-1]}


"6. Cross-Validation"
#We are almost to fitting our models, but rn,
#we need to talk about cv.

#Essentially, CV is a process for reliably estimating performace via training and 
#evaluating the model several times.
#Steps:
    1. split data into K equal parts
    2. Preprocess K-1 training folds
    3. Train model on the same K-1 folds
    4. Preprocess the hold-out using same transformations
    5. Evaluate model using same hold-out fold
    6. Perform steps (2-5) K times
    7. Aggregate performance


https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

clf = GridSearchCV(pipeline,
                   rfhyperparameters,
                   n_jobs = -1, #Uses all processors
                   cv = 10,
                   verbose = 1)
#That's it!

"Metrics"
In order to have multiple metrics, you must make a new dictionary like-so.
scoring = {"AUC": sklearn.metrics.roc_auc_score,
           "Accuracy": sklearn.metrics.accuracy_score}


"7. Fitting the model"
clf.fit(X = x_train, y = y_train)

#To see what the best hyperparameters were:
clf.best_params_

#To see the best score
clf.best_score_


"8. Evaluate the model pipeline on the test data"
#To make new predictions:
y_pred = clf.predict(x_test)

r2_score(y_true = y_test, y_pred = y_pred)
#46.2% R^2 is pretty bad, so linear regression might be better

















