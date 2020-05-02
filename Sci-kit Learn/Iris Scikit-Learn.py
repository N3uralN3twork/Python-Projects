# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:51:44 2019

@author: MatthiasQ
"""
#0. Set working directory
import os
abspath = os.path.abspath('D:/Python Projects/Sci-kit Learn')
os.chdir(abspath)

#1. Load the libraries
import numpy as np
import pandas as pd
import scipy
import sklearn

#2. Prepare the dataset

names = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Class"]
iris = pd.read_csv("IrisData.csv", names = names)

#Change the class names
iris.Class.value_counts()
iris["Class"].replace(
        to_replace = ["Iris-versicolor", "Iris-virginica", "Iris-setosa"],
        value = ["Versicolor", "Virginica", "Setosa"],
        inplace = True)

iris.Class.value_counts()
#Holy shit it worked

"3.  Train-test split"
from sklearn.model_selection import train_test_split

Features = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
x_train, x_test , y_train, y_test = train_test_split(
        iris[Features],
        iris["Class"],
        test_size=0.2, #80/20 split
        random_state=123)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
#The 1st and 2nd should match and the 3rd and 4th should match

"4. Preprocessing"
Continuous = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


#There are no categorical features, only continuous
preProcess = make_column_transformer(
        (StandardScaler(), Continuous))

"5. Model Building"

#Logistic Regression
logistic = make_pipeline(
        preProcess,
        LogisticRegression(verbose=True))

#Multi-layer perceptron
nn = make_pipeline(
        preProcess,
        MLPClassifier(batch_size = 30,
                      alpha = 0.0001,
                      solver = 'sgd',
                      activation = "relu",
                      max_iter=3000))

#Support-vector classifier
svc = make_pipeline(
        preProcess,
        SVC())

"6. Fitting the model"
logistic.fit(x_train, y_train)
nn.fit(x_train, y_train)
svc.fit(x_train, y_train)

"7. Scoring the model"
logistic.score(x_test, y_test)
nn.score(x_test, y_test)
svc.score(x_test, y_test)

#Surprisingly, the logistic model performed better than the nn
#Prolly due to data sparcity and the small network size



help(MLPClassifier)













