# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:24:00 2019

@author: MatthiasQ
Machine Learning and Deep Learning Practice
The goal is to predict whether an employee leaves the company
with various variables like salary
"""
import os
abspath = os.path.abspath('C:/Users/MatthiasQ.MATTQ/Documents/Python Projects/Scikit-LearnTutorial')
os.chdir(abspath)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import scipy

#Read in the dataset
data = pd.read_csv("Churn_Modelling.csv")
#No point in keeping the first 3 variables
data.columns
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)
data.columns #It worked!

#Changing data types:
data.dtypes
#Notice how the categorical variables are set as integers, not good.
data['Geography'] = pd.Categorical(data.Geography)
data['Gender'] = pd.Categorical(data.Gender)
data['HasCrCard'] = pd.Categorical(data.HasCrCard)
data['IsActiveMember'] = pd.Categorical(data.IsActiveMember)
data['Exited'] = pd.Categorical(data.Exited)

data.dtypes
#coolio

"Train-Test Split & Features"
from sklearn.model_selection import train_test_split
Features = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]

Categorical = ["Geography", "Gender", "HasCrCard", "IsActiveMember"]
Continuous  = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
               "EstimatedSalary"]

x_train, x_test, y_train, y_test = train_test_split(
        data[Features],
        data["Exited"],
        test_size=0.2, #80/20 split
        random_state=123)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
#Looks good

"Preprocessing"
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline

preProcess = make_column_transformer(
        (StandardScaler(), Continuous),
        #Must be numeric, not a string
        (OneHotEncoder(sparse=False), Categorical)
        )
#Test if pipeline works
preProcess.fit_transform(x_train).shape
#Nice

"Model Building"
#We'll train 3 different machine learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

logistic = make_pipeline(
           preProcess,
           LogisticRegression(verbose=True))

rf = make_pipeline(
        preProcess,
        RandomForestClassifier(verbose = 1, n_jobs=10))

svc = make_pipeline(
        preProcess,
        SVC(verbose = 1))

"Model Fitting"
logistic.fit(x_train, y_train)
rf.fit(x_train, y_train)
svc.fit(x_train, y_train)

"Model Scoring"
logistic.score(x_test, y_test) #81.3%
rf.score(x_test, y_test)  #84.15%
svc.score(x_test, y_test) #85.8%

"""So it looks like the best model was the support vector classifier
Nice"""


"Deep Learning"
#We just used 3 ML algorithms, but now we'll try some deep learning
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

np.random.seed(123)

#Create the two datasets
X = data[Features]
Y = data["Exited"]


"One-Hot Encoding"
"https://towardsdatascience.com/encoding-categorical-features-21a2651a065c"
#Extract the categorical features using a boolean mask
X.dtypes
categorical_feature_mask = X.dtypes==object
#Filter categorical columns and turn into a list
categorical_cols = X.columns[categorical_feature_mask].tolist()
#Geography and Gender are 2 categorical variables

labelencoder = LabelEncoder()
#Apply the encoder to each categorical column
X[categorical_cols] = X[categorical_cols].apply(lambda col:
    labelencoder.fit_transform(col))

X[categorical_cols].head(10)
#It worked!

#Notice how geography is either 0,1,2
#That's because there are 3 countries in the dataset the employees are from
pd.value_counts(data.Geography)
pd.crosstab(data.Geography, data.Gender)



#Encode the "Exited" values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encodedY = encoder.transform(Y)
#That's it for one-hot encoding!


"Model Building"
#Baseline
model = Sequential()
model.add(Dense(10, input_dim = 10, activation = 'relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))

"Model Compiling"
model.compile(loss = keras.losses.binary_crossentropy,
              optimizer = 'adam',
              metrics = ['accuracy'])

"Model Fitting"
model.fit(x=X, y=encodedY, batch_size=200,
          epochs = 100, verbose=True) 
#interesting





















