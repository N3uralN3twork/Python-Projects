# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 21:45:02 2019

@author: MatthiasQ
@Source: https://archive.ics.uci.edu/ml/datasets/adult
"""
import os
abspath = os.path.abspath('D:/Python Projects/Sci-kit Learn')
os.chdir(abspath)

###############################################################################
#                          1. Importing Libraries                             #
###############################################################################
# For reading, visualizing, and preprocessing data
import numpy as np
import pandas as pd #To read in the data
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn import metrics, preprocessing


# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier



###############################################################################
#                                 2. Get data                                 #
###############################################################################
names = ['Age', 'WorkClass', 'FnlWGT', 'Education',
         'Education_Num', 'Marital', 'Occupation',
         'Relationship', "Race", "Sex", "Capital_Gain",
         "Capital_Loss", "Hours_Week", "Native_Country",
         "Income"]
df = pd.read_excel("Adults.xlsx", names = names)


df.info()
#6 numeric,
#Rest are categorical
df.columns
###############################################################################
#                        3. Preprocessing the datasets                         #
###############################################################################
#Center and scale the numeric features
#OneHotEncode the categorical features
#Do I do this step before or after splitting?
#Before, because running it now will not make it work
Numeric = ["Age", "FnlWGT", "Education_Num", "Capital_Gain",
           "Capital_Loss", "Hours_Week"]
Categorical = ["WorkClass", "Education", "Marital", "Occupation", "Relationship",
               "Race", "Sex", "Native_Country"] #DO NOT INCLUDE THE OUTCOME VARIABLE

Features = df[["Age", "WorkClass", "FnlWGT", "Education",
              "Education_Num", "Marital", "Occupation",
              'Relationship', "Race", "Sex", "Capital_Gain",
              "Capital_Loss", "Hours_Week", "Native_Country"]]

Label = ["Income"]

X = df[Features]
y = df[Label]


null = df.isnull().sum()
print("Number of null values in each column:\n{}".format(null))
#0 missing values



#Explore the categorical features
"Source: https://www.dataquest.io/blog/machine-learning-preparing-data/"
for name in Categorical:
    print(name, ":")
    print(df[name].value_counts(),"\n")

#It says taht there are 2,799 ? values in WorkClass.
#That should leave us with 46,042 observations


"Drop rows with MISSING as the level, like WorkClass and Occupation have"
df = df.drop(df[df.WorkClass == "MISSING"].index) #46,042
df = df.drop(df[df.Occupation == "MISSING"].index)#46,032
df = df.drop(df[df.Native_Country == "MISSING"].index)#45,221

#I believe that should be it.

"Check again for weird levels."
for name in Categorical:
    print(name, ":")
    print(df[name].value_counts(),"\n")
#This still leaves us with plenty of observations.

from sklearn.preprocessing import StandardScaler, OneHotEncoder

#Each level gets its own variable and scale them
scaler = StandardScaler()
OHE = OneHotEncoder(sparse = False) 

scaled_columns =  scaler.fit_transform(df[Numeric])
encoded_columns = OHE.fit_transform(df[Categorical])

processed_data = np.concatenate([scaled_columns, encoded_columns],
                                axis = 1)
###############################################################################
#                       4. Creat train and test set                           #
###############################################################################
#Center and scale the numeric features
#OneHotEncode the categorical features
#Do I do this step before or after splitting?
#Before, because running it after train-test split will not make it work.
X = df[Features]
y = df.drop(columns = Features, axis = 1)
 

x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.20,
                                                    random_state = 123)


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer

preProcess = make_column_transformer(
        (StandardScaler(), Numeric),
        (OneHotEncoder(), Categorical))
data2 = preProcess.fit_transform(Features).toarray()

model = make_pipeline(
        preProcess,
        LogisticRegression())
model.fit(x_train, y_train)
print('logistic regression score: %f' % model.score(x_test, y_test))

HyperParams = {} #Start with an empty dictionary
HyperParams.update({"LogisticRegression":
                                    {"max_iter": [200],
                                     "verbose": [1],
                                     "n_jobs": [-1]
                                     }})
gscv = GridSearchCV(model,HyperParams, cv = 5, n_jobs = -1)

pipeline= EstimatorSelectionHelper(Models, HyperParams)
pipeline.fit(x_train, y_train, cv=4, n_jobs=-1)





























