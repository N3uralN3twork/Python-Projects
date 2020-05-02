# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:13:06 2019

@author: MatthiasQ
"""
import os
abspath = os.path.abspath('C:/Users/MatthiasQ.MATTQ/Desktop/Python Projects/Sci-kit Learn')
os.chdir(abspath)

import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn import metrics, preprocessing
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer



"Get the dataset"
bcancer = pd.read_excel("WisBreastCancer.xlsx")
bcancer.info()

"Train-Test Split"

#Define x and response variables
y = bcancer.Diagnosis
x = bcancer.drop(["Diagnosis", "ID"], axis = 1)


x_train, x_test, y_train, y_test = train_test_split(
                                        x, y,
                                        test_size = 0.20, #80/20
                                        random_state = 123,
                                        stratify = y)

"Preprocessing"
#I believe all of the independent variables are numeric, so we won't be OneHotEncoding anything?
#However, we will standardize the numeric variables


pipeline = make_pipeline(
        preprocessing.StandardScaler(),
        LogisticRegression(), #The model to use
        verbose = True)
pipeline


"Hyperparameters:"
#You can see the hyperparameters for logistic regression when printing the pipeline

logistichyper = {  "logisticregression__fit_intercept": [True], #We want beta0
                   "logisticregression__max_iter": [200], #200 iterations
                   "logisticregression__penalty": ['l1', 'l2'], #L2 regularization
                   "logisticregression__n_jobs": [-1]}  #use all processors
                   
scoring = {"Accuracy": make_scorer(metrics.accuracy_score),
           "AUC": 'roc_auc'}



"Cross-Validation"

clf = GridSearchCV(pipeline,
                   logistichyper,
                   scoring = scoring,
                   n_jobs = -1, #Uses all processors
                   cv = 10,
                   verbose = 1,
                   refit = "AUC")

"Fitting the Model"
clf.fit(x_train, y_train) #Takes less than a second


"Model Evaluation"
clf.best_params_
#97.8% accuracy
clf.cv_results_


"Now let's say I wanted to run multiple models, like KNN" 
"The full tutorial accomplishes this"


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

Models = {
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "SVClassifier": SVC()
         }

HyperParams = {} #Start with an empty dictionary


help(sklearn.linear_model.LogisticRegression())

HyperParams.update({"LogisticRegression":
                                    {"max_iter": [200],
                                     "verbose": [1],
                                     "n_jobs": [-1]
                                     }})
    
    
HyperParams.update({"RandomForest": { 
                                    "n_estimators": [200],
                                    "class_weight": ["balanced"],
                                    "max_features": ["auto", "sqrt", "log2"],
                                    "max_depth" : [3, 4, 5, 6, 7, 8],
                                    "min_samples_split": [0.005, 0.01, 0.05],
                                    "min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                    "n_jobs": [-1]
                                     }})


help(sklearn.ensemble.AdaBoostClassifier())

HyperParams.update({"AdaBoost":
                        {"n_estimators": [150],
                         "learning_rate": [0.001, 0.01, 0.05, 0.10, 0.25, 0.5, 1]
                         }})

    
help(sklearn.svm.SVC())
HyperParams.update({"SVClassifier":
                        {"kernel": ["rbf"],
                         "verbose": [True],
                         "gamma": [0.001, 0.0001]
                         }})


"Model Fitting"

help(GridSearchCV)

#WTF do i do now?
"Source: http://www.davidsbatista.net/blog/2018/02/23/model_optimization/"
class EstimatorSelectionHelper:

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X,y)
            self.grid_searches[key] = gs    

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]        
                scores.append(r.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]
#After this, pass in a dictionary of models and a dictionary of hyperparameters

pipeline = EstimatorSelectionHelper(Models, HyperParams)
pipeline.fit(x_train, y_train, cv=2)
results = pipeline.score_summary(sort_by = 'max_score')















from sklearn.metrics import confusion_matrix, classification_report

"I was thinking of ways to make predictions on all of the models for quite a while"
"Then I realized that would be pointless because I would only need to make predictions
"on the best model from the results."
"In the future, you'd have to read the results table and find the best model at the top
"along with its parameters to train."

#For this example, the best model is AdaBoost with:
#lr = 0.001,
#n_estimators = 300, since same as 400

help(AdaBoostClassifier)
BestModel = AdaBoostClassifier(n_estimators = 300,
                               learning_rate = 0.001,
                                random_state = 123)

BestModel.fit(x_train, y_train)
#Making Predictions

preds = BestModel.predict(x_test)

#############################################################
#                       11. Model Performance               #
#############################################################
from sklearn.metrics import roc_curve, auc
confusion_matrix(y_test, preds)

class_names = ["Malignant", "Benign"]
print(classification_report(y_test, preds,target_names = class_names))





#Use below for binary responses
#False positive rate / True Positive Rates
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, preds, pos_label = )
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
























