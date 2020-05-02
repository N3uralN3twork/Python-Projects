# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:22:54 2019

@author: MatthiasQ
"""
###############################################################################
###                     1.  Define Working Directory                        ###
###############################################################################
import os
abspath = os.path.abspath('D:/Python Projects/Sci-kit Learn')
os.chdir(abspath)

###############################################################################
###                    2. Import Libraries and Models                       ###
###############################################################################
"Import libraries and models you want here"
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn import metrics, preprocessing
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from keras.utils import to_categorical


"Get the dataset"
names = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Class"]
df = pd.read_csv("IrisData.csv", names = names)
df.columns
df.dtypes


###############################################################################
###                      3. Exploratory Data Anaylsis                       ###
###############################################################################
"Exploratory Data Analysis"
plt.style.use('ggplot')
df.head(3)
df.info()
df.describe()
df["Class"].value_counts()
#310 rows
#6 features


"Drop Duplicates"
df = df.drop_duplicates(keep = 'first')


"Variable Classes"
#I believe all of the independent variables are numeric, so we won't be OneHotEncoding anything?
#However, we will standardize the numeric variables
Continuous  = list(df.select_dtypes(include = ['int64', 'float64']).columns)
Categorical = list(df.select_dtypes(include = ['object', 'category']).columns)

Categorical.remove("Class")   #Replace Class with your response variable
Categorical #Check to make sure your response is not in this list


"Tabulation Data"
for name in Categorical:
    print(name, ":")
    print(df[name].value_counts(),"\n")

"Scatterplot Matrix"

#Change colors and first line depending on your response classes.
df.Class.value_counts() 
#Source: https://seaborn.pydata.org/examples/scatterplot_matrix.html
sns.set(style = 'ticks')
sns.pairplot(df, hue = "Class") #Change Class to outcome variable



"Bar Chart"
sns.countplot(x = "Class", data = df)
df["Class"].value_counts()


###############################################################################
###                      3. Feature Engineering                             ###
###############################################################################
"This is where you create new variables for use in your models."
df.columns
#Area = Length * Width
#Create 2 new variables for Petal and Sepal Areas

df["SepalArea"] = df["SepalLength"] * df["SepalWidth"]
df["PetalArea"] = df["PetalLength"] * df["PetalWidth"]

#Reorder:
columns = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth",
           "SepalArea", "PetalArea", "Class"]
df = df[columns]










##############################################################################
###                      4. Preprocessing                                  ###
##############################################################################
"Variable Classes"
#I believe all of the independent variables are numeric, so we won't be OneHotEncoding anything?
#However, we will standardize the numeric variables
Continuous  = list(df.select_dtypes(include = ['int64', 'float64']).columns)
Categorical = list(df.select_dtypes(include = ['object', 'category']).columns)

Categorical.remove("Class")   #Replace Class with your response variable
Categorical #Check to make sure your response is not in this list




"Missing Values"
#To see how many missing values are in each variable
null = df.isnull().sum()
print("Number of null values in each column:\n{}".format(null))

"To remove missing values"
import sys
from impyute.imputation.cs import fast_knn, mice

sys.setrecursionlimit(10000)
imputed_training = fast_knn(df, k = 30)

#OR
imputed_training = mice(df)

_______________________________________________________________________________

"One-Hot Encoding"
#Source:https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
#God-bless this person who made this function
def dummyEncode(data):
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        LE = LabelEncoder()
        for feature in columnsToEncode: 
            try:
                df[feature] = LE.fit_transform(df[feature])
            except:
                print('Error encoding '+feature)
        return df
    
df = dummyEncode(data = df)


"Standard Scaling"
#Pretty much z-scores for each numeric variable
scaler = StandardScaler() #Define the scaling method
df[Continuous] = scaler.fit_transform(df[Continuous]) #Don't scale the target
#This took me like 3 days to figure out dude

del names
del null
del name
del colors
###############################################################################
###                     5. Train-Test Split                                 ###  
###############################################################################        

"Train-Test Split"

#Define X and Y variables

#Replace Class with your response variable
y = df.Class
x = df.drop(["Class"], axis = 1) #Drop any unneeded variables


x_train, x_test, y_train, y_test = train_test_split(
                                        x, y,
                                        test_size = 0.25, #75/25
                                        random_state = 123,
                                        stratify = y)

###############################################################################
###             6. Feature Selection:                                       ###
###############################################################################
"Filter Method: Spearman's Cross Correlation > 0.95"
# Make correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt
corr_matrix = x_train.corr(method = "spearman").abs()

# Draw the heatmap
sns.set(font_scale = 1.0)
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr_matrix, cmap= "viridis", square=True, ax = ax)
f.tight_layout()
plt.savefig("correlation_matrix.png", dpi = 1080)

# Select upper triangle of matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features
x_train = x_train.drop(to_drop, axis = 1)
x_test = x_test.drop(to_drop, axis = 1)

"Backward Elimination"
#Feed all possible variables and reduce # over several iterations
import statsmodels.api as sm
X1 = sm.add_constant(x)
ols = sm.OLS(y, X1).fit()
ols.pvalues

#Remove variables with p-values > 0.05
#Source: https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
cols = list(x.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X1 = x[cols]
    X1 = sm.add_constant(X1)
    model = sm.OLS(y,X1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax > 0.1): #Set the p-value gate here
        cols.remove(feature_with_p_max)
    else:
        break
    
selected_features_BE = cols
print(selected_features_BE)
#Only use important variables with low p-values
df = df[df.isin(selected_features_BE)]


"Embedded Method"
#Extracts features which contribute the most during training
#Irrelevant features are marked with a 0
import matplotlib

Lasso = LassoCV()
Lasso.fit(x, y)
coef = pd.Series(Lasso.coef_, index = x.columns)

print("Lasso picked " + str(sum(coef !=0)) + " variables and eliminated the other "
      + str(sum(coef ==0)) + " variables")

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef = coef.sort_values()
print(imp_coef)
imp_coef.plot(kind = 'barh', color = 'orange')

plt.title("Feature Importance via Lasso Model")


###############################################################################
###                      7. Hyperparameter Tuning                           ###
###############################################################################
"Hyperparameters:"

"Now let's say I wanted to run multiple models, like KNN and AdaBoost" 
"The full tutorial accomplishes this"
"Don't forget: The parameters must be in square brackets

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

#Add models as you wish
Models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "Ada Boost": AdaBoostClassifier(),
        "SVClassifier": SVC()
         }

HyperParams = {} #Start with an empty dictionary


help(sklearn.linear_model.LogisticRegression())

HyperParams.update({"Logistic Regression":
                                    {"max_iter": [200],
                                     "verbose": [1],
                                     "n_jobs": [-1]
                                     }})
    
    
HyperParams.update({"Random Forest": { 
                                    "n_estimators": [350],
                                    "class_weight": ["balanced"],
                                    "max_features": ["auto", "sqrt", "log2"],
                                    "max_depth" : [3, 4, 5, 6, 7, 8],
                                    "min_samples_split": [0.005, 0.01, 0.05],
                                    "min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                    "n_jobs": [-1]
                                     }})


help(sklearn.ensemble.AdaBoostClassifier())

HyperParams.update({"Ada Boost":
                        {"n_estimators": [300, 400],
                         "learning_rate": [0.001, 0.01, 0.05, 0.10, 0.25, 0.5]
                         }})

    
help(sklearn.svm.SVC())
HyperParams.update({"SVClassifier":
                        {"kernel": ["rbf"],
                         "verbose": [True],
                         "gamma": [0.001, 0.0001]
                         }})

help(GridSearchCV)


###############################################################################
###                      8. Running Our Models                              ###
###############################################################################
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

pipeline= EstimatorSelectionHelper(Models, HyperParams)
pipeline.fit(x_train, y_train, cv=3, n_jobs=-1) #Change number 
results = pipeline.score_summary(sort_by = 'mean_score')
print(results)



###############################################################################
#                        9. "Feature Importance"                              #
###############################################################################

"This took quite a while to figure out the kinks"
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

Forest = RandomForestClassifier(n_estimators = 250,
                                random_state = 123,
                                verbose = 2)

Forest.fit(x, y)

Importances = Forest.feature_importances_

x = pd.Series(x.columns)
y = pd.Series(Importances)
y =round(y, 4)




# Bring some raw data.
#Source: https://www.w3resource.com/graphics/matplotlib/barchart/matplotlib-barchart-exercise-5.phpd
x = list(x)
Importance = list(y)

x_pos = [i for i, _ in enumerate(x)]

fig, ax = plt.subplots()
rects1 = ax.bar(x_pos, Importance, color='b')
plt.xlabel("Feature")
plt.ylabel("Gini Importance")
plt.title("Feature Importance via RF")
plt.xticks(x_pos, x, rotation = 90)
# Turn on the grid
plt.minorticks_on()
# Customize the minor grid
def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%f' % float(height),
        ha='center', va='bottom')
autolabel(rects1)

    

###############################################################################
#                     10.  Making Predictions                                   #
###############################################################################
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

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
                               learning_rate = 0.001)

BestModel.fit(x_train, y_train)
#Making Predictions

preds = BestModel.predict(x_test)

confusion_matrix(y_test, preds)

class_names = ["Iris-versicolor", "Iris-setosa", "Iris-virginica"]
print(classification_report(y_test, preds,target_names = class_names))



























