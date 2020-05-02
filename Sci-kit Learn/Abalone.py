# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:02:31 2019

@author: MatthiasQ
"""
###############################################################################
###                     1.  Define Working Directory                        ###
###############################################################################
import os
abspath = os.path.abspath('D:/Python Projects/Sci-kit Learn')
os.chdir(abspath)


###############################################################################
###                    2. Import Libraries, Data, and Models                ###
###############################################################################
"Import libraries and models you want here"
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn import metrics, preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from keras.utils import to_categorical


"Get the dataset"
names = ["Sex", "Length", "Diameter", "Height", "WholeWeight",
         "ShuckedWeight", "VisceraWeight", "ShellWeight","Rings"]
df = pd.read_csv("Abalone.csv", names = names)
df.columns
df.dtypes


###############################################################################
###                      3. Feature Engineering                             ###
###############################################################################
"This is where you create new variables for use in your models."
#We have length, width, and height; so make a volume variable

df["Volume"] = (df["Length"] * df["Diameter"] * df["Height"])
df["AvgWeight"] = (df["WholeWeight"] + df["ShuckedWeight"])/2
df.head(2)

##############################################################################
###                      4. Preprocessing                                  ###
##############################################################################
"Preprocessing"
Continuous  = ["Length", "Diameter", "Height", "WholeWeight",
               "ShuckedWeight", "VisceraWeight", "ShellWeight",
               "Volume", "AvgWeight"]
Categorical = ["Sex"]



"Missing Values"
#To see how many missing values are in each variable
null = df.isnull().sum()
print("Number of null values in each column:\n{}".format(null))

"To remove missing values"



"Tabulation Data"
for name in Categorical:
    print(name, ":")
    print(df[name].value_counts(),"\n")
#What in the world is I?
    

print(df["Rings"].value_counts(), "\n")
#Drop rows with 5 or less obs. for our target variable
keep= [9,10,8,11,7,12,6,13,14,5,15,16,17,4,18,19,20,3,21,23,22]
df = df[df['Rings'].isin(keep)]

print(df["Rings"].value_counts(), "\n")

_______________________________________________________________________________

"One-Hot Encoding"
#Source:https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
#God-bless this person who made this function
def dummyEncode(df):
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        LE = LabelEncoder()
        for feature in columnsToEncode:
            try:
                df[feature] = LE.fit_transform(df[feature])
            except:
                print('Error encoding '+feature)
        return df
    
df = dummyEncode(df)


"Standard Scaling"
#Pretty much z-scores for each numeric variable
scaler = StandardScaler() #Define the scaling method
df[Continuous] = scaler.fit_transform(df[Continuous]) #Don't scale the target
#This took me like 3 days to figure out dude

del names
del null
del name
del keep
###############################################################################
###                     4. Train-Test Split                                 ###  
###############################################################################        

"Train-Test Split"

#Define X and Y variables
y = df.Rings
x = df.drop(["Rings"], axis = 1) #Drop any unneeded variables


x_train, x_test, y_train, y_test = train_test_split(
                                        x, y,
                                        test_size = 0.25, #75/25
                                        random_state = 123,
                                        stratify = y)

###############################################################################
#              5. Feature Selection: Removing highly correlated features      #
###############################################################################
# Filter Method: Spearman's Cross Correlation > 0.95
# Make correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt
corr_matrix = x_train.corr(method = "pearson").abs()

print(corr_matrix)

# Draw the heatmap
sns.set(font_scale = 1.0)
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr_matrix, cmap= "YlGnBu", square=True, ax = ax, annot = True)
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
ols = sm.OLS(y, x).fit()
ols.pvalues

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
    if(pmax > 0.15):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
#Only use important variables with low p-values
df = df[df.isin(selected_features_BE)]
###############################################################################
###                      6. Hyperparameter Tuning                           ###
###############################################################################
"Hyperparameters:"

"Now let's say I wanted to run multiple models, like KNN and AdaBoost" 
"The full tutorial accomplishes this"
"Don't forget: The parameters must be in square brackets

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
                                    {"max_iter": [300],
                                     "verbose": [2],
                                     "n_jobs": [-1]
                                     }})
    
    
HyperParams.update({"RandomForest": { 
                                    "n_estimators": [250],
                                    "class_weight": ["balanced"],
                                    "max_features": ["auto", "sqrt", "log2"],
                                    "max_depth" : [3, 4, 5, 6, 7, 8],
                                    "min_samples_split": [0.005, 0.01, 0.05],
                                    "min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                    "n_jobs": [-1]
                                     }})


help(sklearn.ensemble.AdaBoostClassifier())

HyperParams.update({"AdaBoost":
                        {"n_estimators": [250],
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
###                      7. Running Our Models                              ###
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
pipeline.fit(x_train, y_train, cv=4, n_jobs=-1)
results = pipeline.score_summary(sort_by = 'mean_score')
print(results)



# Get Performance Data


feature_names = x_train.columns
selected_features = feature_names[feature_selector.support_].tolist()




performance_curve = {"Number of Features": list(range(1, len(feature_names) + 1)),
                    "AUC": feature_selector.grid_scores_}
performance_curve = pd.DataFrame(performance_curve)

# Performance vs Number of Features
# Set graph style
sns.set(font_scale = 1.75)
sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
               "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
               'ytick.color': '0.4'})
colors = sns.color_palette("RdYlGn", 20)
line_color = colors[3]
marker_colors = colors[-1]

# Plot
f, ax = plt.subplots(figsize=(13, 6.5))
sns.lineplot(x = "Number of Features", y = "AUC", data = performance_curve,
             color = line_color, lw = 4, ax = ax)
sns.regplot(x = performance_curve["Number of Features"], y = performance_curve["AUC"],
            color = marker_colors, fit_reg = False, scatter_kws = {"s": 200}, ax = ax)

# Axes limits
plt.xlim(0.5, len(feature_names)+0.5)
plt.ylim(0.60, 0.925)

# Generate a bolded horizontal line at y = 0
ax.axhline(y = 0.625, color = 'black', linewidth = 1.3, alpha = .7)

# Turn frame off
ax.set_frame_on(False)

# Tight layout
plt.tight_layout()

# Save Figure
plt.savefig("performance_curve.png", dpi = 1080)






















