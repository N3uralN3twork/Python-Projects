###############################################################################
###                     1.  Define Working Directory                        ###
###############################################################################
import os
abspath = os.path.abspath('C:/Users/MatthiasQ.MATTQ/Desktop/Python Projects/FraudDetection')
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
from sklearn.metrics import make_scorer, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from keras.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold

"Get the dataset"
df = pd.read_csv("creditcard.csv")
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
#284807 rows
#31 columns

#Time doesn't look like a very useful variable
df = df.drop(["Time"], axis = 1)

"Drop Duplicates"
df.shape
df = df.drop_duplicates(keep='first')
df.shape

"Variable Classes"
Continuous = list(df.select_dtypes(include=['int64', 'float64']).columns)
Categorical = list(df.select_dtypes(include=['object', 'category']).columns)

Categorical.remove("Class")  # Replace Class with your response variable
Categorical  # Check to make sure your response is not in this list

"Tabulation Data"
for name in Categorical:
    print(name, ":")
    print(df[name].value_counts(), "\n")

"Scatterplot Matrix"

# Change colors and first line depending on your response classes.
df.Class.value_counts()
# Source: https://seaborn.pydata.org/examples/scatterplot_matrix.html
sns.set(style='ticks')
sns.pairplot(df, hue="Class", diag_kind='kde')  # Change Class to outcome variable

"Bar Chart"
sns.countplot(x="Class", data=df)
df["Class"].value_counts()


###############################################################################
###                      3. Feature Engineering                             ###
###############################################################################
"This is where you create new variables for use in your models."
df._columns

df.Amount.describe()
df["PurchaseSize"] = np.where(df["Amount"] >= 100,
                              "Large", "Small")

#Check to see
df.PurchaseSize.value_counts()




# correlation heatmap of dataset
def correlation_heatmap(df):
    _, ax = plt.subplots(figsize=(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)

    _ = sns.heatmap(
        df.corr(),
        cmap=colormap,
        square=True,
        cbar_kws={'shrink': .9},
        ax=ax,
        annot=True,
        linewidths=0.1, vmax=1.0, linecolor='white',
        annot_kws={'fontsize': 12}
    )

    plt.title('Pearson Correlation of Features', y=1.05, size=15)


correlation_heatmap(df)
##############################################################################
###                      4. Preprocessing                                  ###
##############################################################################
"Variable Classes"
"One-Hot Encoding"
def one_hot(df, cols):
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix = each, drop_first = False)
        df = pd.concat([df, dummies], axis = 1)
    return df
cols = Categorical

df = one_hot(df, cols)
df = df.drop("PurchaseSize", axis = 1)


Continuous = list(df.select_dtypes(include=['int64', 'float64']).columns)
Features =  Continuous
Features = Features.remove("Class")


"Missing Values"
# To see how many missing values are in each variable
null = df.isnull().sum()
print("Number of null values in each column:\n{}".format(null))
#Very nice!
_______________________________________________________________________________
del names
del null
del name
del colors

"Plot relationships"
grid = sns.PairGrid(data=df,
                    vars=Continuous)

grid = grid.map_upper(plt.scatter, color="darkred")
grid = grid.map_diag(plt.hist, bins=10, color="blue")
grid = grid.map_lower(sns.kdeplot, cmap="Reds")

###############################################################################
###                     5. Train-Test Split                                 ###
###############################################################################

"Train-Test Split"

# Define X and Y variables

# Replace Class with your response variable
y = df.Class
x = df.drop(["Class"], axis=1)  # Drop any unneeded variables

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.25,  # 75/25
    random_state=123,
    stratify=y)

###############################################################################
###             6. Feature Selection:                                       ###
###############################################################################
"Dimension Reduction"
# If your dataset that you loaded has a lot of features, that will make the code up ahead run a lot
# slower. So, you can select either PCA or LDA.
# LDA assumes normality and equal variances
# Specify how many variables you'd like to keep below
df.shape
N_Variables = 10
pca = PCA(n_components = N_Variables)
x_train = pca.fit_transform(x_train)
x_test  = pca.fit_transform(x_test)



"Drop Zero Variance Variables"
ZeroVar = VarianceThreshold(threshold = 0)
ZeroVar.fit(x_train)
len(x_train.columns[ZeroVar.get_support()])
constant_columns = [column for column in x_train.columns
                    if column not in x_train.columns[ZeroVar.get_support()]]

print(constant_columns)  # Which variables are constant?

# Original # Features
len(x_train.columns)
# New # Features
len(x_train.columns[ZeroVar.get_support()])

###############################################################################
###                      7. Hyperparameter Tuning                           ###
###############################################################################
"Hyperparameters:"

"Now let's say I wanted to run multiple models, like KNN and AdaBoost"
"The full tutorial accomplishes this"
"Don't forget: The parameters must be in square brackets

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Add models as you wish
Models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Ada Boost": AdaBoostClassifier(),
    "SVClassifier": SVC(),
    "KNN": KNeighborsClassifier()
}

HyperParams = {}  # Start with an empty dictionary

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
    "max_depth": [3, 4, 5, 6, 7, 8],
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
                        {"kernel": ["rbf", "linear"],
                         "verbose": [True],
                         "gamma": [0.001, 0.0001]
                         }})

help(sklearn.neighbors.KNeighborsClassifier)
HyperParams.update({"KNN":
                        {"n_neighbors": [3],
                         "algorithm": ["auto"],
                         "n_jobs": [-1]
                         }})

###############################################################################
###                      8. Running Our Models                              ###
###############################################################################
# WTF do i do now?
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
            gs.fit(X, y)
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
            return pd.Series({**params, **d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params), 1))

            all_scores = np.hstack(scores)
            for p, s in zip(params, all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]


# After this, pass in a dictionary of models and a dictionary of hyperparameters

pipeline = EstimatorSelectionHelper(Models, HyperParams)
pipeline.fit(x_train, y_train, cv=3, n_jobs=-1)  # Change number
results = pipeline.score_summary(sort_by='mean_score')
print(results)

###############################################################################
#                        9. "Feature Importance"                              #
###############################################################################

"This took quite a while to figure out the kinks"
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

Forest = RandomForestClassifier(n_estimators=250,
                                random_state=123,
                                verbose=2)

Forest.fit(x, y)

Importances = Forest.feature_importances_

x = pd.Series(x.columns)
y = pd.Series(Importances)
y = round(y, 4)

"Plotting Feature Importances"
# Source: https://www.w3resource.com/graphics/matplotlib/barchart/matplotlib-barchart-exercise-5.phpd
x = list(x)
Importance = list(y)

x_pos = [i for i, _ in enumerate(x)]

fig, ax = plt.subplots()
rects1 = ax.bar(x_pos, Importance, color='b')
plt.xlabel("Feature")
plt.ylabel("Gini Importance")
plt.title("Feature Importance via RF")
plt.xticks(x_pos, x, rotation=90)
# Turn on the grid
plt.minorticks_on()


# Customize the minor grid
def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                '%f' % float(height),
                ha='center', va='bottom')


autolabel(rects1)

###############################################################################
#                       10.  Making Predictions                                   #
###############################################################################
from sklearn.metrics import confusion_matrix, classification_report

"I was thinking of ways to make predictions on all of the models for quite a while"
"Then I realized that would be pointless because I would only need to make predictions
"on the best model from the results."
"In the future, you'd have to read the results table and find the best model at the top
"along with its parameters to train."

# For this example, the best model is AdaBoost with:
# lr = 0.001,
# n_estimators = 300, since same as 400

help(AdaBoostClassifier)
BestModel = AdaBoostClassifier(n_estimators=300,
                               learning_rate=0.001,
                               random_state=123)

BestModel.fit(x_train, y_train)
# Making Predictions

preds = BestModel.predict(x_test)

#############################################################
#                       11. Model Performance               #
#############################################################
from sklearn.metrics import roc_curve, auc

confusion_matrix(y_test, preds)

class_names = ["Iris-versicolor", "Iris-setosa", "Iris-virginica"]
print(classification_report(y_test, preds, target_names=class_names))

# Use below for binary responses
# False positive rate / True Positive Rates
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, preds)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc

plt.figure(figsize=(10, 10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, color='red', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')






















