"""
Created on Tue Aug 20 14:22:54 2019

@author: MatthiasQ
#Source 1: https://www.kaggle.com/pavansanagapati/simple-tutorial-dimensionality-reduction-methods
"""
###############################################################################
###                     1.  Define Working Directory                        ###
###############################################################################
import os
abspath = os.path.abspath('C:/Users/MatthiasQ.MATTQ/Desktop/Python Projects/Sci-kit Learn')
os.chdir(abspath)
###############################################################################
###                    2. Import Libraries and Models                       ###
###############################################################################
"Import libraries and models you want here"
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn import metrics, preprocessing
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.metrics import make_scorer, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold

"Get the dataset"
names = ["SepaLength", "SepalWidth", "PetalLength", "PetalWidth", "Class"]
df = pd.read_csv("IrisData.csv", names=names)
df.columns
df.dtypes

###############################################################################
###                      3. Exploratory Data Analysis                       ###
###############################################################################
"Exploratory Data Analysis"
plt.style.use('ggplot')
df.head(3)
df.info()
df.describe()
df["Class"].value_counts()
# 150 rows
# 4 features


"Variable Classes"
# I believe all of the independent variables are numeric, so we won't be OneHotEncoding anything?
# However, we will standardize the numeric variables
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
"To rename variables:"
# I intentionally misnamed the SepalLength variable
# in order to show how to rename a variable using pandas

# df.rename(columns = {"old": "new"})

df = df.rename(columns = {"SepaLength": "SepalLength"})
df.columns

"To refactor the levels of a variable in a dataset:"
df2 = df
df["Class"] = df2["Class"].replace({"Iris-virginica": "Virginica",
                                    "Iris-setosa": "Setosa",
                                    "Iris-versicolor": "Versicolor"})

df2["Class"].value_counts()

"To filter a dataset with one condition:"
df2 = df[df2["Class"] == "Setosa"]

"This is where you create new variables for use in your models."
df.columns
# Area = Length * Width
# Create 2 new variables for Petal and Sepal Areas

df["SepalArea"] = df["SepalLength"] * df["SepalWidth"]
df["PetalArea"] = df["PetalLength"] * df["PetalWidth"]

# Reorder:
columns = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth",
           "SepalArea", "PetalArea", "Class"]
df = df[columns]

# Check
df.columns


# correlation heatmap of dataset
def correlation_heatmap(df):
    _, ax = plt.subplots(figsize=(14, 14))
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

"Extracting Parts of a Date"
# Assuming you have a date column:
raw_data = {'name': ['Willard Morris', 'Al Jennings', 'Omar Mullins', 'Spencer McDaniel'],
            'birth_date': ['01-02-1996', '08-05-1997', '04-28-1996', '12-16-1995']}
time = pd.DataFrame(raw_data, index=['Willard Morris', 'Al Jennings', 'Omar Mullins', 'Spencer McDaniel'])
time
# To extract the year:
time["year"] = pd.DatetimeIndex(data=time['birth_date']).year

# To extract the month:
time['month'] = pd.DatetimeIndex(data=time['birth_date']).month

# To extract the weekday:
time["weekday"] = pd.DatetimeIndex(data=time['birth_date']).dayofweek





"Working with Text Variables"

# If you want to say, extract the title from a given name:
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)


df["Title"] = df["Name"].apply(get_title)

# Drop rows with "Major, Mlle, Col, Lady, Countess, Capt, Ms, Sir, Mme, Don, Jonkheer"
# Source: https://stackoverflow.com/questions/28679930/how-to-drop-rows-from-pandas-data-frame-that-contains-a-particular-string-in-a-p
to_drop = ["Major", "Mlle", "Col", "Lady", "Countess", "Capt", "Ms", "Sir", "Mme", "Don", "Jonkheer"]
df = df[~df["Title"].isin(to_drop)]

df.Title.value_counts()




"To combine multiple categories into 1 in case you want to reduce the number of levels:"
#ource: https://www.youtube.com/channel/UCnVzApLJE2ljPZSeQylSEyg/community?lb=UgxR_tShW2Q_p1IYmjh4AaABCQ
top_four = category.value_counts().nlargest(4).index #The number of original levels to keep
top_four

category = category.where(category.isin(top_four), other = "Other")
category.value_counts()

def MakeOther(Category, N):
    top_n = Category.value_counts().nlargest(N).index
    print(top_n)




##############################################################################
###                      4. Preprocessing                                  ###
##############################################################################
"Variable Classes"
# I believe all of the independent variables are numeric, so we won't be OneHotEncoding anything?
# However, we will standardize the numeric variables
Continuous =  list(df.select_dtypes(include=['int64', 'float64']).columns)
Categorical = list(df.select_dtypes(include=['object', 'category']).columns)

Categorical.remove("Class")  # Replace Class with your response variable
Categorical  # Check to make sure your response is not in this list

Features = columns.remove("Class")  # Change Class to your response variable


"To Drop Duplicates:"
df = df.drop_duplicates(keep='first')

"Missing Values"
# To see what % of the data is missing for each variable
missing_values = df.isnull().sum()/len(df)*100
missing_values[missing_values>0].sort_values(ascending = False)
missing_values



"TO REMOVE VARAIBLES WITH HIGH % OF MISSING VALUES:"
#This is a custom function I made based on someone else's work
#Source: https://www.kaggle.com/pavansanagapati/simple-tutorial-dimensionality-reduction-methods
def high_missing_filter(df, missingPercent):
    variables = df.columns
    variable = []
    for i in range (0, len(df.columns)):
        if missing_values[i] <= missingPercent:
            variable.append(variables[i])
    df = df.filter(items = variable)  # Keep only rows with missing % under threshold
    return df

df = high_missing_filter(df = df, missingPercent = 60)
df.shape



"To impute:"
from impyute.imputation.cs import fast_knn, mice
imputed_training = mice(df)


_______________________________________________________________________________
"One-Hot Encoding"


def one_hot(df, cols):
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df

df = one_hot(df, Categorical)

# Probably shouldn't use the one below
"Label Encoding"
# Source:https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
# God-bless this person who made this function
"def dummyEncode(data):
columnsToEncode = list(df.select_dtypes(include=['category', 'object']))
LE = LabelEncoder()
for feature in columnsToEncode:
    try:
        df[feature] = LE.fit_transform(df[feature])
    except:
        print('Error encoding ' + feature)
return df

df = dummyEncode(data=df)


"Standard Scaling"
# Source: https://www.kaggle.com/discdiver/guide-to-scaling-and-standardizing
# Plot the distributions of your variables to check for skew

df.plot.kde()
#Good idea to scale your numeric variabes.
# Pretty much z-scores for each numeric variable
scaler = preprocessing.RobustScaler()  # Define the scaling method

# You shouldn't always use a standard scaler, especially if your data is skewed
"If your data is skewed, use the Robust Scaler"
"If you need normalized features, use Standard Scaler"
"Use MinMax as default, if no outliers"
preprocessing.StandardScaler()
"or"
preprocessing.MinMaxScaler()

# Transform the data with this:
df[Continuous] = scaler.fit_transform(df[Continuous])  # Don't scale the target
# This took me like 3 days to figure out dude

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
# N_Variables = 3
# pca = PCA(n_components = N_Variables)
# x2 = df.loc[:, Features].values
# principalComponents = pca.fit_transform(x2)
# newDF = pd.DataFrame(data = principalComponents,
columns = ["PCA 1", "PCA 2", "PCA 3"])
# DF = pd.concat([newDF, df["Class"]], axis = 1)




"Drop Zero Variance Variables"
"Keep in mind, I made this function myself on 21-9-2019"
"Just some extra practice for definition writing"
def ZeroVariance(x_train, threshold):
    ZeroVar = VarianceThreshold(threshold = threshold)
    ZeroVar.fit(x_train)
    constant_columns = [column for column in x_train.columns
                        if column not in x_train.columns[ZeroVar.get_support()]]
    return constant_columns

ZeroVariance(x_train, threshold = 0) # Which variables are constant?

# Drop the zero-variance variables
x_train = ZeroVar.transform(x_train)
x_test  = ZeroVar.transform(x_test)







"Filter Method: Spearman's Cross Correlation > 0.95"
# Make correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt
corr_matrix = x_train.corr(method = "spearman").abs()

# Select upper triangle of matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop =[column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features
x_train = x_train.drop(to_drop, axis = 1)
x_test = x_test.drop(to_drop, axis = 1)

"Backward Elimination"
# Feed all possible variables and reduce # over several iterations
import statsmodels.api as sm
X1 = sm.add_constant(x)
ols = sm.OLS(y, X1).fit()
ols.pvalues

# Remove variables with p-values > 0.15
# Source: https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
cols = list(x.columns)
pmax = 1
while (len(cols) > 0):
    p = []
    X1 = x[cols]
    X1 = sm.add_constant(X1)
    model = sm.OLS(y, X1).fit()
    p = pd.Series(model.pvalues.values[1:], index=cols)
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if (pmax > 0.15:  # Set the p-value gate here
    cols.remove(feature_with_p_max)
    else:
        break

selected_features_BE = cols
print(selected_features_BE)
# Only use important variables with low p-values
df = df[df.isin(selected_features_BE)]

###############################################################################
###                      7. Hyperparameter Tuning                           ###
###############################################################################
"Hyperparameters:"

"Now let's say I wanted to run multiple models, like KNN and AdaBoost"
"The full tutorial accomplishes this"
"Don't forget: The parameters must be in square brackets"

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Add models as you wish
Models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Ada Boost": AdaBoostClassifier(),
    "SVClassifier": SVC(),
    "KNN": KNeighborsClassifier(),
    "LDA": LinearDiscriminantAnalysis()
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
help(sklearn.discriminant_analysis.LinearDiscriminantAnalysis)
HyperParams.update({"LDA":
                        {"solver": ["eigen"],
                         "store_covariance": [False],
                         "shrinkage": [None]
                         }})

###############################################################################
###                      8. Running Our Models                              ###
###############################################################################
# WTF do i do now?
#It took me a like 2 days to figure out what to do after this step but the search
#was worth it
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

    def fit(self, X, y, cv = 3, n_jobs = -1, verbose = 1, scoring = None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv = cv, n_jobs = n_jobs,
                              verbose = verbose, scoring = "accuracy", refit = refit,
                              return_train_score = True)
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
                "range_score": (max(scores) - min(scores))         #My own metric
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


# After this, pass in a dictionary of models and a dictionary of hyperparameters to use

pipeline = EstimatorSelectionHelper(Models, HyperParams)
pipeline.fit(x_train, y_train,
             cv = 3, n_jobs = -1)  # Change number of cv to whatever you want
results = pipeline.score_summary(sort_by='mean_score')
print(results)

###############################################################################
#                        9. "Feature Importance"                              #
###############################################################################

"This took quite a while to figure out the kinks"

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



