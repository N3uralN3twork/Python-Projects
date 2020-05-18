"""
Created on 9th October, 2019
This is a never-ending project, always something useful to add
Source 1: https://github.com/reiinakano/scikit-plot
Source 2: https://github.com/sepandhaghighi/pycm
Source 3: https://github.com/codiply/blog-ipython-notebooks/blob/master/scikit-learn-estimator-selection-helper.ipynb
Source 4: https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english
Source 5: https://github.com/erykml/medium_articles/blob/master/Machine%20Learning/feature_importance.ipynb
Source 6: https://stackoverflow.com/questions/31632637/label-axes-on-seaborn-barplot
Source 7: https://stackabuse.com/applying-filter-methods-in-python-for-feature-selection/
Source 8: https://scikit-plot.readthedocs.io/en/stable/metrics.html
Source 9: https://statcompute.wordpress.com/2013/08/23/multinomial-logit-with-python/
Notes:
    Supposed to drop Zero-Variance variables before you center and scale them.
    Deal with missing values first instead of ignoring them.
    Use a likelihood-ratio test for feature selection
    Added functionality to identify duplicate variables
"""

###############################################################################
###                     1.  Define Working Directory                        ###
###############################################################################
import os
abspath = os.path.abspath("C:/Users/miqui/OneDrive/Python-Projects/Sci-kit Learn")
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
import missingno as msno
import scikitplot as skplot
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold
import statsmodels.api as sm
pd.set_option('display.max_columns', 10)

"Import the data set"
names = ['Ag', 'WorkClass', 'FnlWGT', 'Education',
         'Education_Num', 'Marital', 'Occupation',
         'Relationship', "Race", "Sex", "Capital_Gain",
         "Capital_Loss", "Hours_Week", "NativeCountry",
         "Income"]
df = pd.read_csv("Datasets/adults.csv", names = names, verbose=1)
#If you have a unique identifier, set that as your index_col when reading in the csv
df.columns
df.dtypes

#6 continuous
#9 categorical

###############################################################################
###                      3. Exploratory Data Analysis                       ###
###############################################################################
"Exploratory Data Analysis"
plt.style.use('ggplot')
df.head(3)
df.info()
df.describe()
df["Income"].value_counts()

#Create a histogram of the continuous variables
df.hist()
plt.show()
df.boxplot()
plt.show()


# 32,561 rows
# 15 features


"Variable Classes"

Continuous  = list(df.select_dtypes(include=['int64', 'float64']).columns)
Categorical = list(df.select_dtypes(include=['object', 'category']).columns)

Categorical.remove("Income")  # Replace Class with your response variable
Categorical  # Check to make sure your response is not in this list

"Tabulation Data"
for name in Categorical:
    print(name, ":")
    print(df[name].value_counts(), "\n")



#It looks like we're going to have to combine some levels to make it more manageable
    #So, like top 5 countries and top 6 employment options
#Also, I'm just going to remove the data where the level of the country is missing
    #This should be done prior to combining levels into an "Other" category

"Scatterplot Matrix"
# Change colors and first line depending on your response classes.
# Source: https://seaborn.pydata.org/examples/scatterplot_matrix.html
sns.set(style = 'ticks')
sns.pairplot(df, hue = "Income", diag_kind = 'kde')  # Change Class to outcome variable
plt.show()


"Bar Charts"
sns.countplot(x = "Income", data = df)
plt.show()
df["Income"].value_counts()
#More than 3 times as many people make less than $50,000 than those who make more.


sns.countplot(x = "Relationship", data = df)
plt.show()
sns.countplot(x = "Marital", data = df)
plt.show()
#I'm just going to combine Married-spouse-absent and Married-AF-Spouse
###############################################################################
###                      3. Feature Engineering                             ###
###############################################################################
"To rename variables:"
# I intentionally misnamed the Age variable
# in order to show how to rename a variable using pandas

# df.rename(columns = {"old": "new"})
df = df.rename(columns = {"Ag": "Age"})
df.columns

"To refactor the levels of a variable in a dataset:"

df["Income"] = df["Income"].map({" <=50K": "Less50K",
                                 " >50K": "More50K"})

"This is where you create new variables for use in your models."
#1. Create a variable based on seniority of the person's age
df["Seniority"] = np.where(df["Age"] >= 50, "Senior", "Youth")

#2. Do they work overtime?
df["Overtime"] = np.where(df["Hours_Week"] > 40, "Yes", "No")

#3. Net Capital Gain
df["NetCapitalGain"] = (df["Capital_Gain"] - df["Capital_Loss"])

#4. Government worker or private sector
gov_dict = {" Private": "No", " Self-emp-not-inc": "No",
            " Local-gov": "Yes", " ?": "?", " State-gov": "Yes",
            " Self-emp-inc": "No", " Federal-gov": "Yes",
            " Without-pay": "No", " Never-worked": "No"}

df["GovWorker"] = df["WorkClass"].map(gov_dict)

#Why tf are there leading spaces in all of these damn entries.

#5. Let's drop the Education variable since EduNum has the same information
#   as well as Capital_Gain and Capital_Loss
to_drop = ["Education", "Capital_Gain", "Capital_Loss", "WorkClass"]
df.drop(to_drop, axis = 1, inplace = True)


#6. Drop rows that have "?" as a value in any categorical variable
#Source: https://stackoverflow.com/questions/31663426/python-pandas-drop-rows-from-data-frame-on-string-match-from-list/31663495
to_drop = ["?", " ?"]
df = df[~df["Occupation"].isin(to_drop)]
df = df[~df["NativeCountry"].isin(to_drop)]
df = df[~df["GovWorker"].isin(to_drop)]


#7. Combine Marital categories manually
marital_dict = {" Never-married": "Never-married",
                " Married-civ-spouse": "Married",
                " Divorced": "Divorced",
                " Separated": "Separated",
                " Widowed": "Widowed",
                " Married-spouse-absent": "Married",
                " Married-AF-spouse": "Married"}

df["Marital"] = df["Marital"].map(marital_dict)
sns.factorplot(x = "Marital", hue = "Income",
               data = df, kind = "count")
plt.show()
#Clearly, if you are married, you're chances of making > 50K are higher
#Those who are single probably lean towards the younger side and thus aren't
#as advanced in his/her careers.


#8. Combine some levels in order to reduce the number of levels for each categorical variable
#df = df.loc[df["NativeCountry"].isin([" United-States", "Mexico"])]
def CombineCategories(df, Category: str, N: int):
    top_N = df[Category].value_counts().nlargest(N).index
    update = df[Category].where(df[Category].isin(top_N), other = "Other")
    return update

country = CombineCategories(df, "NativeCountry", N = 2) #Keep top 2 categories and rest are "Other"
df["NativeCountry"] = country
df["NativeCountry"].value_counts()
#This is nice because it's like a big puzzle that you have to find the pieces for and then fit those pieces in.


occupation = CombineCategories(df, "Occupation", N = 9)
df["Occupation"] = occupation
df.Occupation.value_counts()


#8. Reorder the columns:
columns = ["Age", "FnlWGT", "Education_Num", "Marital", "Occupation",
           "Relationship", "Race", "Sex", "Hours_Week", "NativeCountry",
           "Seniority", "Overtime", "NetCapitalGain", "GovWorker", "Income"]
df = df[columns]

# Check
df.columns


# correlation heatmap of dataset
def correlation_heatmap(df, method = "pearson"):
    _, ax = plt.subplots(figsize=(14, 14))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    _ = sns.heatmap(
        df.corr(method = method),
        cmap=colormap,
        square=True,
        cbar_kws={'shrink': .9},
        ax=ax,
        annot=True,
        linewidths=0.1, vmax=1.0, linecolor='white',
        annot_kws={'fontsize': 12}
    )
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    plt.show()


correlation_heatmap(df, method = "pearson")
#The inter-correlations look good.

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
def get_title(name: str):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)


df["Title"] = df["Name"].apply(get_title)

# Drop rows with "Major, Mlle, Col, Lady, Countess, Capt, Ms, Sir, Mme, Don, Jonkheer"
# Source: https://stackoverflow.com/questions/28679930/how-to-drop-rows-from-pandas-data-frame-that-contains-a-particular-string-in-a-p
to_drop = ["Major", "Mlle", "Col", "Lady", "Countess", "Capt", "Ms", "Sir", "Mme", "Don", "Jonkheer"]
df = df[~df["Title"].isin(to_drop)]

df.Title.value_counts()

#Test
df["AaA"] = 1
##############################################################################
###                      4. Pre-processing                                 ###
##############################################################################
Continuous  = list(df.select_dtypes(include=['int64', 'float64']).columns)

"To Drop Duplicate Observations:"
def duplicate(df): # Addendum: 17 April 2020
    start = df.shape[0]
    df2 = df.drop_duplicates(keep = "first")
    end = df2.shape[0]
    print(f"Dropped {start-end} duplicates")
    return df2
df = duplicate(df)


"To Drop Duplicate Variables:"
def duplicate_vars(df): # Addendum: 17 May 2020
    """Returns all of the duplicate variables by name
       Then you would delete them using a dictionary"""
    variables = list(df.columns)
    predictors = [var for var in variables if variables.count(var) >= 2]
    return set(predictors)

duplicate_vars(df)

"Missing Value Analysis"
# To see what % of the data is missing for each variable
missing_values = df.isnull().sum()/len(df)*100
missing_values[missing_values>0].sort_values(ascending = False)
missing_values

#Heatmap of missing values:
sns.heatmap(df.isnull(), cbar = True)
plt.show()
#All solid means no missing values
#Noice

msno.matrix(df)
plt.show()

# To remove all rows that have 1 or more missing values:
df.shape
df = df.dropna()

"TO REMOVE VARIABLES WITH HIGH % OF MISSING VALUES:"
# This is a custom function I made based on someone else's work
# Source: https://www.kaggle.com/pavansanagapati/simple-tutorial-dimensionality-reduction-methods
def high_missing_filter(df, missingPercent):
    variables = df.columns
    variable = []
    for i in range (0, len(df.columns)):
        if missing_values[i] <= missingPercent:
            variable.append(variables[i])
    result = df.filter(items = variable)  # Keep only rows with missing % under threshold
    return result

df = high_missing_filter(df = df, missingPercent = 60)
df.shape # No missing data in our dataset


"To check for variables that have non-unique values:"
df2 = df
df2["Filler"] = "Not Unique"
def nonunique_columns(df: pd.DataFrame):
    for col in df.columns:
        if len(df[col].unique()) == 1:
            return col

nonunique_columns(df2) #Noice
#Then you would obviously drop the variable(s) listed since they won't add anything
#to your models ahead.

df = df.drop(columns = ["Filler"]) # Pass a list to drop the columns


"To impute:"
from impyute.imputation.cs import mice
imputed_training = mice(df)


"Drop Zero-Variance Variables:"
#Addendum: 3 January 2020
stats = pd.DataFrame(df[Continuous].describe()) #Create the dataset
stats = np.transpose(stats) #Transpose to get stats column-wise
plot = sns.barplot(x = stats.index, y = stats["mean"]) #Plot for easier interpretation
plot.set(ylabel = "Standard Deviation", xlabel = "Variable")
plt.show()

"Keep in mind, I made this function myself on 21-9-2019"
"Just some extra practice for definition writing"
def LowVariance(df, Continuous, Threshold):
    dfCont = df[Continuous] # Select just the continuous data
    selector = VarianceThreshold(threshold = Threshold)
    selector.fit(dfCont)
    constant_columns = [column for column in dfCont.columns
                        if column not in dfCont.columns[selector.get_support()]]
    return constant_columns # Which variables are constant?

#Show the low-variance variables:
LowVariance(df, Continuous, Threshold = 0.0)

# Drop the zero-variance variables here.
df = df.drop("AaA", axis = 1)

Continuous = list(df.select_dtypes(include=['int64', 'float64']).columns)
Categorical = list(df.select_dtypes(include=['object', 'category']).columns)
Categorical.remove("Income")


_______________________________________________________________________________
"1. One-Hot Encoding"
# This should be used for nominal variables
# Make a list of categorical variables first
# Or, better yet, make a list of all categorical variables and then delete just the ordinal ones
Nominal = list(df.select_dtypes(include=['object', 'category']).columns)
# Remove the ordinal and response variables, one by one
Nominal.remove("Income")
Nominal.remove("Seniority")
# One-hot encode the nominal variables
df = pd.get_dummies(df, drop_first = True, columns = Nominal)

"2. Ordinal Label Encoding"
# This should be used for ordinal variables
# What I'm thinking is that ordinal variables should be encoded before nominal variables?
# I talked to my friends and it apparently doesn't matter which variable type you do first.
def OrdinalEncode(data, OrdinalColumns: list):
    #You have to make a list of your ordinal variables first
    OE = OrdinalEncoder()
    for feature in OrdinalColumns:
        try:
            data[feature] = OE.fit_transform(data[feature])
        except:
            print('Error encoding ' + feature)
    return data

Ordinal = ["Seniority"] #This is where you make a list of your ordinal features.
df = OrdinalEncode(data = df, OrdinalColumns = Categorical)
# How do I know that this preserves the order?
# I suppose you could use a custom dictionary for each ordinal variable
# However, that seems tedious...



"3. Scaling Continuous Variables"
# Source: https://www.kaggle.com/discdiver/guide-to-scaling-and-standardizing
# Plot the distributions of your variables to check for skew


df[Continuous].describe()
df[Continuous].plot.kde()
plt.show()

#Good idea to scale your numeric variables.
# Pretty much z-scores for each numeric variable
scaler = preprocessing.RobustScaler()  # Define the scaling method


# You shouldn't always use a standard scaler, especially if your data is skewed
#As we saw earlier, some of the numeric features were quite skewed, like age and FNLWGT
"If your data is skewed, use the Robust Scaler"
"If your features are normally distributed, use the Standard Scaler"
"Use MinMax as default, if no outliers"
preprocessing.StandardScaler()
"or"
preprocessing.MinMaxScaler()

# Transform the data with this:
df[Continuous] = scaler.fit_transform(df[Continuous])  # Don't scale the target
# This took me like 3 days to figure out dude

#10/21/2019
#For some reason, the NetCapitalGain variable won't behave properly and scale down, so I scaled it myself using a Z-score
df["NetCapitalGain"] = (df["NetCapitalGain"] - np.mean(df["NetCapitalGain"])) / np.std(df["NetCapitalGain"])
df["NetCapitalGain"].describe()

df[Continuous].plot.kde()
plt.show()

df[Continuous].describe() # To check your work

"Plot relationships"
grid = sns.PairGrid(data=df,
                    vars=Continuous)
grid = grid.map_upper(plt.scatter, color="darkred")
grid = grid.map_diag(plt.hist, bins=10, color="blue")
grid = grid.map_lower(sns.kdeplot, cmap="Reds")
plt.show()



###############################################################################
###             5. Feature Selection:                                       ###
###############################################################################
"Dimension Reduction"
# If your data that you loaded has a lot of features, that will make the code up ahead run a lot
# slower. So, you can select either PCA or LDA.
# LDA assumes normality and equal variances
# Specify how many variables you'd like to keep below
#N_Variables = 20
#pca = PCA(n_components = N_Variables)
#x2 = df.loc[:, Features].values
#principalComponents = pca.fit_transform(x2)
#newDF = pd.DataFrame(data = principalComponents,
#columns = ["PCA 1", "PCA 2", "PCA 3"])
#DF = pd.concat([newDF, df["Class"]], axis = 1)


"Feature Selection:"
# Addendum: 17 April 2020
# In CDA/MTH 531, you learned about feature selection by fitting a model with and without the covariate,
# after which you would run a LR test or a chi-square test.
# I know how to do this in R, but I'm going to figure out how to do it in Python.
# After spending a few hours on this problem, it makes sense to do the models in both R and Python.

df.columns
y = df["Income"]

# Single covariates:
covariates = [] # Empty list
for column in df.columns: # Select each column in the dataframe
    covariates.append(column)
covariates.remove("Income")

# Build the individual datasets
df.columns
    # Build the intercept-only model
intercept = pd.DataFrame(1, index=np.arange(1,30137), columns=np.arange(1))
# Build the datasets via a for loop:
datasets = []
for column in covariates: # For each covariate
    datasets.append(sm.add_constant(df.loc[:,column])) # Add the intercept to each covariate
datasets


# Build the models using another for loop
results = []
for dataset in datasets:
    results.append(sm.MNLogit(y, dataset).fit()) # Fit a multinomial logistic regression model


# Show the model summaries; no, it's not iterable
results[0].summary()
results[1].summary()
results[2].summary()
results[3].summary()
results[4].summary()
results[5].summary()
results[6].summary()
results[7].summary()
results[8].summary()
results[9].summary()
results[10].summary()
results[11].summary()
results[12].summary()
results[13].summary()

# It looks like, from both R and Python, that we should drop the fnlWGT variable.
# Addendum: 18th May 2020
# However, you learned that sometimes confounding variables may affect your results,
# Thus, we will leave FnlWGT in the upcoming models.

"Removing unnecessary objects:"
del names
del null
del name
del colors
del columns
del to_drop
del scaler
del stats
del country
del occupation
del marital_dict
del gov_dict
del name
del Nominal
del plot
del missing_values
del grid
del pmax
del intercept
del covariates
del column


###############################################################################
###                     6. Train-Test Split                                 ###
###############################################################################

"Train-Test Split"

# Define X and Y variables

# Replace Class with your response variable
y = df["Income"]
x = df.drop(["Income"], axis = 1)  # Drop any unneeded variables

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 0.25,  # 75/25
    random_state = 123,
    stratify = y,
    shuffle=True)

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
                        {"n_estimators": [400],
                         "learning_rate": [0.001, 0.05, 0.10, 0.5]
                         }})

help(sklearn.svm.SVC())
HyperParams.update({"SVClassifier":
                        {"kernel": ["rbf"],
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
# It took me a like 2 days to figure out what to do after this step but the search
# was worth it
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
            print(f"Running GridSearchCV for {key}.")
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

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score', "range_score"]
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]


# After this, pass in a dictionary of models and a dictionary of hyperparameters to use

pipeline = EstimatorSelectionHelper(Models, HyperParams)
pipeline.fit(x_train, y_train,
             cv = 2, n_jobs = -2, refit = False, #Change cv to whatever you want
             scoring = "f1", verbose = 1)
results = pipeline.score_summary(sort_by='mean_score')
print(results)

###############################################################################
#                        9. "Feature Importance"                              #
###############################################################################

"This took quite a while to figure out the kinks"

Forest = RandomForestClassifier(n_estimators = 250,
                                random_state = 123,
                                verbose = 2,
                                n_jobs=10)

Forest.fit(x, y)


def imp_df(column_names, importances):
    df = pd.DataFrame({'feature': column_names,
                       'feature_importance': importances}) \
           .sort_values('feature_importance', ascending = False) \
           .reset_index(drop = True)
    return df

Importances = imp_df(x.columns, Forest.feature_importances_)

Importances

"Plotting Feature Importances"
# Source: https://github.com/erykml/medium_articles/blob/master/Machine%20Learning/feature_importance.ipynb


def var_imp_plot(Importances, Title):
    Importances.columns = ['feature', 'feature_importance']
    sns.barplot(x = "feature_importance", y = 'feature', data = Importances,) \
                .set_title(Title, fontsize = 20)
    plt.show()

var_imp_plot(Importances, Title = "Gini Feature Importances")


###############################################################################
#                       10.  Making Predictions                               #
###############################################################################

"I was thinking of ways to make predictions on all of the models for quite a while"
"Then I realized that would be pointless because I would only need to make predictions
"on the best model from the results."
"In the future, you'd have to read the results table and find the best model at the top
"along with its parameters to train."

# For this example, the best model is AdaBoost with:
# lr = 0.001,
# n_estimators = 400, since same as 500

help(AdaBoostClassifier)
BestModel = AdaBoostClassifier(n_estimators = 400,
                               learning_rate = 0.01,
                               random_state = 123,
                               algorithm = "SAMME.R")

BestModel.fit(x_train, y_train)
# Making Predictions

preds = BestModel.predict(x_test)
probs_multi = BestModel.predict_proba(x_test)
probs = probs_multi[:, 1]
#############################################################
#                       11. Model Performance               #
#############################################################
from pycm import ConfusionMatrix
# Yes, both Sci-kit learn and PYCM give the same confusion matrix, thankfully.

# Confusion Matrix and Extra Statistics:
y_test = LabelEncoder().fit_transform(y_test)
preds = LabelEncoder().fit_transform(preds)
cm = ConfusionMatrix(actual_vector = y_test,
                     predict_vector = preds)
print(cm)



#Plotting the ROC_AUC Curve:
skplot.metrics.plot_roc(y_test,probs_multi)
plt.show()
