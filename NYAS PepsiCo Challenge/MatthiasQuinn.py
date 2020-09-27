"""
Goal: Predict the assessment score (Supervised Regression Task)
Author: Matt Quinn
Date: 25th September 2020
Sources:
    https://www.nyas.org/challenges/pepsico-challenge
    https://stackoverflow.com/questions/8703017/remove-sub-string-by-using-python
    https://stackoverflow.com/questions/1249388/removing-all-non-numeric-characters-from-string-in-python
    https://stackoverflow.com/questions/34214139/python-keep-only-letters-in-string
    https://stackoverflow.com/questions/25646200/python-convert-timedelta-to-int-in-a-dataframe
    http://www.davidsbatista.net/blog/2018/02/23/model_optimization/
    https://github.com/davidsbatista/machine-learning-notebooks/blob/master/hyperparameter-across-models.ipynb
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    https://stackoverflow.com/questions/35552874/get-first-letter-of-a-string-from-column
    https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
    https://stackoverflow.com/questions/11190735/python-matplotlib-superimpose-scatter-plots
    https://www.tensorflow.org/tutorials/keras/regression
    https://stackoverflow.com/questions/55908188/this-model-has-not-yet-been-built-error-on-model-summary
    https://www.tensorflow.org/api_docs/python/tf/keras/metrics
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
    https://stackoverflow.com/questions/23554872/why-does-pycharm-propose-to-change-method-to-static
    https://jmlb.github.io/ml/2017/03/20/CoeffDetermination_CustomMetric4Keras/
    https://github.com/felixTY/FingerNet/issues/1
    https://keras.io/guides/sequential_model/#when-to-use-a-sequential-model
    https://stats.stackexchange.com/questions/240305/where-should-i-place-dropout-layers-in-a-neural-network
    https://stats.stackexchange.com/questions/299292/dropout-makes-performance-worse
    https://arxiv.org/abs/1502.03167

Steps in My Process:
    1. Set the Working Directory
    2. Import the necessary data and libraries:
        a. Set "*" observations equal to missing
        b. You must have a tensorflow version >=2.2
            I. pip install --upgrade tensorflow
    3. Merge the 3 data files into 1 file via left_joins
        a. Remove "Year" from each Site ID
        b. Use left-joins to merge the three files
    4. Exploratory data analysis:
        a. Frequency tables
        b. Histograms
    5. Feature Engineering:
        a. Year
        b. Site
        c. Month
        d. Weekday
        e. Time to Harvest
    6. Pre-process Features:
        a. One-hot Encode
        b. Scaling continuous variables
        c. Missing value Imputation (Iterative)
    7. Train-Test Split
        a. 80-20 rule
        b. do not use the "stratify=" option
    8. Define Hyper-parameter Search Space:
        a. done via dictionaries
        b. follow the first example and read the documentation for each model
    9. Run our Models:
        a. Huge shout-out to David Batista for the starter code
    10. Tune Hyper-parameters for the best model:
        a. look at the previous steps' results
    11. Feature Importance:
        a. GINI index?
    12. Model Validation:
        a. various metrics (R^2)
    13. Making Predictions
    14. Deep Learning:
        a. use a normalization layer since vars have different ranges

"""
###############################################################################
###                     1.  Define Working Directory                        ###
###############################################################################
import os
abspath = os.path.abspath("C:/Users/miqui/OneDrive/Python-Projects/NYAS PepsiCo Challenge")
os.chdir(abspath)
os.listdir()
###############################################################################
###                    2. Import Libraries and Data                       ###
###############################################################################
# Machine learning stuff
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing as preprocess
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Deep learning stuff
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.models import Sequential
import tensorflow_addons as tfa
import tensorflow.keras.metrics as metrics
from tensorflow.keras.layers.experimental import preprocessing

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 200)

# Set the variable names for each dataset:
cropGrain_names = ["ID", "Site ID", "Growth Stage", "Variety", "Date", "Assessment Type", "Assessment Score", "P"]
weather_names = ["Site ID", "Date", "Weather A", "Weather B", "Weather C", "Weather D", "Weather E", "Weather F"]
sites_names = ["Site ID", "Latitude", "Elevation (m)", "Sowing Date", "Harvest Date", "Soil A", "Soil B", "Amount Fertilizer"]


# Import the individual datasets
cropGrain = pd.read_excel("MatthiasQuinn.xlsx", names=cropGrain_names, sheet_name="Crop and Grain Data",
                          na_values=["*"])
weather = pd.read_excel("MatthiasQuinn.xlsx", names=weather_names, sheet_name="Weather Data")
sites = pd.read_excel("MatthiasQuinn.xlsx", names=sites_names, sheet_name="Site Data")

del cropGrain_names
del weather_names
del sites_names
del abspath

cropGrain.info()
cropGrain.describe().transpose()
# Interestingly, there are 14 missing responses?

###############################################################################
###                      3. Data Concatenation                              ###
###############################################################################
def fixSiteIDs(SiteID: str):
    one = re.sub("Year", "", SiteID)
    result = re.sub("  ", " ", one)
    return result


# Test our function:
fixSiteIDs("Site E Year 2015")

cropGrain["Site ID"] = cropGrain["Site ID"].apply(fixSiteIDs)

# Now let's move onto merging the three files into 1:
# First merge the main and sites data by "Site ID":

one = pd.merge(left=cropGrain, right=sites,
               how="left", on="Site ID")

df = pd.merge(left=one, right=weather,
              how="left", on=["Site ID", "Date"])

# Clear up some memory:
del cropGrain
del sites
del weather
del one

# Get some overhead information on our new dataset:
df.info()
###############################################################################
###                      4. Exploratory Data Analysis                       ###
###############################################################################
"Let's start with some frequency tables:"

Categorical = list(df.select_dtypes(include=['object', 'category']).columns)

for name in Categorical:
    print(name, ":")
    print(df[name].value_counts(), "\n")

# Create histograms for each variable:
df.hist()
plt.show()

###############################################################################
###                      5. Feature Engineering                             ###
###############################################################################

# We can extract the Year from each of the Site IDs, via regex:
df["Year"] = df["Site ID"].apply(lambda x: re.sub("[^0-9]", "", x))
df["Year"] = df["Year"].astype("int")  # Convert the new strings to integers
df["Year"].value_counts()

# Extract the Site from each Site ID:
df["Site"] = df["Site ID"].apply(lambda x: re.sub("[^a-zA-Z]+", "", x))
df["Site"].value_counts()

# To extract the month:
df['Month'] = pd.DatetimeIndex(data=df["Date"]).month

# To extract the weekday:
df["Weekday"] = pd.DatetimeIndex(data=df["Date"]).dayofweek

# How much time until harvest?:
df["HarvestTime"] = df["Harvest Date"] - df["Sowing Date"]
df["HarvestTime"] = df["HarvestTime"].dt.days  # Convert to days

# What does the Assessment type start with?:
df["AssessmentPrefix"] = df["Assessment Type"].str[0]
df["AssessmentPrefix"].value_counts()

# Is the Assessment type equal to the form "C-{}"?:
df["IsTypeC"] = np.where(df["AssessmentPrefix"] == "C", 1, 0)

# Retain only the necessary columns and reorder them too:
keep = ["Growth Stage", "Variety", "Assessment Type", "Elevation (m)", "Soil A", "Soil B",
        "Amount Fertilizer", "Weather A", "Weather B", "Weather C", "Weather D", "Weather E", "Weather F",
        "Year", "Site", "Weekday", "HarvestTime", "AssessmentPrefix", "IsTypeC", "Assessment Score"]
clean_df = df[keep]

clean_df.columns

# Create a heat map of the correlations:
def correlation_heatmap(df, method="pearson"):
    _, ax = plt.subplots(figsize=(14, 14))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    _ = sns.heatmap(
        df.corr(method=method),
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


correlation_heatmap(clean_df, method="pearson")

# Inter-correlations look good

###############################################################################
###                      6. Preprocessing Features                          ###
###############################################################################

Continuous = list(clean_df.select_dtypes(include=['int64', 'float64']).columns)

# Check for duplicate rows:
clean_df[clean_df.duplicated()]  # 0 = good

"ENCODING:"
# Encode the categorical variables into numeric:
Nominal = list(clean_df.select_dtypes(include=['object', 'category']).columns)
clean_df = pd.get_dummies(clean_df, drop_first=True)  # You'll have to check the base_case yourself

"Scaling Continuous Variables:"
Continuous = list(clean_df.select_dtypes(include=['int64', 'float64']).columns)
Continuous.remove("Assessment Score")
clean_df[Continuous].describe()

# Transform the data with this:
scaler = preprocess.RobustScaler()  # Define the scaling method
clean_df[Continuous] = scaler.fit_transform(clean_df[Continuous])  # Don't scale the target


# Drop any missing values:
# Will mess up your analyses later on if you don't drop or impute missing values!
# test = clean_df.dropna(axis=1)

np.isnan(clean_df).sum()  # Check the missingness of each variable

def Impute(df):
    names = df.columns
    imputer = IterativeImputer(max_iter=10, random_state=123, )
    clean_df = imputer.fit_transform(X=df)
    clean_df = pd.DataFrame(clean_df)
    clean_df.columns = names
    return clean_df

clean_df = Impute(clean_df)


"Removing unnecessary objects:"
del scaler
del Nominal
del Continuous
del keep
del name
del Nominal
del Categorical
del Continuous


###############################################################################
###                     7. Train-Test Split                                 ###
###############################################################################


# Define X and Y variables
clean_df.columns

# Replace Class with your response variable
y = clean_df["Assessment Score"]
x = clean_df.drop(["Assessment Score"], axis=1)  # Drop any unneeded variables

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.20,  # 80/20 split
    random_state=123,  # Set a random seed for reproducibility
    shuffle=True)

del df
###############################################################################
###                      8. Choosing Parameter Space                        ###
###############################################################################

"Now let's say I wanted to run multiple models, like KNN and AdaBoost"
"The full tutorial accomplishes this"
"Don't forget: The parameters must be in square brackets"

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


# Add models as you wish
# "Model Name": Model Method()
Models = {
    "Random Forest": RandomForestRegressor(),
    "Ada Boost": AdaBoostRegressor(),
    "Linear Regression": LinearRegression(),
    "SVRegression": SVR(),
    "KNN": KNeighborsRegressor(),
    "MLP": MLPRegressor()
    }

##################################################
HyperParams = {}  # Start with an empty dictionary

# Random Forest parameters:
help(sklearn.ensemble.RandomForestRegressor())
HyperParams.update({"Random Forest":
                        {"criterion": ["mse"],
                         "min_samples_split": [2, 3, 5],
                         "max_features": ["auto", "sqrt"],
                         "oob_score": [True],
                         "verbose": [1],
                         "warm_start": [True, False]
                         }})

# ADA Boost parameters
help(sklearn.ensemble.AdaBoostRegressor())
HyperParams.update({"Ada Boost":
                        {"n_estimators": [400],
                         "learning_rate": [0.001, 0.05, 0.10, 0.5]
                         }})

# Linear Regression parameters:
help(sklearn.linear_model.LinearRegression())
HyperParams.update({"Linear Regression":
                        {"fit_intercept": [True]}})

# Support Vector Regression parameters:
help(sklearn.svm.SVR())
HyperParams.update({"SVRegression":
                        {"kernel": ["rbf"],
                         "verbose": [True],
                         "gamma": [0.001, 0.0001]
                         }})

# k-Nearest Neighbors parameters:
help(sklearn.neighbors.KNeighborsRegressor())
HyperParams.update({"KNN":
                        {"n_neighbors": [3],
                         "algorithm": ["auto"],
                         "n_jobs": [-2]
                         }})

# Neural Network parameters
help(sklearn.neural_network.MLPRegressor())
HyperParams.update({"MLP":
                        {"activation": ["tanh"],
                         "solver": ["adam"],
                         "verbose": [True]
                         }})

HyperParams

###############################################################################
###                      9. Running Our Models                              ###
###############################################################################


class EstimatorSelectionHelper:

    def __init__(self, models, params):
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    @staticmethod
    def check_rows(X, y):  # My addendum
        if X.shape[0] == y.shape[0]:
            return True
        else:
            return False

    def fit(self, X, y, **grid_kwargs):
        for key in self.keys:
            print(f"Running GridSearchCV for {key}.")
            model = self.models[key]
            params = self.params[key]
            grid_search = GridSearchCV(model, params, **grid_kwargs)
            grid_search.fit(X, y)
            self.grid_searches[key] = grid_search
        print('DONE.')

    def score_summary(self, sort_by='mean_test_score'):
        frames = []
        for name, grid_search in self.grid_searches.items():
            frame = pd.DataFrame(grid_search.cv_results_)
            frame = frame.filter(regex='^(?!.*param_).*$')
            frame['estimator'] = len(frame) * [name]
            frames.append(frame)

        final = pd.concat(frames)
        final = final.sort_values([sort_by], ascending=False)
        final = final.reset_index()
        final = final.drop(['rank_test_score', 'index'], axis=1)

        columns = final.columns.tolist()
        columns.remove('estimator')
        columns = ['estimator'] + columns  # Reorder columns
        final = final[columns]
        return final


# Running 21 different models:
pipeline = EstimatorSelectionHelper(Models, HyperParams)
pipeline.check_rows(x_train, y_train)  # Good if True
pipeline.fit(X=x_train, y=y_train,
             scoring="r2", n_jobs=-2, refit=False, cv=2, verbose=2)  # Use R^2 (unadjusted) as the metric for evaluation
results = pipeline.score_summary()  # Print the results to a dataframe

# Free up extra memory:
del Categorical
del AdaBoostRegressor
del KNeighborsRegressor
del LinearRegression
del MLPRegressor
del RandomForestRegressor
del SVR
###############################################################################
###                      10. Hyper-parameter Tuning for RF                   ###
###############################################################################

"I was thinking of ways to make predictions on all of the models for quite a while"
"Then I realized that would be pointless because I would only need to make predictions"
"on the best model from the results."
"In the future, you'd have to read the results table and find the best model at the top"
"along with its parameters to train."

# For this example, the best model is the Random Forest (no surprise) with:
# criterion="mse", max_features="auto", min_samples_split=5, oob_score=True, verbose=1

help(RandomForestRegressor)

BestModel = RandomForestRegressor(criterion="mse",
                                  max_features="auto",
                                  min_samples_split=3,
                                  oob_score=True,
                                  verbose=1,
                                  warm_start=True)
# Run the model:
BestModel.fit(X=x_train, y=y_train)

# Check what's available with our newest model:
dir(BestModel)

###############################################################################
###                      11. Feature Importances                            ###
###############################################################################


def imp_df(column_names, importances):
    df = pd.DataFrame({'feature': column_names,
                       'feature_importance': importances}) \
        .sort_values('feature_importance', ascending=False) \
        .reset_index(drop=True)
    return df


Importances = imp_df(x.columns, BestModel.feature_importances_)

Importances


def var_imp_plot(Importances, Title):
    Importances.columns = ['feature', 'feature_importance']
    sns.barplot(x="feature_importance", y='feature', data=Importances, ) \
        .set_title(Title, fontsize=20)
    plt.show()


var_imp_plot(Importances, Title="Gini Feature Importance")

###############################################################################
###                      12. Model Validation                               ###
###############################################################################
BestModel.score(x_test, y_test)  # Test on the unseen data
# A damn good score for sure compared to the baseline (8.79% with just GrowthStage)

###############################################################################
###                      13. Making Predictions                             ###
###############################################################################
# Making Predictions:
preds = BestModel.predict(x_test)  # Make predictions on the test set


full_preds = BestModel.predict(X=x)  # Make predictions on the full dataset!

# Does the original distribution of the scores align with the predicted scores?:

# How do I overlay the two distributions?
# Nevermind
sns.distplot(df["Assessment Score"], color="green")
sns.distplot(full_preds, color="red")
plt.title("Original vs. Predicted")
plt.show()


###############################################################################
###                      14. Deep Learning                                  ###
###############################################################################

"""When the network is small relative to the dataset, regularization is usually unnecessary.
If the model capacity is already low, lowering it further by adding regularization will hurt performance.
I noticed most of your networks were relatively small and shallow."""

"""
BATCH NORMALIZATION:
    Apparently, batch normalization just computes a Z-score:" \
        Given by this information: https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
    We will normalize each scalar feature independently, by making it have the mean of 0 and the variance of 1
    Uses population statistics, not mini-batch
    Apparently gets to convergence with fewer epochs?
"""

"""
Adadelta optimization is a stochastic gradient descent method that is based on
  adaptive learning rate per dimension to address two drawbacks:

  - The continual decay of learning rates throughout training
  - The need for a manually selected global learning rate
"""

# Add an experimental pre-processing layer:
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(x_train))

# Adding a custom metric (R^2 is not built-in to tf for some reason):
def r_square(y_true, y_pred):
    from keras import backend as K
    SSE = K.sum(K.square(y_true-y_pred))
    SST = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SSE/(SST + K.epsilon())


# Add a callback to run during training:
# If the loss doesn't improve after 3 epochs, stop.
callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)

################################
# Build your DNN:
# Fully connected since each layer feeds the next layer
model = Sequential(name="DNN")
model.add(normalizer)
model.add(layer=keras.layers.Dense(64, activation="relu", name="layer1"))
model.add(layer=keras.layers.BatchNormalization(axis=1, center=True, scale=True, name="BatchNorm"))
model.add(layer=keras.layers.Dense(64, activation="relu", name="layer2"))
model.add(layer=keras.layers.Dense(32, activation="relu", name="layer3"))
model.add(layer=keras.layers.Dense(16, activation="relu", name="layer4"))
model.add(layer=keras.layers.Dense(1, name="OutputLayer"))

model.compile(loss='mean_absolute_error',  # Better since we have outliers
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=[r_square, metrics.MeanAbsoluteError(name="MAE"), metrics.RootMeanSquaredError(name="RMSE")])

model.build(input_shape=[8250, 49])

# Get a snapshot of the model built:
model.summary()  # DON'T SKIP ME!!!!

# Train the model:
history = model.fit(
                x=np.array(x_train), y=np.array(y_train),
                validation_split=0.2,
                verbose=1,
                epochs=50,  # Notice that r-square is pretty low for the first 5 epochs
                callbacks=[callback])  # Early stopping

tf_preds = model.predict(x=x)

r_square(y_true=tf.convert_to_tensor(y, np.float32), y_pred=full_preds)  # 98.73411%
r_square(y_true=tf.convert_to_tensor(y, np.float32), y_pred=tf_preds)


dir(history)


# summarize history for R^2:
plt.plot(history.history['r_square'])
plt.plot(history.history['val_r_square'])
plt.title('Model R^2 over Epochs')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()















