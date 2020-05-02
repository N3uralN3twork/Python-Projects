"The goal of this project is to further develop our dimension reduction techniques."
"Source: https://www.kaggle.com/pavansanagapati/simple-tutorial-dimensionality-reduction-methods"

##############################################################
###             1. Import the necessary libraries          ###
##############################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

##############################################################
###             2. Import the dataset(s)                   ###
##############################################################
train = pd.read_csv("application_train.csv")
train.shape
#307,511 rows and 122 columns (hoo-boy)
train.describe



###############################################################
###             3. Dimension Reduction                      ###
###############################################################
#Check the % of missing values for each variable
missing_values = train.isnull().sum()/len(train)*100
missing_values.sort_values(ascending = False)

#This is a custom function I made based on someone else's work
def high_missing_filter(df, missingPercent):
    variables = df.columns
    variable = []
    for i in range (0, len(train.columns)):
        if missing_values[i] <= missingPercent: #Make sure you keep variables with X% or less
            variable.append(variables[i])
    df = df.filter(items = variable) #Keep only rows with missing % under threshold
    return df

df = high_missing_filter(df = train, missingPercent = 50)
df.shape
columns = train.columns
for col in columns:
    train[col].fillna(train[col].mode()[0], inplace = True)
#So we dropped 41 variables just with this filtering method.

df = df.drop(["TARGET", "SK_ID_CURR", "DAYS_ID_PUBLISH"], axis=1)
rf = RandomForestRegressor(random_state = 123, max_depth = 10,
                            verbose = 1)
df = pd.get_dummies(df)

missing_values = df.isnull().sum()/len(df)*100
missing_values.sort_values(ascending = False)


rf.fit(df, train.TARGET)

features = df.columns
importances = rf.feature_importances_
indices = np.argsort(importances[0:20]) #Top 20 features
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

feature = SelectFromModel(rf)
Fit = feature.fit_transform(df, train.TARGET)
Fit.shape
Fit.columns


"t-Distributed Stochastic Neighbor Embedding (t-SNE):"
#non-linear
#looks to find directions that maximize information
#t-distribution of similarity of points is formed


from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4).fit()