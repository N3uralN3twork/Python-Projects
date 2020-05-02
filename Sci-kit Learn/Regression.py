import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression, LogisticRegression
import statsmodels.api as sm

df = datasets.load_diabetes()
df
X = df.data
y = df.target

intercept = sm.add_constant(X)
lm = sm.OLS(y, intercept).fit()
lm.summary()

reg = LinearRegression(X, y, n_jobs = 2).fit




list = [1,2,3,4]
np.transpose(list)

print(list)


