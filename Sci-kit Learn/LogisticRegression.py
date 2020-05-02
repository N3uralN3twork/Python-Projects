"""Logistic Regression using Python"""

#Used to predicted categorical outcomes
#Main Libraries: numpy, pandas, statsmodels, pylab

import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

"The dataset:"
#College admission dataset
#Predictor variables:
    #gpa
    #gre score
    #rank of the applicant's undergraduate college
#Respone variables:
    #Admit, whether the candidate made it into the college

df = pd.read_csv("https://stats.idre.ucla.edu/stat/data/binary.csv")
df.head(3)

df.columns = ["admit", "gre", "gpa", "prestige"]

#Frequency table of prestige and admittance
pd.crosstab(df['admit'], df['prestige'], rownames = ['admit'])
#Seems that the lower the prestige #, the higher you're chance of being admitted is.

df.hist()
#Clearly see that gpa is left-skewed

"Dummy variables:"
dummy_ranks = pd.get_dummies(df["prestige"], prefix = "prestige")
dummy_ranks.head(3)

dummy_ranks = dummy_ranks.drop(["prestige_1"], axis=1)
dummy_ranks
cols_to_keep = ['admit', 'gre', 'gpa']
data = df[cols_to_keep].join(dummy_ranks)

data.head(3)

"Constant term:"
#statsmodels require that a constant term is defined explicitly
data["intercept"] = 1.0


x = data.drop(["admit"], axis = 1)
y = data["admit"]


"Logistic Regression with Logit link:"
#We'll be treating "prestige_1" as our dummy baseline level

logit = sm.Logit(y, x)

result = logit.fit()

print(result.summary())

#Why can't I get this type of output in R.



"Odds Ratio:"
#Tells you how a 1 unit increase/decrease in a variable affects the odds of being admitted.
print(np.exp(result.params))

#We can expect the odds of being admitted to decrease by about 50% if the prestige of the
#undergraduate school is 2.
#For every one unit increase in a student's GRE score, the odds of being admitted increase by about a factor of 1


#For a 95% confidence interval of the coefficients.
print(np.exp(result.conf_int()))

