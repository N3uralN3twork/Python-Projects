import os
abspath = os.path.abspath('C:/Users/MatthiasQ.MATTQ/Desktop/Python Projects/Sci-kit Learn')
os.chdir(abspath)
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

##############Example 1#######################

# This particular dataset is from R
df = pd.read_csv("Datasets/Tooth.csv")

#Retrieve the names of the columns for future use.
df.columns

# Building the 2 way ANOVA model
model = ols('len ~ C(supp)*C(dose)', df).fit()

# ANOVA table
model.summary()

# Coefficients table
res = sm.stats.anova_lm(model, typ=2)
res

# Post-hoc Comparison Testing:
# We really shouldn't be comparing the individual means since our
# interaction term is significant, but oh well.

mcomp = sm.stats.multicomp.MultiComparison(df["len"], df["supp"])
mcomp = mcomp.tukeyhsd()
print(mcomp)


##############Example 2#######################
sodium = pd.read_csv("Datasets/SodiumIntake.csv")

sodium.info()
sodium.columns

model2 = ols("Sodium ~ Instructor+Supplement",
             data=sodium).fit()

model2.summary()