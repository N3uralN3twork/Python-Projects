"""Repeated Measures ANOVA:
Goal: Learn about repeated measures analysis of variance and how to apply it
Author: Matthias Quinn
Sources: https://statistics.laerd.com/statistical-guides/repeated-measures-anova-statistical-guide.php
         https://www.marsja.se/repeated-measures-anova-in-python-using-statsmodels/
         https://www.marsja.se/four-ways-to-conduct-one-way-anovas-using-python/"""

"Notes:"
#Like a 1-Way ANOVA, but you test related groups that are dependent and correlated
#One of the factors has to be repeatedly measured
#Questions to ask yourself:
    #Is there a direct relationship between each pair of observations?
    #Do all observations have the same number of data points?
#It looks like we treat the subject as one of the factors to find variation within?
#Helps expand variability from within-group component of total variability

#Null Hypothesis: mu1 = mu2 = ... mu3
#Alternative: at least two means are significantly different
#F statistic = MS(conditions) / MS(error)


"1. One-Way ANOVA:"
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM

df = pd.read_csv("C:/Users/MatthiasQ.MATTQ/Desktop/Python Projects/Statistics/OneWayAnova Data.csv")
#Goal is to figure out whether the noise condition affects response time

df.head(3)

#Set-up the ANOVA model like in R:
model = ols("ResponseTime ~ Condition",
            data = df).fit()

#Conduct the ANOVA
aov_table = sm.stats.anova_lm(model, typ = 2)
aov_table

#Pairwise comparisons:
pairwise = model.t_test_pairwise("Condition")
pairwise.result_frame
#Interpret:
#As is evident, it appears that the mean response time is different for at least one of the noise groups.
#In addition, it appears that the mean response time between quite and noisy conditions


"2. Repeated Measures ANOVA:"
RMANOVA = AnovaRM(data = df, depvar = "ResponseTime",
                  subject = "Subject", within = ["Condition"]).fit()

print(RMANOVA)
#So it doesn't look like we are interested in the Subjects themselves,
#Seems like they are treated as blocks?
#That was surprisingly simple





