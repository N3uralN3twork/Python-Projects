# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:08:15 2019

@author: MatthiasQ
"""

"Linear Regression"
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

#1. Import your data:
Air = pd.read_excel("C:/Users/MatthiasQ.MATTQ/Downloads/AirQualityUCI/AirQualityUCI.xlsx")

#2. To convert a file to a dataframe
Air = pd.DataFrame(Air) #Pretty much just like R's

#3. Basic pandas functions
Air.head() #Check out the beginning of your data
Air.dtypes #Check out the data types like R's str()
Air.values #Check out just the raw data
Air.info() #Get high-level summary of dataset
Air.describe # Just like summary() in R
#Air.dropna #Drops and missing data
Air[''].mean() #To find the function of a specific 

#To rename a specific column

Air.rename(columns={"PT08.S1(CO)":"Tin"}, inplace=True)
Air.info()

#To rename all of the columns
"Format":
    "Old":"New",
Air.rename(columns = {'Date':'Date',
                      'Time':'Time',
                      'CO(GT)':'Avg.CO',
                      'Tin':'Tin',
                      'NMHC(GT)':'NMHC',
                      'C6H6(GT)':'Benzene',
                      'PT08.S2(NMHC)':'Titania',
                      'NOx(GT)':'Avg.NOx',
                      'PT08.S3(NOx)':'Tungsten_Oxide',
                      'NO2(GT)':'Avg.NO2',
                      'PT08.S4(NO2)':'Tungsten_Oxide2',
                      'PT08.S5(O3)':'Indium_Oxide',
                      'T':'Temperature',
                      'RH':"Relative_Humidity",
                      "AH":"Absolute_Humidity"},
                      inplace=True) #Inplace replaces the old names
Air.info()
#3. To create a dataframe
Months = {'Month': ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"],
          'Count': [31,28,31,30,31,30,31,31,30,31,30,31]}
Months = pd.DataFrame(Months)
Months
#Just like that

Months['Count'].describe() #Describe a specific variable

#4. Plotting your data
sns.pairplot(Air) #all variables plot
sns.distplot(Air['points']) #Distribution plot of specific variable

#5. Checking correlations
Air.corr()
sns.heatmap(Air.corr()) #Heatmap of correlations

#6 Linear Regression
"Selecting your variables:"

X = Air[["Avg.CO", "Tin", "NMHC", "Benzene", "Titania", "Avg.NOx", "Tungsten_Oxide",
         "Avg.NO2", "Tungsten_Oxide2", "Temperature", "Relative_Humidity"]]
Y = Air[["Indium_Oxide"]]
X = sm.add_constant(X)
"Training and test split:"
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size = 0.3, #70% of data is training set
                                                    random_state = 101)
"Creating and Training the model:"
lm = LinearRegression(fit_intercept= True,
                      normalize = True)
lm.fit(X = X_train, y = Y_train)

"or"

model = sm.OLS(Y, X).fit()

"Predicting from the model:"
predictions = lm.predict(X_test)
predictions2 = model.predict(X)
plt.scatter(x = Y_test, y = predictions)

"Model Summary:"
model.summary()
