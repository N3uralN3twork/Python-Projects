"""
Author: Matt Quinn
Date: 25th May 2020
Goal: Predict which tweets are about real disasters and which are not
"""

#########################################################
###           1. Set the working directory            ###
#########################################################
import os
abspath = os.path.abspath("C:/Users/miqui/OneDrive/Python-Projects/NLP")
os.chdir(abspath)
#########################################################
###           2. Import Data and Libraries            ###
#########################################################
import pandas as pd


train = pd.read_csv("Tweets/Tweets_train.csv", header=0)
test = pd.read_csv("Tweets/Tweets_test.csv", header=0)

#########################################################
###           3. Data Cleaning                        ###
#########################################################

"Drop the first, second, and third variables"
"I only want to make predictions based on the tweet's contents"
"There is quite a bit of preprocessing that needs to be done to clean up these tweets"

train.columns
test.columns

train = train.drop(["id", "keyword", "location"], axis=1)
test = test.drop(["id", "keyword", "location"], axis=1)


train.head(3)






