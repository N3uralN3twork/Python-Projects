"""
Title: AG News Topics
Author: Matt Quinn
Date: 25th May 2020
Goal: Predict which topic a particular new's article belongs to
Sources:
    http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html
    https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf
    http://www.kecl.ntt.co.jp/uebetsu/index.html
Dataset:
    https://data.wluper.com/
"""


#########################################################
###           1. Set the working directory            ###
#########################################################
import os
abspath = os.path.abspath("C:/Users/miqui/OneDrive/Python-Projects/NLP/AG's News Topic")
os.chdir(abspath)
#########################################################
###           2. Import Data and Libraries            ###
########################################################
import pandas as pd

# Import the training and testing datasets
train = pd.read_csv("AG__FULL.csv", header=None)
test = pd.read_csv("AG__TEST.csv", header=None)


train.head()
train.tail()
#########################################################
###           3. Data Cleaning                        ###
#########################################################

# Create two new variables: Title and Topic

train.columns = ["Title", "Topic"] # Just like R
test.columns = ["Title", "Topic"]
















