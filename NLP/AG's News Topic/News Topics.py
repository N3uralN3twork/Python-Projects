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

train.columns = ["Title", "Topic"]  # Just like R
test.columns = ["Title", "Topic"]

train.head()

train["Topic"].value_counts()
test["Topic"].value_counts()
# So there are 30,000 articles from each topic in the training set
# 1,900 for each topic in the test set



"Cleaning the Data:"

# From a previous project:
def Clean_DF_Text(text):
    import re
    from nltk.corpus import stopwords
    import unicodedata
    text = re.sub('<[^<]+?>', '', text) #Remove HTML
    text = re.sub("[^a-zA-Z]", " ", text) #Remove punctuation
    text = re.sub('[0-9]+', " ", text) #Remove numbers
    text = text.strip()     #Remove whitespaces
    text = re.sub(" +", " ", text) #Remove extra spaces
    text = text.lower()     #Lowercase text
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore") #Remove accents
    stopwords = stopwords.words('english')
    stopwords.remove("not") # Important custom words to keep
    words = text.split()
    clean_words = [word for word in words if word not in stopwords] # List comprehension
    text = ' '.join(clean_words)
    return text

# Apply the cleaning function to our articles:

train["Clean"] = train["Title"].apply(Clean_DF_Text)
test["Clean"] = test["Title"].apply(Clean_DF_Text)

train.head()
test.head()









