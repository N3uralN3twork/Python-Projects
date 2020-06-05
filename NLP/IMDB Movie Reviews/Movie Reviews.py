"""
Title: IMDB Movie Reviews
Author: Matt Quinn
Date: 29th May 2020
Goal: Predict the sentiment of a movie review
Sources:

Dataset:
    https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
    https://www.kaggle.com/atulanandjha/imdb-50k-movie-reviews-test-your-bert
Notes:
    To get the current full dataset, I just rbinded the two above data sources.
    ELMO is a rather large model (>350 MB)
"""

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# Tokenize Sentence

text = "Who was Jim Henson? He was a puppeteer."
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)

# That was stupid. Don't need a huge model to just tokenize a text for me when I have NLTK.


#########################################################
###           1. Set the working directory            ###
#########################################################
import os
abspath = os.path.abspath("C:/Users/miqui/OneDrive/Python-Projects/NLP/IMDB Movie Reviews")
os.chdir(abspath)
#########################################################
###           2. Import Data and Libraries            ###
########################################################
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub

sns.set()
pd.set_option("display.max_colwidth", 200)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

train = pd.read_csv("IMDB Reviews.csv", header=0)
test = pd.read_csv("test.csv", header=0)

train.head()
test.head()

# Import the ELMO model

elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)

#########################################################
###           3. Data Cleaning                        ###
#########################################################

train.columns
test.columns

# From a previous project
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

train["Clean"] = train["Review"].apply(Clean_DF_Text)
test["Clean"] = test["Review"].apply(Clean_DF_Text)

train.head()

















