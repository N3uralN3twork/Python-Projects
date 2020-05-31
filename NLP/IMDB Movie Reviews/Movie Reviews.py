"""
Title: IMDB Movie Reviews
Author: Matt Quinn
Date: 29th May 2020
Goal: Use BERT to predict the sentiment of a movie review
Sources:

Dataset:
    https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
    https://www.kaggle.com/atulanandjha/imdb-50k-movie-reviews-test-your-bert
Notes:
    To get the current full dataset, I just rbinded the two above data sources.
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
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence, text
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, GlobalMaxPool1D
from tensorflow.keras.models import Model, Sequential
sns.set()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

train = pd.read_csv("IMDB Reviews.csv", header=0)
test = pd.read_csv("test.csv", header=0)






