"""
Title: AG News Topics
Author: Matt Quinn
Date: 25th May 2020
Goal: Predict which topic a particular new's article belongs to
Sources:
    http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html
    https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf
    http://www.kecl.ntt.co.jp/uebetsu/index.html
    https://developers.google.com/machine-learning/guides/text-classification
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
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text

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

"Gather dataset metrics:"

# Number of samples: 120,000
# Number of classes: 4
# Number of samples per class: 30,000
# Median number of words per sample: 30

def get_num_words_per_sample(sample_texts):
    """Returns the median number of words per sample given corpus.
    # Arguments
        sample_texts: list, sample texts.
    # Returns
        int, median number of words per sample.
    """
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)
get_num_words_per_sample(train["Title"])

def plot_sample_length_distribution(sample_texts):
    """
    Plots the sample length distribution.
    # Arguments
        samples_texts: list, sample texts.
    """
    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()

plot_sample_length_distribution(train["Title"])

From our experiments, we have observed that the ratio of “number of samples” (S) to “number of words per sample” (W) correlates with which model performs well.

"""
From our experiments, we have observed that the ratio of “number of samples” (S) to “number of words per sample” (W) correlates with which model performs well.
When the value for this ratio is small (<1500), small multi-layer perceptrons that take n-grams as input (which we'll call Option A) perform better or at least as well as sequence models.
MLPs are simple to define and understand, and they take much less compute time than sequence models. 
When the value for this ratio is large (>= 1500), use a sequence model (Option B).
In the steps that follow, you can skip to the relevant subsections (labeled A or B) for the model type you chose based on the samples/words-per-sample ratio.
"""

# Vectorization parameters
# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

# Limit on the length of text sequences. Sequences longer than this
# will be truncated.
MAX_SEQUENCE_LENGTH = 500

def sequence_vectorize(train_texts, val_texts):
    """Vectorizes texts as sequence vectors.

    1 text = 1 sequence vector with fixed length.

    # Arguments
        train_texts: list, training text strings.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val, word_index: vectorized training and validation
            texts and word index dictionary.
    """
    # Create vocabulary with training texts.
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(train_texts)

    # Vectorize training and validation texts.
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)

    # Get max sequence length.
    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    # Fix sequence length to max value. Sequences shorter than the length are
    # padded in the beginning and sequences longer are truncated
    # at the beginning.
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)
    return x_train, x_val, tokenizer.word_index

x_train, x_val, word_index = sequence_vectorize(train["Clean"], test["Clean"])

word_index

