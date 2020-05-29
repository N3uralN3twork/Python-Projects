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
    https://pbpython.com/categorical-encoding.html
    https://www.tensorflow.org/api_docs/python/tf/keras/Model
Dataset:
    https://data.wluper.com/
Notes:
    It should be noted that the accuracy of the model does not change much (about 1%)
    whether you input cleaned text data or the original text.
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
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing import sequence, text
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, GlobalMaxPool1D
from tensorflow.keras.models import Model, Sequential
sns.set()

# Import the training and testing datasets
train = pd.read_csv("AG__FULL.csv", header=None)
test = pd.read_csv("AG__TEST.csv", header=None)


train.head()
train.tail()
train.info()
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

# Rearrange columns:

train = train[["Title", "Clean", "Topic"]]
test = test[["Title", "Clean", "Topic"]]


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
    lengths = [len(i) for i in sample_texts]
    plt.hist(lengths, 50)
    plt.xlabel('Length of the title')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()

plot_sample_length_distribution(train["Title"])


"""
From our experiments, we have observed that the ratio of “number of samples” (S) to “number of words per sample” (W) correlates with which model performs well.
When the value for this ratio is small (<1500), small multi-layer perceptrons that take n-grams as input (which we'll call Option A) perform better or at least as well as sequence models.
MLPs are simple to define and understand, and they take much less compute time than sequence models. 
When the value for this ratio is large (>= 1500), use a sequence model (Option B).
"""

"Gather dataset metrics:"

# Number of samples: 120,000
# Number of classes: 4
# Number of samples per class: 30,000
# Median number of words per sample: 30

120000/30
"Thus, a sequence model should be used."

#########################################################
###           4. Word Embeddings                      ###
#########################################################

# Vectorization parameters
# Limit on the number of features. We use the top 20K features.
TOP_K = 20000 # words

# Limit on the length of text sequences. Sequences longer than this
# will be truncated.
# Figure out an ideal value based on the length of text graph above!!!!!!!
MAX_SEQUENCE_LENGTH = 500

def sequence_vectorize(train_texts, val_texts):
    """
    Vectorizes texts as sequence vectors.

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

"Label Vectorization:"
"""
We can simply convert labels into values in range [0, num_classes - 1]
"""

y_train = LabelEncoder().fit_transform(train["Topic"])
y_val = LabelEncoder().fit_transform(test["Topic"])
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)


"Which layer should we use last?:"
def get_last_layer_units_and_activation(num_classes):
    """Gets the # units and activation function for the last network layer.
    # Arguments
        num_classes: int, number of classes.
    # Returns
        units, activation values.
    """
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation

# In this project, we have 4 classes
get_last_layer_units_and_activation(num_classes=4)


"Build the sequence model:"
"""
Learning word relationships works best over many samples
The following code constructs a four-layer sepCNN model:
"""

maxlen = 124
inp = Input(shape=(maxlen, ))
embed_size = 128
x = Embedding(TOP_K, embed_size)(inp)
x = LSTM(60, return_sequences=True, name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(4, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)

# OR

maxlen = 124 # Must equal the # of columns in x_train!!!
model = Sequential()
model.add(Embedding(TOP_K, 128, input_length=maxlen, name="Embedding"))
model.add(LSTM(units=60, return_sequences=True, name="LSTM_layer"))
model.add(GlobalMaxPool1D())
model.add(Dropout(rate=0.1))
model.add(Dense(units=50, activation="relu"))
model.add(Dropout(rate=0.1))
model.add(Dense(units=4, activation="softmax"))



"Compile the Model:"
model.summary()
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', "AUC"])


"Fit the Model:"
# Using 2 epochs
history = model.fit(x_train, y_train,
          epochs=4,
          verbose=1,
          validation_data=(x_val, y_val),
          batch_size=64)




"Plot the Model:"

def plot_history(history):
    sns.set()
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

        ## As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plot_history(history=history)

































