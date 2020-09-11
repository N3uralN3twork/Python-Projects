"""
Title: IMDB Movie Reviews
Author: Matt Quinn
Date: 29th May 2020
Goal: Predict the sentiment of a movie review
Sources:
    https://colab.research.google.com/drive/1AstCNMK5_5MMKznrcKslUCFMCCNXk_ae#scrollTo=zxN6MvxWTSxt
    https://www.tensorflow.org/hub/tutorials/tf2_text_classification
Dataset:
    https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
    https://www.kaggle.com/atulanandjha/imdb-50k-movie-reviews-test-your-bert
Notes:
    To get the current full dataset, I just rbinded the two above data sources.
    ELMO is a rather large model (>350 MB)
"""

# That was stupid. Don't need a huge model to just tokenize a text for me when I have NLTK.


#########################################################
###           1. Set the working directory            ###
#########################################################
import os
abspath = os.path.abspath("C:/Users/miqui/OneDrive/Python-Projects/NLP/IMDB Movie Reviews")
os.chdir(abspath)
os.listdir()
#########################################################
###           2. Import Data and Libraries            ###
#########################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub

sns.set()
pd.set_option("display.max_colwidth", 200)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

train = pd.read_csv("IMDB Reviews.csv", header=0)
test = pd.read_csv("test.csv", header=0)

train.head()
test.head()
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
    # This will take a minute:
train["Clean"] = train["Review"].apply(Clean_DF_Text)
test["Clean"] = test["Review"].apply(Clean_DF_Text)

# Take a peek:
train.head()

# Turn Sentiments into 0's and 1's:
train["SentimentBinary"] = train["Sentiment"].map({"negative": 0, "positive": 1})
train["SentimentBinary"].value_counts()
test["SentimentBinary"] = test["Sentiment"].map({"negative": 0, "positive": 1})
test["SentimentBinary"].value_counts()



# Exploring the distribution of sentence lengths to figure out
# what max sentence length to set and how much padding to do.

lengths = [len(i) for i in train["Clean"]]
sns.distplot(lengths)
plt.show()


model = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(model, output_shape=2, input_shape=[],
                           dtype=tf.string, trainable=True)


model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(1))

# Take an overhead look of the model:
model.summary()

# Compile the model:
# Using the ADAM optimizer
# (Binary) Accuracy will be the metric used here

model.compile(optimizer="adam",
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name="accuracy")])

# Prepare your training and testing sets (using a subset of 10,000 training examples):
x_val = train["Clean"][:10000]
partial_x_train = train["Clean"][10000:]
y_val = train["SentimentBinary"][:10000]
partial_y_train = train["SentimentBinary"][10000:]

# Fit the neural network model:
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=10,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
