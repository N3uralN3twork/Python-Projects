import os
abspath = os.path.abspath('C:/Users/MatthiasQ.MATTQ/Desktop/Python Projects/NLP')
os.chdir(abspath)
import pandas as pd
names = ["ID", "Sentiment", "Review"]
reviews = pd.read_excel("IMDBReviews.xlsx",
                        names = names)

reviews = reviews.drop(["ID"], axis = 1)
reviews.columns

pd.crosstab(reviews["Sentiment"],
            columns = 'count')

#How convenient

import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stopwords = stopwords.words("english")
stopwords = set(stopwords)

lemma = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemma.lemmatize(token) for token in text.split(" ")]
    text = [lemma.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stopwords]
    text = " ".join(text)
    return text

reviews["Review"] = reviews.Review.apply(lambda x: clean_text(x))




"Model Building"
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers



max_features = 6000
tokenizer = Tokenizer(num_words = max_features)
tokenizer.fit_on_texts(reviews["Review"])
list_tokenized_train = tokenizer.texts_to_sequences(reviews["Review"])


maxlen = 130
X_t = pad_sequences(list_tokenized_train, maxlen = maxlen)
y = reviews["Review"]

embed_size = 128

model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))

model.summary()
#Number of parameters
model.count_params()

model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

model.fit(x = X_t, y = y,
          batch_size = 100,
          epochs = 1,
          verbose = 1,
          validation_split = 0.2)
