"""Essentially, the goal is to classify whether a given movie review
has a positive or negative sentiment. Not sure how this is going to work tho.
Source 1: https://towardsdatascience.com/multi-class-text-classification-with-sklearn-and-nltk-in-python-a-software-engineering-use-case-779d4a28ba5
"""
###############################################################################
###                     1.  Define Working Directory                        ###
###############################################################################
import os
abspath = os.path.abspath('C:/Users/miqui/OneDrive/Python-Projects/NLP')
os.chdir(abspath)
os.listdir()
###############################################################################
###                    2. Import Libraries, Models, and Data                ###
###############################################################################
import pandas as pd
import numpy as np
from pycm import ConfusionMatrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
RottenTomatoes = pd.read_excel(io = "RottenTomatoes.xlsx",
                               sheet_name = "RottenTomatoes")

###############################################################################
###                   3. Clean the movie reviews                            ###
###############################################################################
"I'm using a few code snipptes from the Preprocessing Text project."

RottenTomatoes.columns
RottenTomatoes.shape
#10,000 reviews
RottenTomatoes["Freshness"].value_counts().plot.bar()
#Looks balanced

"To clean multiple texts in a dataframe quickly"
def Clean_DF_Text(text):
    import re
    from nltk.corpus import stopwords
    text = re.sub('<[^<]+?>', '', text) #Remove HTML
    text = re.sub("[^a-zA-Z]", " ", text) #Remove punctuation
    text = re.sub('[0-9]+', " ", text) #Remove numbers
    text = text.strip()     #Remove whitespaces
    text = re.sub(" +", " ", text) #Remove extra spaces
    text = text.lower()     #Lowercase text
    stopwords = stopwords.words('english')
    words = text.split()
    clean_words = [word for word in words if word not in stopwords]
    text = ' '.join(clean_words)
    return text

#Apply our function across all 10,000 reviews
RottenTomatoes["Clean"] = RottenTomatoes["Review"].apply(Clean_DF_Text)

RottenTomatoes["Clean"].head(2)

#Remove the original dirty text
RottenTomatoes = RottenTomatoes.drop(["Review"], axis = 1)


############################################################################
###                     4. Train-Test Split                              ###
############################################################################
x = RottenTomatoes["Clean"]
y = RottenTomatoes["Freshness"]

x_train, x_test, y_train, y_test  = train_test_split(x, y,
                                                     random_state = 123,
                                                     shuffle = True,
                                                     train_size = 0.7)

x_train.shape
x_test.shape
#############################################################################
###                      5. Converting Text to Features                   ###
#############################################################################
"I legit have no idea what to do after this point (20th October, 2019)"
"How in the world do you transform a bunch of text to numbers?"
"In what representation would that work?"

#Each corpus is transformed into vector space model using TF-IDF vectorizer to extract features
vectorizer = TfidfVectorizer(min_df = 3, #Lower = higher number of columns
                             stop_words = "english",
                             sublinear_tf = True,
                             norm = "l2", #L2 normalizer
                             ngram_range = (1, 2))

final = vectorizer.fit_transform(RottenTomatoes["Clean"]).toarray()

final.shape
#10,000 rows and 8,856 features


###############################################################################
###                         6. Model Building and Prediction                ###
###############################################################################

#I doubt we need all 8,856 features so we'll use a chi-square test to select the best features
#The chi-square test will filter out features that are irrelevant for classification
#We'll use a pipeline to vectorize, select the best features, and build a classifier

pipeline = Pipeline([("Vectorizer", vectorizer),
                     ("ChiSquare", SelectKBest(chi2, k = 1200)),
                     ("Classifier", RandomForestClassifier(n_estimators = 400,
                                                           n_jobs = -1,
                                                           random_state = 123,
                                                           verbose = 1))])
model = pipeline.fit(x_train, y_train)

ytest = np.array(y_test)

#Make our predictions
preds = model.predict(x_test)

#Confusion Matrix and Extra Statistics:
cm = ConfusionMatrix(actual_vector = ytest,
                     predict_vector = preds)
print(cm)





###############################################################################
###                         7. Neural Network                               ###
###############################################################################

max_features = 6000
tokenizer = Tokenizer(num_words = max_features)
tokenizer.fit_on_texts(RottenTomatoes['Clean'])
list_tokenized_train = tokenizer.texts_to_sequences(RottenTomatoes["Clean"])

maxlen = 130
X_t = pad_sequences(list_tokenized_train, maxlen = maxlen)
y = RottenTomatoes["Freshness"]
y = np.asarray(y)

dict = {"fresh": 1,
        "rotten": 0}

RottenTomatoes["Freshness"] = RottenTomatoes["Freshness"].map(dict)

RottenTomatoes["Freshness"].value_counts()



embed_size = 128
model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation = "relu"))
model.add(Dense(10, activation = "relu"))
model.add(Dropout(0.05))
model.add(Dense(1, activation = "sigmoid"))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

batch_size = 100
epochs = 5
model.fit(X_t, y,
          batch_size = batch_size,
          epochs = epochs,
          validation_split=0.2,
          verbose = 1)