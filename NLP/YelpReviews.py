"""
Title: Yelp Pizza Reviews
Author: Matt Quinn
Date: 2nd May, 2020
Sources:
    https://towardsdatascience.com/build-and-compare-3-models-nlp-sentiment-prediction-67320979de61
    http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-19_17.html
Dataset:
    https://raw.githubusercontent.com/SarthakRana/Restaurant-Reviews-using-NLP-/master/Restaurant_Reviews.tsv
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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from pycm import ConfusionMatrix

yelp = pd.read_excel("YelpReviews.xlsx", sheet_name="Pizza", header=0)


# Get an overview of what we're dealing with
yelp.head(10) # 1,000 obs. & 1 predictor variable
yelp.columns
yelp["Review"].describe()
pd.crosstab(index=yelp["Liked"], columns="count")
# Looks like there are 4 duplicates
# 498 dislikes
# 502 likes
# Pretty balanced data set

#########################################################
###           3. Data Cleaning                        ###
#########################################################
"Removing duplicate values:"
# From an earlier project:
def duplicate(df): # Addendum: 17 April 2020
    start = df.shape[0]
    df2 = df.drop_duplicates(keep = "first")
    end = df2.shape[0]
    print(f"Dropped {start-end} duplicates")
    return df2
yelp = duplicate(yelp)

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
    words = text.split()
    clean_words = [word for word in words if word not in stopwords] # List comprehension
    text = ' '.join(clean_words)
    return text

# Apply the cleaning function to our reviews:

yelp.columns
yelp["Clean"] = yelp["Review"].apply(Clean_DF_Text)

yelp.head()

# Easy as that


# Creating a bag-of-words model
    # Allows us to extract features from our text data
    # Setting the max-features option sets the max number of words to use

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(yelp["Clean"]).toarray()
y = yelp["Liked"].values

X.shape # 996 obs. and top 1500 words

# Splitting into training and test sets:
    # 80/20 split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


#########################################################
###           4. Modelling                            ###
#########################################################
"Naive Bayes Model"

# In a previous R project, we learned about the NB model
    # Based on Bayes' theorem
    # Assumes independence and normally distributed data

nb = GaussianNB().fit(X_train, y_train)
nb

# Predictions:
nb_preds = nb.predict(X_test)

# Confusion Matrix and Other Statistics
cm = ConfusionMatrix(actual_vector = y_test,
                     predict_vector = nb_preds)
print(cm)


"Decision Tree Model"

dtree = DecisionTreeClassifier(criterion="entropy", random_state=123).fit(X_train, y_train)
dtree

dt_preds = dtree.predict(X_test)

# Confusion Matrix and Other Statistics
print(ConfusionMatrix(actual_vector=y_test,
                      predict_vector=dt_preds))

# Worse than the NB classifier

"Random Forest Model"

rf = RandomForestClassifier(n_estimators=300, criterion="entropy", random_state=123).fit(X_train, y_train)
rf

rf_preds = rf.predict(X_test)

# Confusion Matrix and Other Statistics

print(ConfusionMatrix(actual_vector=y_test,
                      predict_vector=rf_preds))

# Now we know how to predict with text data!
