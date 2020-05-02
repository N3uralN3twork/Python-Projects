"""Date: 21st October 2019
   Author: Matthias Quinn
   Goals: Predicting how many points a wine will receive from the sommelier
          Find alternatives to TF-IDF
   Dataset: https://www.kaggle.com/zynicide/wine-reviews
   Sources: https://towardsdatascience.com/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa
            https://www.geeksforgeeks.org/python-program-to-count-words-in-a-sentence/
   """

"NOTES:"
***I just learned about text classification, so I was wondering if I could extend that to a multivariate perspective
***Instead of just the reviews, I would incorporate other predictor variables.
***Feel like this would help for the upcoming DataFest, but I also have an exam this week...
***Hard part is figuring out how to make the textual data into useful features while retaining as much information as possible.
***Bag of words model just one-hot-encodes every single unique word, which I don't want
***TF-IDF does not retain semantics, structure, sequence, or context of text
***Goal is to find alternatives to TF-IDF
***Let us try word embeddings
***How to remove accents from text data?
***One of the features could be the length of the review
###############################################################################
###                     1.  Define Working Directory                        ###
###############################################################################
import os
abspath = os.path.abspath('C:/Users/MatthiasQ.MATTQ/Desktop/Python Projects/NLP')
os.chdir(abspath)
###############################################################################
###                    2. Import Libraries, Models, and Data                ###
###############################################################################
import pandas as pd
import numpy as np
import re
import nltk
import spacy
import unicodedata
import seaborn as sns
from textblob import TextBlob
from pycm import ConfusionMatrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

wines = pd.read_excel("wines.xlsx")


################################################################################
###                   3. Data PreProcessing                                  ###
################################################################################
wines.head(2)
wines.shape #129,971 rows
to_drop = ["id", "region_2", "designation", "taster_name", "taster_twitter_handle", "title", "winery"]
wines = wines.drop(columns = to_drop, axis = 1)
wines.columns


#Rename the columns:
# df.rename(columns = {"old": "new"})
wines = wines.rename(columns = {"country": "Country",
                                "description": "Review",
                                "points": "Points",
                                "price": "Price",
                                "province": "Province",
                                "region_1": "Region",
                                "variety": "Variety"})
wines.columns


#Text Cleaner:
#From the Preprocessing Text Project.
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
    clean_words = [word for word in words if word not in stopwords]
    text = ' '.join(clean_words)
    return text

wines["Review"] = wines["Review"].apply(Clean_DF_Text)
wines["Variety"] = wines["Variety"].astype(str) #Turn into strings first
wines["Variety"] = wines["Variety"].apply(Clean_DF_Text)


#Outcome variable = # points
#Is it normally distributed?
ax = sns.distplot(wines["Points"])
#Yup!

Categorical = list(wines.select_dtypes(include=['object', 'category']).columns)
for name in Categorical:
    print(name, ":")
    print(wines[name].value_counts(), "\n")

def CombineCategories(df, Category, N):
    top_N = df[Category].value_counts().nlargest(N).index
    update = df[Category].where(df[Category].isin(top_N),
                                other = "Other")
    return update
variety = CombineCategories(wines, "Variety", N = 20) #Keep top 2 categories and rest are "Other"
wines["Variety"] = variety
wines["Variety"].value_counts()

Country = CombineCategories(wines, "Country", N = 12)
wines["Country"] = Country
################################################################################
###                   4. Feature Engineering                                 ###
################################################################################
"1. Length of the review:"
def Length(text):
    Length = len(text)
    return Length
Length("Hello, how was your day?")
wines["ReviewLength"] = wines["Review"].apply(Length)

"2. Number of words in the review:"
def NumWords(text):
    NumWords = len(text.split())
    return NumWords

wines["NumWords"] = wines["Review"].apply(NumWords)

#Probably highly correlated, I'd imagine

"3. Review Sentiment:"
wines["Sentiment"] = wines["Review"].apply(lambda x: TextBlob(x).sentiment[0])

wines["Sentiment"].describe()



################################################################################
###                   5. Train-test Split                                    ###
################################################################################
x = wines[["Country", "Review", "Price", "Province", "Region", "Variety", "ReviewLength", "NumWords", "Sentiment"]]
y = wines["Points"]

x_train, x_test, y_train, y_test  = train_test_split(x, y,
                                                     random_state = 123,
                                                     shuffle = True,
                                                     train_size = 0.7)

################################################################################
###                   6. Converting Text to Features                         ###
################################################################################
"4. Universal Sentence Encoder:"



