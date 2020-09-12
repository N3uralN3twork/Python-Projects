"""
Sources:
    https://stackoverflow.com/questions/5511708/adding-words-to-nltk-stoplist
    https://towardsdatascience.com/basic-tweet-preprocessing-in-python-efd8360d529e
    https://stackoverflow.com/questions/2527892/parsing-a-tweet-to-extract-hashtags-into-an-array
"""
#########################################################
###           1. Set the working directory            ###
#########################################################
import os
abspath = os.path.abspath("C:/Users/miqui/OneDrive/Python-Projects/NLP/Covid19")
os.chdir(abspath)
os.listdir()
#########################################################
###           2. Import Data and Libraries            ###
#########################################################
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline

pd.set_option("display.max_colwidth", 200)

tweets = pd.read_csv("2020-03-29 Coronavirus Tweets.CSV", header=0)
tweets.info()

# Select only the English tweets:
# 313,036 tweets are in English
EN_tweets = tweets[tweets["lang"] == "en"]
# Remove dataset to clear space:
del tweets

EN_tweets


# Create a helper function to clean the tweets:
# This is from a previous project

def Clean_DF_Text(text):
    """Order matters so don't move things around"""
    from nltk.corpus import stopwords
    import unicodedata
    text = re.sub(r'http\S+', " ", text)  # Remove urls
    text = re.sub('<[^<]+?>', '', text)  # Remove HTML tags
    text = re.sub("[^a-zA-Z]", " ", text)  # Remove punctuation
    text = re.sub('[0-9]+', " ", text)  # Remove numbers
    text = text.strip()  # Remove whitespaces
    text = re.sub(" +", " ", text)  # Remove extra spaces
    text = text.lower()  # Lowercase text
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")  # Remove accents
    stopwords = stopwords.words('english')
    newStopWords = ["co", "https", "p"]  # Add custom stopwords
    stopwords.extend(newStopWords)  # Don't use append!
    stopwords.remove("not")  # Important custom words to keep
    words = text.split()  # Split the words into separate items
    clean_words = [word for word in words if word not in stopwords]  # List comprehension to remove stopwords
    clean_words = [word for word in clean_words if len(word) > 2]  # Remove words that are shorter than 3 letters
    text = ' '.join(clean_words)
    return text


# Apply the cleaning function to our articles:
    # This will take a minute:
EN_tweets["Clean"] = EN_tweets["text"].apply(Clean_DF_Text)

# Extract the hashtags from a tweet:
EN_tweets["Hashtags"] = EN_tweets["text"].apply(lambda x: re.findall(r"#(\w+)", x))

# Take a sample of the dataset for testing, cause otherwise, you'll have an out-of-memory error:
sample = EN_tweets.sample(n=1000)
sample = sample.sort_values(by=["created_at"])

# Select only the necessary columns:
sample = sample[["created_at", "text", "Clean", "Hashtags"]]

# Take a look at some cleaned text to check if it's a good standard:
sample["Clean"]

"Sentiment Analysis via Hugging Face:"
corpus = list(sample["Clean"].values)

# Choose which task you want to perform:
nlp_sentiment = pipeline("sentiment-analysis")

# Run the model:
# This will take some time:
sample["Sentiment"] = nlp_sentiment(corpus)

# Extract sentiment and score into separate columns:
sample["Sentiment_Label"] = [x.get("label") for x in sample["Sentiment"]]
sample["Sentiment_Score"] = [x.get("score") for x in sample["Sentiment"]]  # the Score is a probability

# Turn negative sentiment have a negative sentiment score:
sample["Sentiment_Score"] = np.where(
    sample["Sentiment_Label"] == "NEGATIVE", -(sample["Sentiment_Score"]), sample["Sentiment_Score"]
)


# Create an Interactive Time-Series Plot of Sentiment:
fig = go.Figure(
    data=go.Scatter(
        x=sample["created_at"],
        y=sample["Sentiment_Score"]
    )
)
fig.update_layout(
    title=go.layout.Title(
        text="Sentiment Analysis of Tweets in March"
    ),
    autosize=False,
    width=1200,
    height=600
)

fig.update_xaxes(rangeslider_visible=True)

# Invert the y-axis:
fig.write_html("plot.html")

