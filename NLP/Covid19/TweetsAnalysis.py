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
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline

pd.set_option("display.max_colwidth", 200)

tweets = pd.read_csv("2020-03-29 Coronavirus Tweets.CSV", header=0, verbose=1)

# Select only the English tweets:
EN_tweets = tweets["lang"]=="en"
EN_tweets = tweets[EN_tweets]


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
EN_tweets["Clean"] = EN_tweets["text"].apply(Clean_DF_Text)


# Take a sample of the dataset for testing, cause otherwise, you'll have an out-of-memory error:
sample = EN_tweets.sample(n=1000)
sample = sample.sort_values(by=["created_at"])

# Select only the necessary columns:
sample = sample[["created_at", "text", "Clean"]]

"Sentiment Analysis via Hugging Face:"
corpus = list(sample["Clean"].values)

nlp_sentiment = pipeline("sentiment-analysis")

# Run the model:
# This will take some time:
sample["Sentiment"] = nlp_sentiment(corpus)

# Extract sentiment and score into separate columns:
sample["Sentiment_Label"] = [x.get("label") for x in sample["Sentiment"]]
sample["Sentiment_Score"] = [x.get("score") for x in sample["Sentiment"]]

# Turn negative sentiment have a negative sentiment score:
sample["Sentiment_Score"] = np.where(
    sample["Sentiment_Label"] == "NEGATIVE", -(sample["Sentiment_Score"]), sample["Sentiment_Score"]
)

# Create an Interactive Heatmap of Sentiments:
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

