import os
abspath = os.path.abspath('C:/Users/MatthiasQ.MATTQ/Desktop/Python Projects/NLP')
os.chdir(abspath)

import pandas as pd
df = pd.read_csv("wine.csv", index_col = 0)
import numpy as np
from PIL import Image
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


#Select only US wines
df.columns
US = df[df["country"] == "US"]

#Lowercase title and description
stopwords = set(STOPWORDS)
stopwords.update(['wine', 'drink', 'now', 'flavor', 'flavors'])

df["description"] = df["description"].str.lower()
df["title"] = df["title"].str.lower()

text = df.description[0]

#Making a word cloud
wordcloud = WordCloud(
                max_words = 100,
                stopwords = stopwords).generate(text)
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')