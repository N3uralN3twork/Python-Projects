import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS



text = "Hello There https://www.google.com"
re.sub("((www\.[^\s]+)|(https?://[^\s]+))", "", text)
text = re.sub(r'http\S+', " ", text)  # Second