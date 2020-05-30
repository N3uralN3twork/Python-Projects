"""
Title: IMDB Movie Reviews
Author: Matt Quinn
Date: 29th May 2020
Goal: Use BERT to predict the sentiment of a movie review
Sources:

Dataset:
    https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
Notes:

"""

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# Tokenize Sentence

text = "Who was Jim Henson? He was a puppeteer."
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)