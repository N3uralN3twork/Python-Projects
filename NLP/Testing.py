from transformers import pipeline
nlp = pipeline(task = "sentiment-analysis")
print(nlp("I love you"))
print(nlp("Today was a really good day for me!"))