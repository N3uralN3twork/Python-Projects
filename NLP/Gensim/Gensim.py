import gensim
from gensim import corpora
from pprint import pprint
from gensim.utils import simple_preprocess
from smart_open import smart_open
from gensim.summarization import summarize, keywords
import os

documents = ["The Saudis are preparing a report that will acknowledge that",
           "Saudi journalist Jamal Khashoggi's death was the result of an",
           "interrogation that went wrong, one that was intended to lead",
           "to his abduction from Turkey, according to two sources."]

documents2 = ["One source says the report will likely conclude that",
              "the operation was carried out without clearance and",
              "transparency and that those involved will be held",
              "responsible. One of the sources acknowledged that the",
              "report is still being prepared and cautioned that",
               "things could change."]

#Split the sentence into words:
texts = [[text for text in text.split()] for text in text1]
dictionary = corpora.Dictionary(texts)
print(dictionary)
print(dictionary.token2id)

#Create gensime dictionary from a single text file:
dictionary = corpora.Dictionary(simple_preprocess(doc = line, deacc = True)
                                for line in open('sample.txt', encoding = 'utf-8'))
#Token to ID map:
dictionary.token2id

#Once you have updated the dictionary, all you need to do to create a bag of words corpus
#is to pass the tokenized list of words to the Dictionary.doc2bow()

my_docs = ["Who let the dogs out?",
           "Who? Who? Who? Who?"]
#Tokenize the docs
tokenized_list = [simple_preprocess(doc) for doc in my_docs]

#Create the corpus
mydict = corpora.Dictionary()
mycorpus = [mydict.doc2bow(doc, allow_update=True) for doc in tokenized_list]
pprint(mycorpus)
#The (4,4) means the word with id = 4 appears 4 times in the second text
word_counts = [[(mydict[id], count) for id, count in line] for line in mycorpus]
print(word_counts)


#To summarize the documents:
text = " ".join((line for line in smart_open("sample.txt", encoding = 'utf-8')))
print(summarize(text, word_count = 20))

#Important keywords:
print(keywords(text))
