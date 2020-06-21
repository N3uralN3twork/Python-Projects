

from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
corpus = api.load("text8")
model = Word2Vec(corpus)
dir(model)

