
import numpy as np

embeddings_dictionary = dict()
glove_file = open("C:/Users/miqui/OneDrive/Python-Projects/NLP/GLOVE Embeddings/glove.6B.100d.txt",
                  encoding="utf-8")
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype="float32")
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()
