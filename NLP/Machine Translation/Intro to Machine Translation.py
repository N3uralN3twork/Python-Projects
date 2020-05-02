# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:02:54 2019

@author: MatthiasQ

Translating French to English
using various NMT models
"""
import os
abspath = os.path.abspath('D:/Python Projects/NLP/deu-eng')
os.chdir(abspath)

import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array
 
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# split a loaded document into sentences
def to_pairs(doc):
	lines = doc.strip().split('\n')
	pairs = [line.split('\t') for line in  lines]
	return pairs
 
# clean a list of lines
def clean_pairs(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for pair in lines:
		clean_pair = list()
		for line in pair:
			# normalize unicode characters
			line = normalize('NFD', line).encode('ascii', 'ignore')
			line = line.decode('UTF-8')
			# tokenize on white space
			line = line.split()
			# convert to lowercase
			line = [word.lower() for word in line]
			# remove punctuation from each token
			line = [word.translate(table) for word in line]
			# remove non-printable chars form each token
			line = [re_print.sub('', w) for w in line]
			# remove tokens with numbers in them
			line = [word for word in line if word.isalpha()]
			# store as string
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	return array(cleaned)
 
# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved as: %s' % filename)
 
# load dataset
filename = 'deu.txt'
doc = load_doc(filename)
# split into english-german pairs
pairs = to_pairs(doc)
# clean sentences
clean_pairs = clean_pairs(pairs)
#Save clean pairs to file
"save_clean_data(clean_pairs, 'english-german.pkl')"
#Check

for i in range(30):
    print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))
#Cool

#Save some space
del pairs
del clean_pairs

"2. Split Text"
from pickle import load
from pickle import dump
from numpy.random import rand
from numpy.random import shuffle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

#Load the clean dataset
def load_clean_sentence(filename):
    return load(open(filename, 'rb'))

def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print("Saved as: %s" % filename)

#Load dataset
raw_dataset = load_clean_sentence('english-german.pkl')

#Reduce dataset size
n_sentences = 10000
dataset = raw_dataset[:n_sentences, :]

del raw_dataset
#Random Shuffle
shuffle(dataset)

#Train-test set
train, test = dataset[:9000], dataset[9000:]
#Save
save_clean_data(dataset, 'english-german-both.pkl')
save_clean_data(train, 'english-german-train.pkl')
save_clean_data(test,  'english-german-test.pkl')

#Load datasets
dataset = load_clean_sentence('english-german-both.pkl')
train = load_clean_sentence('english-german-train.pkl')
test  = load_clean_sentence('english-german-test.pkl')

"Tokenizer"
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

#Max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)

#Prepare the english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])

print("English Vocabulary Size: %d" % eng_vocab_size)
print("English Max Length: %d" % (eng_length))

#Prepare the german tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size= len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])

#Encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    #Integer encode sequences
    X = tokenizer.text_to_sequences(lines)
    #pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X

"One-hot encode english output sequences"
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y



"Prepare training data"
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_output(trainY, eng_vocab_size)

"Prepare validation data"
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
testY = encode_output(testY, eng_vocab_size)
 
"Define model"
model = define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')

print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True)

filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=2, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)


















    
    
    
    
    
    
    