# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:02:54 2019
Translating German to English using various
NMT models
@author: MatthiasQ
"""
import string
import re
from numpy import array, argmax, random, take
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
import matplotlib.pyplot as plt
pd.set_option('display.max_colwidth', 200)

"Reading in the English and German Datasets"

