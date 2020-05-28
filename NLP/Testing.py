import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing import sequence, text
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, GlobalMaxPool1D
from tensorflow.keras.models import Model, Sequential

maxlen = 124
inp = Input(shape=(maxlen, ))
embed_size = 128
x = Embedding(TOP_K, embed_size)(inp)
x = LSTM(60, return_sequences=True, name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(4, activation="sigmoid")(x)

model = Sequential()
model.add(Embedding(20000, 128, input_length=124, name="Embedding"))
model.add(LSTM(units=60, return_sequences=True, name="LSTM_layer"))
model.add(GlobalMaxPool1D())
model.add(Dropout(rate=0.1))
model.add(Dense(units=50, activation="relu"))
model.add(Dropout(rate=0.1))
model.add(Dense(units=4, activation="sigmoid"))

model.summary()

model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=1,
          verbose=1,
          validation_data=(x_val, y_val))







