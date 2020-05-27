from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras import utils
from keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(x_train, maxlen=20000)
x_val = pad_sequences(x_val, maxlen=20000)

from keras.utils import to_categorical
y_binary = to_categorical(y_train)
y_val = to_categorical(y_val)

model = Sequential()
model.add(Dense(10000, input_shape=(20000,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_binary,
          epochs=1,
          validation_data=(x_val, y_val))