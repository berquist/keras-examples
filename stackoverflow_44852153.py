# Adapted from https://stackoverflow.com/q/44852153

import keras
from keras import Model
from keras.layers import Dense, Input, LSTM

maxlen = 40
chars = ["a", "b", "c", "d", "e", "f", "g"]
books = ["I am a book", "I am another book"]

net_input = Input(shape=(maxlen, len(chars)), name="net_input")
lstm_out = LSTM(128, input_shape=(maxlen, len(chars)))(net_input)

book_out = Dense(len(books), activation="softmax", name="book_output")(lstm_out)
char_out = Dense(len(chars) - 4, activation="softmax", name="char_output")(lstm_out)

x = keras.layers.concatenate([book_out, char_out])
net_output = Dense(len(chars) + len(books), activation="sigmoid", name="net_output")(x)

model = Model(inputs=[net_input], outputs=[net_output])
