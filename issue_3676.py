import numpy as np
from keras.engine.topology import Input, merge, Merge
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Masking
from keras.engine.training import Model
from numpy import dtype, int32

num_classes = 10
num_symbols = 10
datapoints = 100
seq_len = 10
x = np.random.randint(num_symbols, size=(datapoints, seq_len))
y = np.random.randint(num_classes, size=(datapoints, seq_len))
y_one_hot = np.zeros((datapoints, seq_len, num_classes))

for i in range(datapoints):
    for j in range(seq_len):
        y_one_hot[i][j][y[i][j]] = 1

embedding_size = 10
embedding_weights = np.zeros((num_symbols, embedding_size))

for i in range(num_symbols):
    embedding_weights[i] = np.random.rand(embedding_size)

input = Input(shape=(seq_len,), dtype="int32")
embedding = Embedding(num_symbols, embedding_size, input_length=seq_len)

embedded_input = embedding(input)

mask = Masking(mask_value=0)(embedded_input)

bidirect = Bidirectional(LSTM(100, return_sequences=True))(mask)

final = TimeDistributed(Dense(num_classes, activation="softmax"))(bidirect)

model = Model(input=[input], output=[final])

model.compile(
    loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
)

model.fit(x, y_one_hot)
