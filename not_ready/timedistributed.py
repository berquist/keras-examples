import numpy as np

from tqdm import tqdm

from keras import Model
from keras.layers import Dense, Input, LSTM, Masking, TimeDistributed
from keras.models import Sequential


n_neurons = 5
n_batch = 5
n_epoch = 1000

OPTIMIZER = "adam"
LOSS = "mean_squared_error"

length = 5
seq = np.array([i / length for i in range(length)])

### one-to-one
# (n, 1, 1): n batches, 1 time step, 1 feature/step

X = seq.reshape(-1, 1, 1)
y = seq.reshape(-1, 1)

model_1 = Sequential(
    [
        LSTM(n_neurons, input_shape=(1, 1)),
        Dense(1),
    ]
)
model_1.compile(optimizer=OPTIMIZER, loss=LOSS)
model_1.summary()
for _ in tqdm(range(n_epoch)):
    model_1.fit(X, y, epochs=1, batch_size=n_batch, verbose=0)
result = model_1.predict(X, batch_size=n_batch, verbose=0)
print(result)

### many-to-one (without TimeDistributed)
# (1, n, 1): 1 batch, n time steps, 1 feature/step

X = seq.reshape(1, -1, 1)
y = seq.reshape(1, -1)

model_2 = Sequential(
    [
        LSTM(n_neurons, input_shape=(length, 1)),
        Dense(length),
    ]
)
model_2.compile(optimizer=OPTIMIZER, loss=LOSS)
model_2.summary()
for _ in tqdm(range(n_epoch // 2)):
    model_2.fit(X, y, epochs=1, batch_size=1, verbose=0)
result = model_2.predict(X, batch_size=n_batch, verbose=0)
print(result)

### many-to-many (with TimeDistributed)
# (1, n, 1): 1 batch, n time steps, 1 feature/step

X = seq.reshape(1, -1, 1)
y = seq.reshape(1, -1, 1)

model_3 = Sequential(
    [
        LSTM(n_neurons, input_shape=(length, 1), return_sequences=True),
        TimeDistributed(Dense(1)),
    ]
)
model_3.compile(optimizer=OPTIMIZER, loss=LOSS)
model_3.summary()
for _ in tqdm(range(n_epoch)):
    model_3.fit(X, y, epochs=1, batch_size=n_batch, verbose=0)
result = model_3.predict(X, batch_size=n_batch, verbose=0)
print(result)
