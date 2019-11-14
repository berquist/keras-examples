import numpy as np

from tqdm import tqdm

from keras import Model
from keras.layers import Dense, Input, LSTM, Masking, TimeDistributed


n_neurons = 5
n_batch = 5
n_epoch = 1000

OPTIMIZER = "adam"
LOSS = "mean_squared_error"

length = 24
# seq = np.array([i / length for i in range(length)])
seq = np.array([3.0 + (i / length) for i in range(length)])

X2 = seq.reshape(4, 3, 2)
# (4, 3, 1)
y2 = np.array([[1.0], [2.0], [3.0], [4.0]])

# l1_4 = Input(shape=(3, 2))
# l2_4 = LSTM(n_neurons)(l1_4)
# l3_4 = Dense(1)(l2_4)
# model_4 = Model(inputs=l1_4, outputs=l3_4)
# model_4.compile(optimizer=OPTIMIZER, loss=LOSS)
# model_4.summary()
# for _ in tqdm(range(n_epoch)):
#     model_4.fit(X2, y2, epochs=1, batch_size=n_batch, verbose=0)
# model_4.predict(X2, batch_size=n_batch, verbose=0)

# l1_5 = Input(shape=(3, 2))
# l2_5 = Masking(mask_value=0.0)(l1_5)
# l3_5 = LSTM(n_neurons)(l2_5)
# l4_5 = Dense(1)(l3_5)
# model_5 = Model(inputs=l1_5, outputs=l4_5)
# model_5.compile(optimizer=OPTIMIZER, loss=LOSS)
# model_5.summary()
# for _ in tqdm(range(n_epoch)):
#     model_5.fit(X2, y2, epochs=1, batch_size=n_batch, verbose=0)
# model_5.predict(X2, batch_size=n_batch, verbose=0)

y22 = (y2 * np.ones((1, 3)))[..., np.newaxis]

l1_6 = Input(shape=(3, 2))
l2_6 = LSTM(n_neurons, return_sequences=True)(l1_6)
l3_6 = TimeDistributed(Dense(1))(l2_6)
model_6 = Model(inputs=l1_6, outputs=l3_6)
model_6.compile(optimizer=OPTIMIZER, loss=LOSS)
model_6.summary()
for _ in tqdm(range(n_epoch)):
    model_6.fit(X2, y22, epochs=1, batch_size=n_batch, verbose=0)
res = model_6.predict(X2, batch_size=n_batch, verbose=0)
print(y22)
print(res)

l1_7 = Input(shape=(3, 2))
l2_7 = Masking(mask_value=0.0)(l1_7)
l3_7 = LSTM(n_neurons, return_sequences=True)(l2_7)
l4_7 = TimeDistributed(Dense(1))(l3_7)
model_7 = Model(inputs=l1_7, outputs=l4_7)
model_7.compile(optimizer=OPTIMIZER, loss=LOSS)
model_7.summary()
model_7.set_weights(model_6.get_weights())
res = model_7.predict(X2, batch_size=n_batch, verbose=0)
print(y22)
print(res)

# Change the masking value to be something crazy
l1_8 = Input(shape=(3, 2))
l2_8 = Masking(mask_value=-999.0)(l1_8)
l3_8 = LSTM(n_neurons, return_sequences=True)(l2_8)
l4_8 = TimeDistributed(Dense(1))(l3_8)
model_8 = Model(inputs=l1_8, outputs=l4_8)
model_8.compile(optimizer=OPTIMIZER, loss=LOSS)
model_8.summary()
model_8.set_weights(model_6.get_weights())
res = model_8.predict(X2, batch_size=n_batch, verbose=0)
print(y22)
print(res)
