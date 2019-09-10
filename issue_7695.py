import numpy as np
from keras import Model
from keras.layers import Bidirectional, LSTM, Input, Masking

np.set_printoptions(linewidth=200)

## https://github.com/keras-team/keras/issues/7695

n_samples = 2
dx = 2
dy = 3
mask_value = -1

X = np.random.randint(5, size=(n_samples, dx, dy))
X[1, 0, :] = mask_value

inp = Input(shape=(dx, dy))
x = Masking(mask_value=-1.0)(inp)

lstm = LSTM(4, return_sequences=True)(x)
model_1 = Model(inputs=inp, outputs=lstm)
model_1.summary()
print(model_1.predict(X))

bilstm = Bidirectional(LSTM(4, return_sequences=True), merge_mode="concat")(x)
model_2 = Model(inputs=inp, outputs=bilstm)
model_2.summary()
print(model_2.predict(X))
