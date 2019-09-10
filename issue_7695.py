import numpy as np
from keras import Model
from keras.layers import Bidirectional, LSTM, Input, Masking

np.set_printoptions(linewidth=200)

## https://github.com/keras-team/keras/issues/7695

n_samples = 2
dx = 2
dy = 3
dout = 4
mask_value = -1

X = np.random.randint(5, size=(n_samples, dx, dy))
X[1, 0, :] = mask_value

inp = Input(shape=(dx, dy))
x = Masking(mask_value=-1.0)(inp)

lstm = LSTM(dout, return_sequences=True)(x)
model_1 = Model(inputs=inp, outputs=lstm)
model_1.summary()
model_1.set_weights(
    [np.ones(l.shape) * i for i, l in enumerate(model_1.get_weights(), 2)]
)
model_1.compile(optimizer="rmsprop", loss="mae")
y_true = np.ones((n_samples, dx, model_1.layers[2].output_shape[-1]))
y_pred_1 = model_1.predict(X)
print(y_pred_1)
unmasked_loss = np.abs(1 - y_pred_1).mean()
masked_loss = np.abs(1 - y_pred_1[y_pred_1 != 0.0]).mean()
keras_loss = model_1.evaluate(X, y_true, verbose=0)
print(f"unmasked loss: {unmasked_loss}")
print(f"masked loss: {masked_loss}")
print(f"evaluate with Keras: {keras_loss}")

bilstm = Bidirectional(LSTM(4, return_sequences=True), merge_mode="concat")(x)
model_2 = Model(inputs=inp, outputs=bilstm)
model_2.summary()
model_2.set_weights(
    [np.ones(l.shape) * i for i, l in enumerate(model_2.get_weights(), 2)]
)
model_2.compile(optimizer="rmsprop", loss="mae")
y_true = np.ones((n_samples, dx, model_2.layers[2].output_shape[-1]))
y_pred_2 = model_2.predict(X)
print(y_pred_2)
unmasked_loss = np.abs(1 - y_pred_2).mean()
masked_loss = np.abs(1 - y_pred_2[y_pred_2 != 0.0]).mean()
keras_loss = model_2.evaluate(X, y_true, verbose=0)
print(f"unmasked loss: {unmasked_loss}")
print(f"masked loss: {masked_loss}")
print(f"evaluate with Keras: {keras_loss}")
