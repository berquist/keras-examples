import numpy as np
from keras import Model
from keras.layers import (
    Bidirectional,
    LSTM,
    Input,
    Masking,
    Concatenate,
    Dense,
    TimeDistributed,
)

np.set_printoptions(linewidth=250)

## https://github.com/keras-team/keras/issues/7695

n_samples = 2
dx = 2
dy = 3
dout = 20
mask_value = -1

X = np.random.randint(5, size=(n_samples, dx, dy))
X[1, 0] = mask_value
sample_weight = np.ones(shape=(n_samples, dx))
sample_weight[1, 0] = 0

inp = Input(shape=(dx, dy))

# A known working example.
x = Masking(mask_value=-1.0)(inp)
lstm = LSTM(dout, return_sequences=True)(x)
model_1 = Model(inputs=inp, outputs=lstm)
model_1.summary()
model_1.set_weights(
    [np.ones(l.shape) * i for i, l in enumerate(model_1.get_weights(), 2)]
)
model_1.compile(optimizer="rmsprop", loss="mae")
y_true = np.ones((n_samples, dx, model_1.layers[-1].output_shape[-1]))
y_pred_1 = model_1.predict(X)
print(y_pred_1)
unmasked_loss = np.abs(1 - y_pred_1).mean()
masked_loss = np.abs(1 - y_pred_1[y_pred_1 != 0.0]).mean()
keras_loss = model_1.evaluate(X, y_true, verbose=0)
print(f"unmasked loss: {unmasked_loss}")
print(f"masked loss: {masked_loss}")
print(f"evaluate with Keras: {keras_loss}")

lstm_3 = LSTM(dout, return_sequences=True)(inp)
model_3 = Model(inputs=inp, outputs=lstm_3)
model_3.summary()
model_3.compile(optimizer="rmsprop", loss="mae")
y_true = np.ones((n_samples, dx, model_3.layers[-1].output_shape[-1]))
y_true[1, 0] = 0.5
y_pred_3_before = model_3.predict(X)
print(y_pred_3_before)
model_3.fit(X, y_true, epochs=20)
y_pred_3 = model_3.predict(X)
print(y_pred_3)
unmasked_loss = np.abs(1 - y_pred_3).mean()
masked_loss = np.abs(1 - y_pred_3[y_pred_3 != 0.0]).mean()
keras_loss = model_3.evaluate(X, y_true, verbose=0)
print(f"unmasked loss: {unmasked_loss}")
print(f"masked loss: {masked_loss}")
print(f"evaluate with Keras: {keras_loss}")

lstm_3 = LSTM(dout, return_sequences=True)(inp)
dense = TimeDistributed(Dense(dout, activation="softmax"))(lstm_3)
model_3 = Model(inputs=inp, outputs=dense)
model_3.summary()
model_3.compile(optimizer="rmsprop", loss="mae")
y_true = np.ones((n_samples, dx, model_3.layers[-1].output_shape[-1]))
y_true[1, 0] = 0.5
y_pred_3_before = model_3.predict(X)
print(y_pred_3_before)
model_3.fit(X, y_true, epochs=20)
y_pred_3 = model_3.predict(X)
print(y_pred_3)
unmasked_loss = np.abs(1 - y_pred_3).mean()
masked_loss = np.abs(1 - y_pred_3[y_pred_3 != 0.0]).mean()
keras_loss = model_3.evaluate(X, y_true, verbose=0)
print(f"unmasked loss: {unmasked_loss}")
print(f"masked loss: {masked_loss}")
print(f"evaluate with Keras: {keras_loss}")
