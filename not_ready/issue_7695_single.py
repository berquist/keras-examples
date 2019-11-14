import numpy as np
from keras import Model
from keras.layers import Bidirectional, LSTM, Input, Masking, Concatenate, Dense

np.set_printoptions(linewidth=200)

## https://github.com/keras-team/keras/issues/7695

n_samples = 2
dx = 2
dy = 3
dout = 7
mask_value = -1

X = np.random.randint(5, size=(n_samples, dx, dy))
X[1, 0, :] = mask_value

inp = Input(shape=(dx, dy))
x = Masking(mask_value=-1.0)(inp)
import pdb; pdb.set_trace()
lstm_fw = LSTM(dout, return_sequences=True, go_backwards=False)(x)
lstm_bw = LSTM(dout, return_sequences=True, go_backwards=True)(x)
concat = Concatenate(axis=-1)([lstm_fw, lstm_bw])
model_3 = Model(inputs=inp, outputs=concat)
model_3.summary()
model_3.set_weights(
    [np.ones(l.shape) * i for i, l in enumerate(model_3.get_weights(), 2)]
)
model_3.compile(optimizer="rmsprop", loss="mae")
y_true = np.ones((n_samples, dx, model_3.layers[-1].output_shape[-1]))
y_pred_3 = model_3.predict(X)
print(y_pred_3)
unmasked_loss = np.abs(1 - y_pred_3).mean()
masked_loss = np.abs(1 - y_pred_3[y_pred_3 != 0.0]).mean()
keras_loss = model_3.evaluate(X, y_true, verbose=0)
print(f"unmasked loss: {unmasked_loss}")
print(f"masked loss: {masked_loss}")
print(f"evaluate with Keras: {keras_loss}")
