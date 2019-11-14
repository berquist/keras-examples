from pprint import pprint

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

n_samples = 4
dx = 2
dy = 3
dout = 1
mask_value = -1
epochs = 20

X1 = np.random.randint(5, size=(n_samples, dx, dy))
X2 = X1.copy()
y_true = np.ones((n_samples, dx, dout))
X2[2, 0] = mask_value
X2[3, 1] = mask_value
sample_weight = np.ones_like(y_true)
sample_weight[2, 0] = 0
sample_weight[3, 1] = 0

inp_1 = Input(shape=(dx, dy))
mask_1 = Masking(mask_value=mask_value)(inp_1)
lstm_1 = LSTM(dout, return_sequences=True)(mask_1)
dense_1 = TimeDistributed(Dense(dout))(lstm_1)
model_1 = Model(inputs=inp_1, outputs=dense_1)
model_1.summary()
model_1.compile(optimizer="rmsprop", loss="mae", sample_weight_mode="temporal")

model_1_untrained_weights = model_1.get_weights()

print(
    "The losses are not going to be consistent with each other because the"
    "masking layer breaks the model."
)
y_pred = model_1.predict(X2, verbose=0)
unmasked_loss = np.average(np.abs(y_true - y_pred))
masked_loss = np.average(np.abs(y_true - y_pred[y_pred != 0.0]))
weighted_loss = np.average(np.abs(y_true - y_pred), weights=sample_weight)
keras_loss = model_1.evaluate(X2, y_true, verbose=0)
keras_loss_weighted = model_1.evaluate(
    X2, y_true, sample_weight=sample_weight[..., 0], verbose=0
)
print("-- model 1 --")
print(f"unmasked loss: {unmasked_loss}")
print(f"masked loss: {masked_loss}")
print(f"weighted loss: {weighted_loss}")
print(f"evaluate with Keras: {keras_loss}")
print(f"evaluate with Keras (weighted): {keras_loss_weighted}")

inp_2 = Input(shape=(dx, dy))
lstm_2 = LSTM(dout, return_sequences=True)(inp_2)
dense_2 = TimeDistributed(Dense(dout))(lstm_2)
model_2 = Model(inputs=inp_2, outputs=dense_2)
model_2.summary()
model_2.compile(optimizer="rmsprop", loss="mae", sample_weight_mode="temporal")

model_2.set_weights(model_1_untrained_weights)

y_pred = model_2.predict(X2, verbose=0)
unmasked_loss = np.average(np.abs(y_true - y_pred))
masked_loss = np.average(np.abs(y_true - y_pred[y_pred != 0.0]))
weighted_loss = np.average(np.abs(y_true - y_pred), weights=sample_weight)
keras_loss = model_2.evaluate(X2, y_true, verbose=0)
keras_loss_weighted = model_2.evaluate(
    X2, y_true, sample_weight=sample_weight[..., 0], verbose=0
)
print("-- model 2 --")
print(f"unmasked loss: {unmasked_loss}")
print(f"masked loss: {masked_loss}")
print(f"weighted loss: {weighted_loss}")
print(f"evaluate with Keras: {keras_loss}")
print(f"evaluate with Keras (weighted): {keras_loss_weighted}")

model_1.fit(X2, y_true, epochs=epochs, verbose=0)
model_2.fit(X2, y_true, epochs=epochs, verbose=0, sample_weight=sample_weight[..., 0])

print("--- after training ---")

y_pred = model_1.predict(X2, verbose=0)
unmasked_loss = np.average(np.abs(y_true - y_pred))
masked_loss = np.average(np.abs(y_true - y_pred[y_pred != 0.0]))
weighted_loss = np.average(np.abs(y_true - y_pred), weights=sample_weight)
keras_loss = model_1.evaluate(X2, y_true, verbose=0)
keras_loss_weighted = model_1.evaluate(
    X2, y_true, sample_weight=sample_weight[..., 0], verbose=0
)
print("-- model 1 --")
print(f"unmasked loss: {unmasked_loss}")
print(f"masked loss: {masked_loss}")
print(f"weighted loss: {weighted_loss}")
print(f"evaluate with Keras: {keras_loss}")
print(f"evaluate with Keras (weighted): {keras_loss_weighted}")
np.testing.assert_approx_equal(unmasked_loss, masked_loss)
np.testing.assert_approx_equal(weighted_loss, keras_loss)
# keras_loss_weighted not equal to anything here

y_pred = model_2.predict(X2, verbose=0)
unmasked_loss = np.average(np.abs(y_true - y_pred))
masked_loss = np.average(np.abs(y_true - y_pred[y_pred != 0.0]))
weighted_loss = np.average(np.abs(y_true - y_pred), weights=sample_weight)
keras_loss = model_2.evaluate(X2, y_true, verbose=0)
keras_loss_weighted = model_2.evaluate(
    X2, y_true, sample_weight=sample_weight[..., 0], verbose=0
)
print("-- model 2 --")
print(f"unmasked loss: {unmasked_loss}")
print(f"masked loss: {masked_loss}")
print(f"weighted loss: {weighted_loss}")
print(f"evaluate with Keras: {keras_loss}")
print(f"evaluate with Keras (weighted): {keras_loss_weighted}")
np.testing.assert_approx_equal(unmasked_loss, masked_loss)
np.testing.assert_approx_equal(unmasked_loss, keras_loss)
np.testing.assert_approx_equal(weighted_loss, keras_loss_weighted)

### Now try with BiLSTMs

dout = 5

inp = Input(shape=(dx, dy))
lstm = Bidirectional(LSTM(dout, return_sequences=True), merge_mode="concat")(inp)
dense = TimeDistributed(Dense(1))(lstm)
model_3 = Model(inputs=inp, outputs=dense)
model_3.summary()
model_3.compile(optimizer="rmsprop", loss="mae", sample_weight_mode="temporal")

y_pred = model_3.predict(X2, verbose=0)
unmasked_loss = np.average(np.abs(y_true - y_pred))
masked_loss = np.average(np.abs(y_true - y_pred[y_pred != 0.0]))
weighted_loss = np.average(np.abs(y_true - y_pred), weights=sample_weight)
keras_loss = model_3.evaluate(X2, y_true, verbose=0)
keras_loss_weighted = model_3.evaluate(
    X2, y_true, sample_weight=sample_weight[..., 0], verbose=0
)
print(f"unmasked loss: {unmasked_loss}")
print(f"masked loss: {masked_loss}")
print(f"weighted loss: {weighted_loss}")
print(f"evaluate with Keras: {keras_loss}")
print(f"evaluate with Keras (weighted): {keras_loss_weighted}")
np.testing.assert_approx_equal(unmasked_loss, masked_loss)
np.testing.assert_approx_equal(unmasked_loss, keras_loss)
np.testing.assert_approx_equal(weighted_loss, keras_loss_weighted)

model_3_untrained_weights = model_3.get_weights()
model_3.fit(X2, y_true, epochs=epochs, verbose=0, sample_weight=sample_weight[..., 0])
model_3_final_weights = model_3.get_weights()

print("--- after training ---")

y_pred = model_3.predict(X2, verbose=0)
unmasked_loss = np.average(np.abs(y_true - y_pred))
masked_loss = np.average(np.abs(y_true - y_pred[y_pred != 0.0]))
weighted_loss = np.average(np.abs(y_true - y_pred), weights=sample_weight)
keras_loss = model_3.evaluate(X2, y_true, verbose=0)
keras_loss_weighted = model_3.evaluate(
    X2, y_true, sample_weight=sample_weight[..., 0], verbose=0
)
print(f"unmasked loss: {unmasked_loss}")
print(f"masked loss: {masked_loss}")
print(f"weighted loss: {weighted_loss}")
print(f"evaluate with Keras: {keras_loss}")
print(f"evaluate with Keras (weighted): {keras_loss_weighted}")
np.testing.assert_approx_equal(unmasked_loss, masked_loss)
np.testing.assert_approx_equal(unmasked_loss, keras_loss)
np.testing.assert_approx_equal(weighted_loss, keras_loss_weighted)

pprint([l1 - l2 for l1, l2 in zip(model_3_untrained_weights, model_3_final_weights)])
