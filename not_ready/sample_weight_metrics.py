import numpy as np
from keras import Model
from keras.layers import Dense, Input, TimeDistributed

import common

np.set_printoptions(linewidth=250)


n_samples = 4
dx = 2
dy = 3
dout = 1
mask_value = -1
epochs = 40

X = np.random.randint(5, size=(n_samples, dx, dy))
y_true = np.ones((n_samples, dx, dout))
X[2, 0] = mask_value
X[3, 1] = mask_value
sample_weight = np.ones_like(y_true)
sample_weight[2, 0] = 0
sample_weight[3, 1] = 0


inp = Input(shape=(dx, dy))
dense = TimeDistributed(Dense(dout))(inp)
model_8 = Model(inputs=inp, outputs=dense)
model_8.summary()
model_8.compile(
    optimizer="rmsprop",
    loss="mae",
    sample_weight_mode="temporal",
    metrics=["binary_accuracy"],
)
# model_8.set_weights(weights)
y_pred = model_8.predict(X, verbose=0)
unmasked_loss = common.mae(y_true, y_pred, mask=False)
masked_loss = common.mae(y_true, y_pred, mask=True)
weighted_loss = common.mae(y_true, y_pred, mask=False, weights=sample_weight)
keras_loss = model_8.evaluate(X, y_true, verbose=0)
keras_loss_weighted = model_8.evaluate(
    X, y_true, sample_weight=sample_weight[..., 0], verbose=0
)
print(f"unmasked loss: {unmasked_loss}")
print(f"masked loss: {masked_loss}")
print(f"weighted loss: {weighted_loss}")
print(f"evaluate with Keras: {keras_loss}")
print(f"evaluate with Keras (weighted): {keras_loss_weighted}")
np.testing.assert_approx_equal(unmasked_loss, masked_loss)
np.testing.assert_approx_equal(unmasked_loss, keras_loss[0])
np.testing.assert_approx_equal(weighted_loss, keras_loss_weighted[0])

weights = model_8.get_weights()

inp = Input(shape=(dx, dy))
dense = TimeDistributed(Dense(dout))(inp)
model_9 = Model(inputs=inp, outputs=dense)
model_9.summary()
model_9.compile(
    optimizer="rmsprop",
    loss="mae",
    sample_weight_mode="temporal",
    weighted_metrics=["binary_accuracy"],
)
model_9.set_weights(weights)
y_pred = model_9.predict(X, verbose=0)
unmasked_loss = common.mae(y_true, y_pred, mask=False)
masked_loss = common.mae(y_true, y_pred, mask=True)
weighted_loss = common.mae(y_true, y_pred, mask=False, weights=sample_weight)
keras_loss = model_9.evaluate(X, y_true, verbose=0)
keras_loss_weighted = model_9.evaluate(
    X, y_true, sample_weight=sample_weight[..., 0], verbose=0
)
print(f"unmasked loss: {unmasked_loss}")
print(f"masked loss: {masked_loss}")
print(f"weighted loss: {weighted_loss}")
print(f"evaluate with Keras: {keras_loss}")
print(f"evaluate with Keras (weighted): {keras_loss_weighted}")
np.testing.assert_approx_equal(unmasked_loss, masked_loss)
np.testing.assert_approx_equal(unmasked_loss, keras_loss[0])
np.testing.assert_approx_equal(weighted_loss, keras_loss_weighted[0])

inp = Input(shape=(dx, dy))
dense = TimeDistributed(Dense(dout))(inp)
model_10 = Model(inputs=inp, outputs=dense)
model_10.summary()
model_10.compile(
    optimizer="rmsprop",
    loss="mae",
    sample_weight_mode="temporal",
    metrics=["binary_accuracy"],
    weighted_metrics=["binary_accuracy"],
)
model_10.set_weights(weights)
y_pred = model_10.predict(X, verbose=0)
unmasked_loss = common.mae(y_true, y_pred, mask=False)
masked_loss = common.mae(y_true, y_pred, mask=True)
weighted_loss = common.mae(y_true, y_pred, mask=False, weights=sample_weight)
keras_loss = model_10.evaluate(X, y_true, verbose=0)
keras_loss_weighted = model_10.evaluate(
    X, y_true, sample_weight=sample_weight[..., 0], verbose=0
)
print(f"unmasked loss: {unmasked_loss}")
print(f"masked loss: {masked_loss}")
print(f"weighted loss: {weighted_loss}")
print(f"evaluate with Keras: {keras_loss}")
print(f"evaluate with Keras (weighted): {keras_loss_weighted}")
np.testing.assert_approx_equal(unmasked_loss, masked_loss)
np.testing.assert_approx_equal(unmasked_loss, keras_loss[0])
np.testing.assert_approx_equal(weighted_loss, keras_loss_weighted[0])
