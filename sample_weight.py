import numpy as np
from keras import Model
from keras.layers import Dense, Input, Masking, TimeDistributed

from common import mae


np.set_printoptions(linewidth=250)

## https://github.com/keras-team/keras/issues/7695

n_samples = 4
dx = 2
dy = 3
dout = 1
mask_value = -1
epochs = 40

## zeroth example: single dense layer, no temporal dimension, no sample weights, no masking

X = np.random.randint(5, size=(n_samples, dy))
y_true = np.ones((n_samples, dout))

inp = Input(shape=(dy,))
dense = Dense(dout)(inp)
model = Model(inputs=inp, outputs=dense)
model.summary()
model.compile(optimizer="rmsprop", loss="mae")
y_pred = model.predict(X, verbose=0)
unmasked_loss = np.average(np.abs(y_true - y_pred))
# Why evaluate against `y_true`? The model thinks that `y_pred` is the correct answer for `X` with
# its current weights, so of course its loss will be zero.
keras_loss = model.evaluate(X, y_true, verbose=0)
print(unmasked_loss - keras_loss)
np.testing.assert_approx_equal(unmasked_loss, keras_loss)

## first example: single dense layer, no temporal dimension, only use the first sample
# no need for a new model, only the prediction is changing

sample_weight = np.ones_like(y_true)
sample_weight[1:] = 0
weighted_loss = np.average(np.abs(y_true - y_pred), weights=sample_weight)
keras_loss_weighted = model.evaluate(
    X, y_true, sample_weight=sample_weight[:, 0], verbose=0
)
print(weighted_loss - keras_loss_weighted)
np.testing.assert_approx_equal(weighted_loss, keras_loss_weighted)

## second example: single dense layer, temporal dimension without TimeDistributed, no sample
## weights, no masking

X = np.random.randint(5, size=(n_samples, dx, dy))
y_true = np.ones((n_samples, dx, dout))

inp = Input(shape=(dx, dy))
dense = Dense(dout)(inp)
model = Model(inputs=inp, outputs=dense)
model.summary()
model.compile(optimizer="rmsprop", loss="mae", sample_weight_mode="temporal")
y_pred = model.predict(X, verbose=0)
unmasked_loss = mae(y_true, y_pred, mask=False)
keras_loss = model.evaluate(X, y_true, verbose=0)
print(unmasked_loss - keras_loss)
np.testing.assert_approx_equal(unmasked_loss, keras_loss)

weights = model.get_weights()

## third example: single dense layer, temporal dimension without TimeDistributed, only use the first
## sample, no masking

sample_weight = np.ones_like(y_true)
sample_weight[1:] = 0
unmasked_loss = mae(y_true, y_pred, weights=sample_weight, mask=False)
keras_loss = model.evaluate(X, y_true, sample_weight=sample_weight[..., 0], verbose=0)
print(weighted_loss - keras_loss_weighted)
np.testing.assert_approx_equal(weighted_loss, keras_loss_weighted)

## fourth example: single dense layer, temporal dimension with TimeDistributed, no sample weights,
## no masking

# X = np.random.randint(5, size=(n_samples, dx, dy))
# X[1, 0] = mask_value
# sample_weight = np.ones(shape=(n_samples, dx))
# sample_weight[1, 0] = 0

inp = Input(shape=(dx, dy))
dense = TimeDistributed(Dense(dout))(inp)
model = Model(inputs=inp, outputs=dense)
model.summary()
model.compile(optimizer="rmsprop", loss="mae", sample_weight_mode="temporal")
model.set_weights(weights)
y_pred = model.predict(X, verbose=0)
unmasked_loss = mae(y_true, y_pred, mask=False)
keras_loss = model.evaluate(X, y_true, verbose=0)
print(unmasked_loss - keras_loss)
np.testing.assert_approx_equal(unmasked_loss, keras_loss)

## fifth example: single dense layer, temporal dimension with TimeDistributed, only use the first
## sample, no masking

sample_weight = np.ones_like(y_true)
sample_weight[1:] = 0
weighted_loss = mae(y_true, y_pred, weights=sample_weight, mask=False)
keras_loss_weighted = model.evaluate(
    X, y_true, sample_weight=sample_weight[..., 0], verbose=0
)
print(weighted_loss - keras_loss_weighted)
np.testing.assert_approx_equal(weighted_loss, keras_loss_weighted)

##############################################################################

## sixth example: single dense layer, temporal dimension with TimeDistributed, no sample weights,
## with masking

X[2, 0] = mask_value
X[3, 1] = mask_value

inp = Input(shape=(dx, dy))
mask = Masking(mask_value=mask_value)(inp)
dense = TimeDistributed(Dense(dout))(mask)
model_6 = Model(inputs=inp, outputs=dense)
model_6.summary()
model_6.compile(optimizer="rmsprop", loss="mae")
model_6.set_weights(weights)
y_pred = model_6.predict(X, verbose=0)
unmasked_loss = mae(y_true, y_pred, mask=False)
masked_loss = mae(y_true, y_pred, mask=True)
keras_loss = model_6.evaluate(X, y_true, verbose=0)
print(f"unmasked loss: {unmasked_loss}")
print(f"masked loss: {masked_loss}")
print(f"evaluate with Keras: {keras_loss}")
np.testing.assert_approx_equal(masked_loss, keras_loss)

## seventh example: single dense layer, temporal dimension with TimeDistributed, sample weights that
## mimic the above masked samples

sample_weight = np.ones_like(y_true)
sample_weight[2, 0] = 0
sample_weight[3, 1] = 0

inp = Input(shape=(dx, dy))
dense = TimeDistributed(Dense(dout))(inp)
model_7 = Model(inputs=inp, outputs=dense)
model_7.summary()
model_7.compile(optimizer="rmsprop", loss="mae", sample_weight_mode="temporal")
model_7.set_weights(weights)
y_pred = model_7.predict(X, verbose=0)
unmasked_loss = mae(y_true, y_pred, mask=False)
masked_loss = mae(y_true, y_pred, mask=True)
weighted_loss = mae(y_true, y_pred, mask=False, weights=sample_weight)
keras_loss_weighted = model_7.evaluate(
    X, y_true, sample_weight=sample_weight[..., 0], verbose=0
)
print(f"unmasked loss: {unmasked_loss}")
print(f"masked loss: {masked_loss}")
print(f"weighted loss: {weighted_loss}")
print(f"evaluate with Keras (weighted): {keras_loss_weighted}")
np.testing.assert_approx_equal(unmasked_loss, masked_loss)
np.testing.assert_approx_equal(weighted_loss, keras_loss_weighted)
