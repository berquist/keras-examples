import numpy as np
from keras import Model
from keras.layers import Dense, Input, TimeDistributed

from common import set_model_weights_to_unity, get_gradient, mae


np.set_printoptions(linewidth=250)

# n_samples = 4
n_samples = 1
dx = 2
dy = 3
dout = 1
mask_value = -1
epochs = 40

X = np.random.randint(10, size=(n_samples, dx, dy))
y_true = np.ones((n_samples, dx, dout))

# X[2, 0] = mask_value
# X[3, 1] = mask_value

sample_weight = np.ones_like(y_true)
# sample_weight[2, 0] = 0
# sample_weight[3, 1] = 0
sample_weight[0, 0] = 0

inp = Input(shape=(dx, dy))
dense = TimeDistributed(Dense(dout))(inp)
model = Model(inputs=inp, outputs=dense)
model.summary()
model.compile(optimizer="rmsprop", loss="mae", sample_weight_mode="temporal")
set_model_weights_to_unity(model)

y_pred = model.predict(X, verbose=0)
unmasked_loss = mae(y_true, y_pred, mask=False)
masked_loss = mae(y_true, y_pred, mask=True)
weighted_loss = mae(y_true, y_pred, mask=False, weights=sample_weight)
keras_loss = model.evaluate(X, y_true, verbose=0)
keras_loss_weighted = model.evaluate(
    X, y_true, sample_weight=sample_weight[..., 0], verbose=0
)
print(f"unmasked loss: {unmasked_loss}")
print(f"masked loss: {masked_loss}")
print(f"weighted loss: {weighted_loss}")
print(f"evaluate with Keras: {keras_loss}")
print(f"evaluate with Keras (weighted): {keras_loss_weighted}")
np.testing.assert_approx_equal(unmasked_loss, masked_loss)
np.testing.assert_approx_equal(weighted_loss, keras_loss_weighted)

print("samples:")
print(X)
print("sample weights:")
print(sample_weight)
print("gradient (weighted):")
print(get_gradient(model)([X, sample_weight[..., 0], y_true, 0]))
print("gradient (all samples):")
print(get_gradient(model)([X, np.ones_like(sample_weight[..., 0]), y_true, 0]))
