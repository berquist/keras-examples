### TODO explain these

## eighth example: single LSTM layer that automatically incorporates the temporal dimension using
## `return_sequences`, ...

inp = Input(shape=(dx, dy))
mask = Masking(mask_value=mask_value)(inp)
lstm = LSTM(dout, return_sequences=True)(mask)
model = Model(inputs=inp, outputs=lstm)
model.summary()
model.compile(optimizer="rmsprop", loss="mae", sample_weight_mode="temporal")
y_pred = model.predict(X, verbose=0)
unmasked_loss = np.average(np.abs(y_true - y_pred))
masked_loss = np.average(np.abs(y_true - y_pred[y_pred != 0.0]))
weighted_loss = np.average(np.abs(y_true - y_pred), weights=sample_weight)
keras_loss = model.evaluate(X, y_true, verbose=0)
print(f"unmasked loss: {unmasked_loss}")
print(f"masked loss: {masked_loss}")
print(f"weighted loss: {weighted_loss}")
print(f"evaluate with Keras: {keras_loss}")

# weights = model.get_weights()

inp = Input(shape=(dx, dy))
lstm = LSTM(dout, return_sequences=True)(inp)
dense = TimeDistributed(Dense(dout, activation="sigmoid"))(lstm)
model = Model(inputs=inp, outputs=dense)
model.summary()
model.compile(optimizer="rmsprop", loss="mae", sample_weight_mode="temporal")
y_pred = model.predict(X, verbose=0)
unmasked_loss = np.average(np.abs(y_true - y_pred))
masked_loss = np.average(np.abs(y_true - y_pred[y_pred != 0.0]))
weighted_loss = np.average(np.abs(y_true - y_pred), weights=sample_weight)
keras_loss = model.evaluate(X, y_true, verbose=0)
keras_loss_weighted = model.evaluate(X, y_true, sample_weight=sample_weight[..., 0], verbose=0)
print(f"unmasked loss: {unmasked_loss}")
print(f"masked loss: {masked_loss}")
print(f"weighted loss: {weighted_loss}")
print(f"evaluate with Keras: {keras_loss}")
print(f"evaluate with Keras (weighted): {keras_loss_weighted}")

# weights = model.get_weights()
