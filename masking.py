import numpy as np

from keras import Model
from keras.layers import Bidirectional, Input, LSTM, Masking, TimeDistributed


if __name__ == "__main__":

    ## Adapted from https://stackoverflow.com/q/47057361/. Do the Keras built-in loss functions
    ## correctly handle masking?

    max_sentence_length = 5
    character_number = 2

    input_tensor = Input(shape=(max_sentence_length, character_number))
    masked_input = Masking(mask_value=0)(input_tensor)
    output = LSTM(3, return_sequences=True)(masked_input)
    model = Model(input_tensor, output)
    model.compile(optimizer="adam", loss="mae")
    model.summary()

    X = np.array(
        [
            [[0, 0], [0, 0], [1, 0], [0, 1], [0, 1]],
            [[0, 0], [0, 1], [1, 0], [0, 1], [0, 1]],
        ]
    )
    y_true = np.ones((2, max_sentence_length, 3))
    y_pred = model.predict(X)
    print(y_pred)
    print(y_pred.shape)

    # See if the loss computed by model.evaluate() is equal to the masked loss
    unmasked_loss = np.abs(1 - y_pred).mean()
    masked_loss = np.abs(1 - y_pred[y_pred != 0.0]).mean()
    keras_loss = model.evaluate(X, y_true, verbose=0)
    print(f"unmasked loss: {unmasked_loss}")
    print(f"masked loss: {masked_loss}")
    print(f"evaluate with Keras: {keras_loss}")
    # Why is this zero?
    # print(f"evaluate with Keras: {model.evaluate(X, y_pred, verbose=0)}")

    ## try again using a non-zero mask value

    masked_input = Masking(mask_value=8)(input_tensor)
    output = LSTM(3, return_sequences=True)(masked_input)
    model = Model(input_tensor, output)
    model.compile(optimizer="adam", loss="mae")
    model.summary()

    X = np.array(
        [
            [[8, 8], [8, 8], [1, 0], [0, 1], [0, 1]],
            [[8, 8], [0, 1], [1, 0], [0, 1], [0, 1]],
        ]
    )
    y_true = np.ones((2, max_sentence_length, 3))
    y_pred = model.predict(X)
    print(y_pred)
    print(y_pred.shape)

    # See if the loss computed by model.evaluate() is equal to the masked loss
    unmasked_loss = np.abs(1 - y_pred).mean()
    masked_loss = np.abs(1 - y_pred[y_pred != 0.0]).mean()
    keras_loss = model.evaluate(X, y_true, verbose=0)
    print(f"unmasked loss: {unmasked_loss}")
    print(f"masked loss: {masked_loss}")
    print(f"evaluate with Keras: {keras_loss}")

    ## try again using a floating-point mask value

    masked_input = Masking(mask_value=0.0)(input_tensor)
    output = LSTM(3, return_sequences=True)(masked_input)
    model = Model(input_tensor, output)
    model.compile(optimizer="adam", loss="mae")
    model.summary()

    X = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        ]
    )
    y_true = np.ones((2, max_sentence_length, 3))
    y_pred = model.predict(X)
    print(y_pred)
    print(y_pred.shape)

    unmasked_loss = np.abs(1 - y_pred).mean()
    masked_loss = np.abs(1 - y_pred[y_pred != 0.0]).mean()
    keras_loss = model.evaluate(X, y_true, verbose=0)
    print(f"unmasked loss: {unmasked_loss}")
    print(f"masked loss: {masked_loss}")
    print(f"evaluate with Keras: {keras_loss}")

    ## try again using a BiLSTM

    masked_input = Masking(mask_value=0.0)(input_tensor)
    output = Bidirectional(LSTM(3, return_sequences=True), merge_mode="sum")(
        masked_input
    )
    model = Model(input_tensor, output)
    model.compile(optimizer="adam", loss="mae")
    model.summary()

    X = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        ]
    )
    y_true = np.ones((2, max_sentence_length, 3))
    y_pred = model.predict(X)
    print(y_pred)
    print(y_pred.shape)

    unmasked_loss = np.abs(1 - y_pred).mean()
    masked_loss = np.abs(1 - y_pred[y_pred != 0.0]).mean()
    keras_loss = model.evaluate(X, y_true, verbose=0)
    print(f"unmasked loss: {unmasked_loss}")
    print(f"masked loss: {masked_loss}")
    print(f"evaluate with Keras: {keras_loss}")

    ## try again using TimeDistributed

    masked_input = Masking(mask_value=0.0)(input_tensor)
    output = TimeDistributed(
        Bidirectional(LSTM(3, return_sequences=True), merge_mode="sum")
    )(masked_input)
    model = Model(input_tensor, output)
    model.compile(optimizer="adam", loss="mae")
    model.summary()

    X = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        ]
    )
    y_true = np.ones((2, max_sentence_length, 3))
    y_pred = model.predict(X)
    print(y_pred)
    print(y_pred.shape)

    unmasked_loss = np.abs(1 - y_pred).mean()
    masked_loss = np.abs(1 - y_pred[y_pred != 0.0]).mean()
    keras_loss = model.evaluate(X, y_true, verbose=0)
    print(f"unmasked loss: {unmasked_loss}")
    print(f"masked loss: {masked_loss}")
    print(f"evaluate with Keras: {keras_loss}}")
