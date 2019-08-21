"""Adapted from https://stackoverflow.com/a/53664580/3249688"""

import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb



if __name__ == "__main__":

    N_UNIQUE_WORDS = 10000  # cut texts after this number of words
    MAXLEN = 200
    BATCH_SIZE = 1024

    # https://stackoverflow.com/a/56243777/3249688
    np_load_old = np.load
    np.load = lambda *args, **kwargs: np_load_old(*args, allow_pickle=True, **kwargs)

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=N_UNIQUE_WORDS)
    x_train = sequence.pad_sequences(x_train, maxlen=MAXLEN)
    x_test = sequence.pad_sequences(x_test, maxlen=MAXLEN)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    np.load = np_load_old

    model = Sequential(
        [
            Embedding(N_UNIQUE_WORDS, 128, input_length=MAXLEN),
            Bidirectional(LSTM(64)),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    print("Train...")
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=4,
        validation_data=[x_test, y_test],
    )
    print("Evaluate...")
    model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
