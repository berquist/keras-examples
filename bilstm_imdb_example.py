"""Adapted from https://stackoverflow.com/a/53664580/3249688"""

import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb


def load_data(n_unique_words=None, maxlen=None):
    # https://stackoverflow.com/a/56243777/3249688
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=n_unique_words)
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":

    np_load_old = np.load
    np.load = lambda *args, **kwargs: np_load_old(*args, allow_pickle=True, **kwargs)

    N_UNIQUE_WORDS = 10000  # cut texts after this number of words
    MAXLEN = 200
    BATCH_SIZE = 1024

    x_train, y_train, x_test, y_test = load_data(
        n_unique_words=N_UNIQUE_WORDS, maxlen=MAXLEN
    )

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

    # Remove restriction on the number of unique words.
    (x_train2, y_train2), (x_test2, y_test2) = imdb.load_data(num_words=None)
    x_train2, y_train2, x_test2, y_test2 = load_data(n_unique_words=None, maxlen=MAXLEN)
    N_UNIQUE_WORDS = max(np.max(np.max(x_train2)), np.max(np.max(x_test2)))
    # This is probably too large.

    model2 = Sequential(
        [
            # (number of possible tokens, dimension of embedding space)
            Embedding(N_UNIQUE_WORDS, 128, input_length=MAXLEN),
            Bidirectional(LSTM(64)),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    model2.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    print("Train...")
    model2.fit(x_train2, y_train2, batch_size=BATCH_SIZE, epochs=4, validation_data=[x_test2, y_test2])
    print("Evaluate...")
    model2.evaluate(x_test2, y_test2, batch_size=BATCH_SIZE)

    # Increase the size of the embedding space.
    model3 = Sequential(
        [
            # (number of possible tokens, dimension of embedding space)
            Embedding(N_UNIQUE_WORDS, 768, input_length=MAXLEN),
            Bidirectional(LSTM(64)),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    model3.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    print("Train...")
    model3.fit(x_train2, y_train2, batch_size=BATCH_SIZE, epochs=4, validation_data=[x_test2, y_test2])
    print("Evaluate...")
    model3.evaluate(x_test2, y_test2, batch_size=BATCH_SIZE)

    np.load = np_load_old
