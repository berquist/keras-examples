import numpy as np
from keras import Model
from keras.layers import (
    Bidirectional,
    Dense,
    Embedding,
    Input,
    LSTM,
    Masking,
    TimeDistributed,
)


if __name__ == "__main__":

    num_classes = 2
    num_symbols = 2
    datapoints = 4
    seq_len = 5
    x = np.random.randint(num_symbols, size=(datapoints, seq_len))
    y = np.random.randint(num_classes, size=(datapoints, seq_len))
    y_one_hot = np.zeros((datapoints, seq_len, num_classes))

    for i in range(datapoints):
        for j in range(seq_len):
            y_one_hot[i][j][y[i][j]] = 1

    embedding_size = 10
    embedding_weights = np.zeros((num_symbols, embedding_size))

    for i in range(num_symbols):
        embedding_weights[i] = np.random.rand(embedding_size)

    input_layer = Input(shape=(seq_len,), dtype=np.int32)
    embedding = Embedding(num_symbols, embedding_size, input_length=seq_len)
    embedded_input = embedding(input_layer)
    mask = Masking(mask_value=0)(embedded_input)
    bidirect = Bidirectional(LSTM(100, return_sequences=True))(mask)
    final = TimeDistributed(Dense(num_classes, activation="softmax"))(bidirect)

    model = Model(inputs=[input_layer], outputs=[final])
    model.compile(
        loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
    )
    model.fit(x, y_one_hot)

    print(x)
    print(y)
    print(y_one_hot)
    print(model.predict(x))
    model.save("issue_3676.model")
