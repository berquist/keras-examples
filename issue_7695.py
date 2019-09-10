from keras.layers import Embedding, LSTM, Dense, Input, Masking
from keras.layers.wrappers import Bidirectional
from keras.models import Model
import numpy as np
import tensorflow as tf

## https://github.com/keras-team/keras/issues/7695

vec = np.random.randn(3, 5)
inp = Input((3,))
x = Masking(mask_value=-1.0)(inp)
x = Embedding(3, 5, weights=[vec], input_length=3, trainable=False)(x)
x = Bidirectional(LSTM(10, return_sequences=True))(x)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(x, {inp: [[0, 2, -1], [1, -1, -1]]}))
