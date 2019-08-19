from functools import reduce
import re
import tarfile

import numpy as np

from keras.utils.data_utils import get_file
from keras.layers import recurrent, Embedding, Dense, concatenate, Input
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences


def tokenize(sent):
    """Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    """
    return [x.strip() for x in re.split(r"(\W+)+", sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    """Parse stories provided in the bAbi tasks format

    If only_supporting is true,
    only the sentences that support the answer are kept.
    """
    data = []
    story = []
    for line in lines:
        line = line.decode("utf-8").strip()
        nid, line = line.split(" ", 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if "\t" in line:
            q, a, supporting = line.split("\t")
            q = tokenize(q)
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append("")
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    """Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.

    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    """
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    return [
        (flatten(story), q, answer)
        for story, q, answer in data
        if not max_length or len(flatten(story)) < max_length
    ]


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    xs = []
    xqs = []
    ys = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    return (
        pad_sequences(xs, maxlen=story_maxlen),
        pad_sequences(xqs, maxlen=query_maxlen),
        np.array(ys),
    )


if __name__ == "__main__":

    RNN = recurrent.LSTM
    EMBED_HIDDEN_SIZE = 50
    SENT_HIDDEN_SIZE = 100
    QUERY_HIDDEN_SIZE = 100
    BATCH_SIZE = 32
    EPOCHS = 20
    print(
        "RNN / Embed / Sent / Query = {}, {}, {}, {}".format(
            RNN, EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE, QUERY_HIDDEN_SIZE
        )
    )

    try:
        path = get_file(
            "babi-tasks-v1-2.tar.gz",
            origin="https://s3.amazonaws.com/text-datasets/"
            "babi_tasks_1-20_v1-2.tar.gz",
        )
    except:
        print(
            "Error downloading dataset, please download it manually:\n"
            "$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2"
            ".tar.gz\n"
            "$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz"
        )
        raise

    # Default QA1 with 1000 samples
    # challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
    # QA1 with 10,000 samples
    # challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
    # QA2 with 1000 samples
    challenge = "tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt"
    # QA2 with 10,000 samples
    # challenge = 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt'
    with tarfile.open(path) as tar:
        train = get_stories(tar.extractfile(challenge.format("train")))
        test = get_stories(tar.extractfile(challenge.format("test")))

    vocab = set()
    for story, q, answer in train + test:
        vocab |= set(story + q + [answer])
    vocab = sorted(vocab)

    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    story_maxlen = max(map(len, (x for x, _, _ in train + test)))
    query_maxlen = max(map(len, (x for _, x, _ in train + test)))

    x, xq, y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
    tx, txq, ty = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

    print("vocab = {}".format(vocab))
    print("x.shape = {}".format(x.shape))
    print("xq.shape = {}".format(xq.shape))
    print("y.shape = {}".format(y.shape))
    print("story_maxlen, query_maxlen = {}, {}".format(story_maxlen, query_maxlen))

    print("Build model...")

    sentence = Input(shape=(story_maxlen,), dtype="int32")
    encoded_sentence = Embedding(vocab_size, EMBED_HIDDEN_SIZE)(sentence)
    encoded_sentence = RNN(SENT_HIDDEN_SIZE)(encoded_sentence)

    question = Input(shape=(query_maxlen,), dtype="int32")
    encoded_question = Embedding(vocab_size, EMBED_HIDDEN_SIZE)(question)
    encoded_question = RNN(QUERY_HIDDEN_SIZE)(encoded_question)

    merged = concatenate([encoded_sentence, encoded_question])
    preds = Dense(vocab_size, activation="softmax")(merged)

    model = Model([sentence, question], preds)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    print("Training")
    model.fit([x, xq], y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.05)

    print("Evaluation")
    loss, acc = model.evaluate([tx, txq], ty, batch_size=BATCH_SIZE)
    print("Test loss / test accuracy = {:.4f} / {:.4f}".format(loss, acc))
    # [1.7842422828674316, 0.232]

    ######################################################################

    from keras.layers import Bidirectional

    # replace LSTMs with BiLSTMs of half size, summing the Bi-LSTM output
    # [1.7144300785064697, 0.288]

    sentence2 = Input(shape=(story_maxlen,), dtype="int32")
    encoded_sentence2 = Embedding(vocab_size, EMBED_HIDDEN_SIZE)(sentence2)
    encoded_sentence2 = Bidirectional(RNN(SENT_HIDDEN_SIZE // 2), merge_mode="sum")(encoded_sentence2)

    question2 = Input(shape=(query_maxlen,), dtype="int32")
    encoded_question2 = Embedding(vocab_size, EMBED_HIDDEN_SIZE)(question2)
    encoded_question2 = Bidirectional(RNN(QUERY_HIDDEN_SIZE // 2), merge_mode="sum")(encoded_question2)

    merged2 = concatenate([encoded_sentence2, encoded_question2])
    preds2 = Dense(vocab_size, activation="softmax")(merged2)

    model2 = Model([sentence2, question2], preds2)
    model2.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # replace LSTMs with BiLSTMs of half size, concatentating the Bi-LSTM output
    # [1.6985360298156738, 0.291]

    sentence3 = Input(shape=(story_maxlen,), dtype="int32")
    encoded_sentence3 = Embedding(vocab_size, EMBED_HIDDEN_SIZE)(sentence3)
    encoded_sentence3 = Bidirectional(RNN(SENT_HIDDEN_SIZE // 2), merge_mode="concat")(encoded_sentence3)

    question3 = Input(shape=(query_maxlen,), dtype="int32")
    encoded_question3 = Embedding(vocab_size, EMBED_HIDDEN_SIZE)(question3)
    encoded_question3 = Bidirectional(RNN(QUERY_HIDDEN_SIZE // 2), merge_mode="concat")(encoded_question3)

    merged3 = concatenate([encoded_sentence3, encoded_question3])
    preds3 = Dense(vocab_size, activation="softmax")(merged3)

    model3 = Model([sentence3, question3], preds3)
    model3.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # same as above but much smaller LSTMs
    # [1.742978988647461, 0.25]

    sentence4 = Input(shape=(story_maxlen,), dtype="int32")
    encoded_sentence4 = Embedding(vocab_size, EMBED_HIDDEN_SIZE)(sentence4)
    encoded_sentence4 = Bidirectional(RNN(SENT_HIDDEN_SIZE // 5), merge_mode="concat")(encoded_sentence4)

    question4 = Input(shape=(query_maxlen,), dtype="int32")
    encoded_question4 = Embedding(vocab_size, EMBED_HIDDEN_SIZE)(question4)
    encoded_question4 = Bidirectional(RNN(QUERY_HIDDEN_SIZE // 5), merge_mode="concat")(encoded_question4)

    merged4 = concatenate([encoded_sentence4, encoded_question4])
    preds4 = Dense(vocab_size, activation="softmax")(merged4)

    model4 = Model([sentence4, question4], preds4)
    model4.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
