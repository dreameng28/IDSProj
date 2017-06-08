from configures import *
import pickle
import numpy as np


def get_sen_and_next(sentence, seq_len, sequences, next_words):
    for i in range(0, len(sentence) - seq_len):
        seq = [0] * (MAX_SENTENCE_LEN - seq_len) + sentence[i: i + seq_len]
        next_word = sentence[i + seq_len]
        sequences.append(seq)
        next_words.append(next_word)


def data_split():
    sequences = list()
    next_words = list()
    with open(sentence_path, 'rb') as f:
        sentences = pickle.load(f)
    for sentence in sentences:
        max_sen_len = min(len(sentence), MAX_SENTENCE_LEN)
        for sen_len in range(1, max_sen_len + 1):
            get_sen_and_next(sentence, sen_len, sequences, next_words)

    print(sequences)
    print(next_words)
    print(len(sequences))
    print(len(next_words))
    sequences = np.asarray(sequences, dtype='int32')
    next_words = np.asarray(next_words, dtype='int32')

    with open(sequences_path, 'wb') as f:
        pickle.dump(sequences, f)

    with open(next_words_path, 'wb') as f:
        pickle.dump(next_words, f)


if __name__ == '__main__':
    data_split()
