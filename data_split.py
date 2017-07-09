from configures import *
import pickle
import numpy as np


def process_ip_sequence(ip_sequence, threshold=0.7):
    new_ip_sequence = []
    for i in range(len(ip_sequence)):
        if i == 0:
            new_ip_sequence.append(1)
            continue
        similarity = threshold
        most_similar_index = i + 1
        for j in range(i):
            if len(ip_sequence[i] & ip_sequence[j]) / len(ip_sequence[i] | ip_sequence[j]) > similarity:
                similarity = len(ip_sequence[i] & ip_sequence[j]) / len(ip_sequence[i] | ip_sequence[j])
                most_similar_index = j + 1
        new_ip_sequence.append(most_similar_index)
    return new_ip_sequence


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


def get_ip_sen_and_next(sentence, seq_len, sequences, next_words):
    for i in range(0, len(sentence) - seq_len):
        s = sum(sentence[i: i + seq_len + 1], [])
        s = process_ip_sequence(s)
        seq = [0] * (MAX_IP_SENTENCE_LEN - seq_len * 2) + s[:-2]
        next_word = s[-2:]
        sequences.append(seq)
        next_words.append(next_word)


def ip_data_split():
    ip_sequences = list()
    next_src_des_ips = list()
    with open(ip_sentence_path, 'rb') as f:
        ip_sentences = pickle.load(f)
    for ip_sentence in ip_sentences:
        max_sen_len = min(len(ip_sentence), MAX_IP_SENTENCE_LEN // 2)
        for sen_len in range(1, max_sen_len + 1):
            get_ip_sen_and_next(ip_sentence, sen_len, ip_sequences, next_src_des_ips)

    print(ip_sequences)
    print(next_src_des_ips)
    print(len(ip_sequences))
    print(len(next_src_des_ips))
    ip_sequences = np.asarray(ip_sequences, dtype='int32')
    next_src_des_ips = np.asarray(next_src_des_ips, dtype='int32')
    print(ip_sequences[21409])
    print(next_src_des_ips[21409])

    with open(ip_sequences_path, 'wb') as f:
        pickle.dump(ip_sequences, f)

    with open(next_src_des_ips_path, 'wb') as f:
        pickle.dump(next_src_des_ips, f)

if __name__ == '__main__':
    data_split()
    ip_data_split()
