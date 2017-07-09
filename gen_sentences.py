from data_view import DataView
import os
from configures import *
import pandas as pd
import pickle


class WordInfo:
    def __init__(self, order_start, time_start, times, src_ips, des_ips):
        self.order_start = order_start
        self.time_start = time_start
        self.times = times
        self.src_ips = src_ips
        self.des_ips = des_ips


def gen_sentence(df):
    words = dict()
    i = 0
    while i < len(df):
        key = tuple(df.iloc[i][[theme]].values)
        # print(key)
        if key in words.keys():
            words[key].times += 1
            words[key].src_ips.add(df.at[i, src_ip])
            words[key].des_ips.add(df.at[i, des_ip])
        else:
            words[key] = WordInfo(df.at[i, order_num], df.at[i, occur_time], 1, {df.at[i, src_ip]}, {df.at[i, des_ip]})
        i += 1
    sorted_words = sorted(words.items(), key=lambda d: int(d[1].order_start))
    event_words = list(map(lambda x: theme_to_int[x[0][0]], sorted_words))
    print(event_words)
    ip_words = list(map(lambda x: [x[1].src_ips, x[1].des_ips], sorted_words))
    # ip_words = sum(ip_words, [])
    # ip_words = process_ip_sequence(ip_words)
    print(ip_words)
    return event_words, ip_words


def gen_all_sentences():
    relevant_ip_files = os.listdir(relevant_ip_path)
    sentences = list()
    ip_sentences = list()
    for each in relevant_ip_files:
        if each[-3:] == 'csv':
            df = pd.read_csv(relevant_ip_path + each)
            single_sentence, single_ip_sentence = gen_sentence(df)
            sentences.append(single_sentence)
            ip_sentences.append(single_ip_sentence)
    with open(sentence_path, 'wb') as f:
        pickle.dump(sentences, f)
    with open(ip_sentence_path, 'wb') as f:
        pickle.dump(ip_sentences, f)


if __name__ == '__main__':
    dv = DataView()
    themes = list(dv.theme_set)
    theme_to_int = dict((c, i + 1) for i, c in enumerate(themes))
    int_to_theme = dict((i + 1, c) for i, c in enumerate(themes))
    with open(theme2int_path, 'wb') as f:
        pickle.dump(theme_to_int, f)
    with open(int2theme_path, 'wb') as f:
        pickle.dump(int_to_theme, f)
    print(theme_to_int)
    print(int_to_theme)
    gen_all_sentences()
