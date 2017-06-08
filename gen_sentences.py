from data_view import DataView
import os
from configures import *
import pandas as pd
import pickle

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


class WordInfo:
    def __init__(self, order_start, time_start, times):
        self.order_start = order_start
        self.time_start = time_start
        self.times = times


def gen_sentence(df):
    words = dict()
    i = 0
    while i < len(df):
        key = tuple(df.iloc[i][[theme]].values)
        if key in words.keys():
            words[key].times += 1
        else:
            words[key] = WordInfo(df.at[i, order_num], df.at[i, occur_time], 1)
        i += 1
    sorted_words = sorted(words.items(), key=lambda d: int(d[1].order_start))
    words = list(map(lambda x: theme_to_int[x[0][0]], sorted_words))
    print(words)
    return words


def gen_all_sentences():
    relevant_ip_files = os.listdir(relevant_ip_path)
    sentences = list()
    for each in relevant_ip_files:
        if each[-3:] == 'csv':
            df = pd.read_csv(relevant_ip_path + each)
            single_sentence = gen_sentence(df)
            sentences.append(single_sentence)
    with open(sentence_path, 'wb') as f:
        pickle.dump(sentences, f)


if __name__ == '__main__':
    gen_all_sentences()
