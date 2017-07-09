import pickle
from configures import *
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Activation
from keras.utils import np_utils
import numpy as np
import time
from keras.models import load_model
import os


def shuffle_data(x, y):
    index = np.random.permutation(len(x))
    x = x[index]
    y = y[index]
    return x, y


def load_data(train_split):
    with open(sequences_path, 'rb') as f:
        sequences = pickle.load(f)

    with open(next_words_path, 'rb') as f:
        next_words = pickle.load(f)

    sequences, next_words = shuffle_data(sequences, next_words)

    train_len = int(train_split * len(sequences))
    y = np_utils.to_categorical(next_words)

    x_train = sequences[: train_len]
    x_test = sequences[train_len:]
    y_train = y[: train_len]
    y_test = y[train_len:]
    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):
    model = Sequential()
    model.add(Embedding(151, 200, input_length=MAX_SENTENCE_LEN, mask_zero=True))
    model.add(Dropout(0.2))
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(200, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(151, activation='relu'))
    model.add(Dense(151))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=300, epochs=100, validation_split=0.15, shuffle=True)
    model.save(model_path + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '.hdf5')


def data_predict(x_test):
    model = load_model(model_path + '2017-06-10 12:04:28.hdf5')
    pre_data = list()
    for each in x_test:
        each = np.array(each).reshape((1, MAX_SENTENCE_LEN))
        y_pre = model.predict(each)
        index = np.argmax(y_pre)
        print(index)
        pre_data.append(index)
    with open(result_path + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '.pkl', 'wb') as f:
        pre_data = np.array(pre_data, dtype='int32')
        pickle.dump(pre_data, f)


def test_accuracy(y_test):
    results = os.listdir(result_path)
    accuracy_list = list()
    for result in results:
        if result[-3:] == 'pkl':
            with open(result_path + result, 'rb') as f:
                pre_data = pickle.load(f)
                pre_data = np.asarray(pre_data, dtype='int32')
                print(pre_data)
                print(y_test)
                y_true = pre_data == y_test
                print(y_true)
                accuracy = np.sum(y_true == True) / len(y_true)
                accuracy_list.append(accuracy)
    return accuracy_list


def trans_y(y):
    y_ = list()
    for each in y:
        index = np.argmax(each)
        y_.append(index)
    y_ = np.array(y_, dtype='int32')
    return y_


def filter_x_y(x_array, y_array, n):
    _index = list(map(lambda x: True if np.sum(x == 0) == MAX_SENTENCE_LEN - n else False, x_array))

    return x_array[_index], y_array[_index]

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data(.9)
    # x_test, y_test = filter_x_y(x_test, y_test, 8)
    # print(x_test)
    train_model(x_train, y_train)
    # data_predict(x_test)
    # y_test = trans_y(y_test)
    # accuracy_list = test_accuracy(y_test)
    # print(accuracy_list)
