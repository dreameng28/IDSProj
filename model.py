from keras.layers import Input, Embedding, LSTM, Dense, Dropout, dot
from keras.models import Model, Sequential
from configures import *
import numpy as np
import pickle
from keras.utils import np_utils
import time


def shuffle_data(x1, x2, y1, y2):
    index = np.random.permutation(len(x1))
    x1 = x1[index]
    y1 = y1[index]
    x2 = x2[index]
    y2 = y2[index]
    return x1, x2, y1, y2


def load_data(train_split):
    with open(sequences_path, 'rb') as f:
        sequences = pickle.load(f)

    with open(next_words_path, 'rb') as f:
        next_words = pickle.load(f)

    with open(ip_sequences_path, 'rb') as f:
        ip_sequences = pickle.load(f)

    with open(next_src_des_ips_path, 'rb') as f:
        next_src_des_ips = pickle.load(f)

    sequences, ip_sequences, next_words, next_src_des_ips = \
        shuffle_data(sequences, ip_sequences, next_words, next_src_des_ips)

    next_src_ips = next_src_des_ips[:, 0]
    next_des_ips = next_src_des_ips[:, 1]

    train_len = int(train_split * len(sequences))
    y_event = np_utils.to_categorical(next_words)
    y_src = np_utils.to_categorical(next_src_ips, 23)
    y_des = np_utils.to_categorical(next_des_ips, 23)

    x_event_train = sequences[: train_len]
    x_event_test = sequences[train_len:]
    x_src_ip_train = ip_sequences[: train_len, 0::2]
    x_src_ip_test = ip_sequences[train_len:, 0::2]
    x_des_ip_train = ip_sequences[: train_len, 1::2]
    x_des_ip_test = ip_sequences[train_len:, 1::2]
    y_event_train = y_event[: train_len]
    y_event_test = y_event[train_len:]
    y_src_ip_train = y_src[: train_len]
    y_src_ip_test = y_src[train_len:]
    y_des_ip_train = y_des[: train_len]
    y_des_ip_test = y_des[train_len:]

    return ([x_event_train, x_src_ip_train, x_des_ip_train], [y_event_train, y_src_ip_train, y_des_ip_train]), \
           ([x_event_test, x_src_ip_test, x_des_ip_test], [y_event_test, y_src_ip_test, y_des_ip_test])

event_input = Input((10,))
src_ip_input = Input((10,))
des_ip_input = Input((10,))

m1 = Sequential()
m1.add(Embedding(151, 200, input_length=MAX_SENTENCE_LEN, mask_zero=True))
m1.add(Dropout(0.2))

event_embedding = m1(event_input)

m2 = Sequential()
m2.add(Embedding(23, 200, input_length=MAX_SENTENCE_LEN, mask_zero=True))
m2.add(Dropout(0.2))

src_ip_embedding = m2(src_ip_input)
des_ip_embedding = m2(des_ip_input)

event_output = LSTM(200, return_sequences=True, dropout=0.2)(event_embedding)
event_output = LSTM(200, return_sequences=True, dropout=0.2)(event_output)
event_output = LSTM(200, return_sequences=False, dropout=0.2)(event_output)
event_output = Dense(151, activation='relu')(event_output)
event_output = Dense(151, activation='softmax')(event_output)

src_ip_event_info = dot([src_ip_embedding, event_embedding], (2, 2))
des_ip_event_info = dot([des_ip_embedding, event_embedding], (2, 2))

m3 = Sequential()
m3.add(LSTM(200, return_sequences=True, dropout=0.2, input_shape=(10, 10)))
m3.add(LSTM(200, return_sequences=True, dropout=0.2))
m3.add(LSTM(200, return_sequences=False, dropout=0.2))
m3.add(Dense(23, activation='relu'))
m3.add(Dense(23, activation='softmax'))

src_ip_output = m3(src_ip_event_info)
des_ip_output = m3(des_ip_event_info)

model = Model(inputs=[event_input, src_ip_input, des_ip_input], outputs=[event_output, src_ip_output, des_ip_output])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train, test = load_data(.9)

with open(test_data_path, 'wb') as f:
    pickle.dump(test, f)

model.fit(train[0], train[1], batch_size=300, epochs=100, validation_split=0.15, shuffle=True)
model.save(model_path + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '.hdf5')
