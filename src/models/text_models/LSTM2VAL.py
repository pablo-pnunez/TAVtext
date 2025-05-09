# -*- coding: utf-8 -*-
from src.Common import print_g
from src.models.text_models.VALModel import VALModel
from src.sequences.BaseSequence import BaseSequence, BaseSequenceXY

import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.models import KeyedVectors


class LSTM2VAL(VALModel):
    """ Predecir, a partir de una review codificada mendiante una LSTM utilizando las palabras del W2V, la nota de dicha review """
    def __init__(self, config, dataset, w2v_model):
        self.W2V_MODEL = w2v_model
        VALModel.__init__(self, config=config, dataset=dataset)

    def get_model(self):
        print_g("Loading w2v...")

        word_vectors = self.W2V_MODEL.MODEL.wv
        w2v_emb_size = word_vectors.vectors.shape[1]
        embedding_matrix = np.zeros((self.DATASET.DATA["VOCAB_SIZE"], w2v_emb_size))
        for word, i in self.DATASET.DATA["WORD_INDEX"].items():
            try:
                embedding_vector = word_vectors[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                continue

        # Borrar modelo para ahorrar memoria
        del word_vectors

        model = self.get_sub_model(w2v_emb_size, embedding_matrix)

        print(model.summary())

        return model

    def get_sub_model(self, w2v_emb_size, embedding_matrix):
        mv = self.CONFIG["model"]["model_version"]
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Embedding(self.DATASET.DATA["VOCAB_SIZE"], w2v_emb_size, weights=[embedding_matrix], trainable=False, mask_zero=True))

        if mv == "0":
            model.add(tf.keras.layers.LSTM(128))
            model.add(tf.keras.layers.Dense(64, activation='relu'))
            model.add(tf.keras.layers.Dense(32, activation='relu'))
            model.add(tf.keras.layers.Dense(16, activation='relu'))
            model.add(tf.keras.layers.Dense(1))

        if mv == "1":
            model.add(tf.keras.layers.LSTM(128))
            model.add(tf.keras.layers.Dropout(.4))
            model.add(tf.keras.layers.Dense(64, activation='relu'))
            model.add(tf.keras.layers.Dropout(.3))
            model.add(tf.keras.layers.Dense(32, activation='relu'))
            model.add(tf.keras.layers.Dropout(.2))
            model.add(tf.keras.layers.Dense(16, activation='relu'))
            model.add(tf.keras.layers.Dropout(.1))
            model.add(tf.keras.layers.Dense(1))

        if mv == "2":
            model.add(tf.keras.layers.LSTM(64))
            model.add(tf.keras.layers.Dense(32, activation='relu'))
            model.add(tf.keras.layers.Dense(16, activation='relu'))
            model.add(tf.keras.layers.Dense(1))

        if mv == "3":
            model.add(tf.keras.layers.LSTM(256))
            model.add(tf.keras.layers.Dense(128, activation='relu'))
            model.add(tf.keras.layers.Dense(64, activation='relu'))
            model.add(tf.keras.layers.Dense(32, activation='relu'))
            model.add(tf.keras.layers.Dense(16, activation='relu'))
            model.add(tf.keras.layers.Dense(1))

        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=self.CONFIG["model"]["learning_rate"]), metrics=['mean_absolute_error'])

        return model

    def evaluate(self, test=False):

        if test:
            test_set = LSTM2VALsequence(self, set_name="TEST")
        else:
            test_set = LSTM2VALsequence(self, is_dev=1)

        ret = self.MODEL.evaluate(test_set, verbose=0)

        print_g(dict(zip(self.MODEL.metrics_names, ret)))

    def get_train_dev_sequences(self):
        train = LSTM2VALsequence(self, is_dev=0)
        dev = LSTM2VALsequence(self, is_dev=1)

        self.DATASET.DATA["TRAIN_DEV"] = []

        return train, dev


class LSTM2VALsequence(BaseSequenceXY):

    def __init__(self, model, set_name="TRAIN_DEV", is_dev=-1):
        self.IS_DEV = is_dev
        self.SET_NAME = set_name
        BaseSequenceXY.__init__(self, parent_model=model)

    def init_data(self):
        ret = self.MODEL.DATASET.DATA[self.SET_NAME]

        if self.IS_DEV >= 0:
            ret = ret.loc[ret["dev"] == self.IS_DEV]

        with tf.device("GPU"):
            X = tf.constant(np.row_stack(ret.seq))
            Y = tf.constant(ret.rating)
            del ret

        return (X, Y)

    def preprocess_input(self, batch_data):
        return batch_data

    def preprocess_output(self, batch_data):
        return batch_data
