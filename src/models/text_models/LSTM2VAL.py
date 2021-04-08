# -*- coding: utf-8 -*-
from src.Common import print_g
from src.models.KerasModelClass import KerasModelClass
from src.sequences.BaseSequence import BaseSequence

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.models import KeyedVectors


class LSTM2VAL(KerasModelClass):
    """ Predecir, a partir de una review codificada mendiante una LSTM utilizando las palabras del W2V, la nota de dicha review """
    def __init__(self, config, dataset, w2v_model):
        self.W2V_MODEL = w2v_model
        KerasModelClass.__init__(self, config=config, dataset=dataset)

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

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Embedding(self.DATASET.DATA["VOCAB_SIZE"], w2v_emb_size, weights=[embedding_matrix], trainable=False, mask_zero=True))
        model.add(tf.keras.layers.LSTM(128))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=self.CONFIG["model"]["learning_rate"]), metrics=['mean_absolute_error'])

        return model

    def baseline(self, test=False):
        """ Predecir la media """

        if not test:
            the_mean = self.DATASET.DATA["TRAIN_DEV"].loc[self.DATASET.DATA["TRAIN_DEV"].dev==0].rating.mean()
            mae = np.abs(the_mean - self.DATASET.DATA["TRAIN_DEV"].loc[self.DATASET.DATA["TRAIN_DEV"].dev==1].rating.values).mean()
        else:
            the_mean = self.DATASET.DATA["TRAIN_DEV"].rating.mean()
            mae = np.abs(the_mean - self.DATASET.DATA["TEST"].rating.values).mean()
        
        ttl = "TEST" if test else "DEV" 

        print_g("%s baseline MAE: %.4f" % (ttl, mae))

    def get_train_dev_sequences(self):
        train = LSTM2VALsequence(self, is_dev=0)
        dev = LSTM2VALsequence(self, is_dev=1)

        return train, dev


class LSTM2VALsequence(BaseSequence):

    def __init__(self, model, set_name="TRAIN_DEV", is_dev=-1):
        self.IS_DEV = is_dev
        self.SET_NAME = set_name
        BaseSequence.__init__(self, parent_model=model)

    def init_data(self):
        ret = self.MODEL.DATASET.DATA[self.SET_NAME]

        if self.IS_DEV >= 0:
            ret = ret.loc[ret["dev"] == self.IS_DEV]

        return ret

    def preprocess_input(self, batch_data):
        return np.row_stack(batch_data.seq)

    def preprocess_output(self, batch_data):
        return batch_data["rating"].values
