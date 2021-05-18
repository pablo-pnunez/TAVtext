# -*- coding: utf-8 -*-
from src.Common import print_g
from src.models.text_models.RSTModel import RSTModel
from src.sequences.BaseSequence import BaseSequence

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer


class LSTM2RST(RSTModel):
    """ Predecir, a partir de una review codificada mendiante LSTM, el restaurante de la review """

    def __init__(self, config, dataset, w2v_model):
        self.W2V_MODEL = w2v_model
        RSTModel.__init__(self, config=config, dataset=dataset)

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

        return model

    def get_sub_model(self, w2v_emb_size, embedding_matrix):
        mv = self.CONFIG["model"]["model_version"]
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Embedding(self.DATASET.DATA["VOCAB_SIZE"], w2v_emb_size, weights=[embedding_matrix], trainable=False, mask_zero=True))

        if mv == "0":
            model.add(tf.keras.layers.LSTM(128))
            model.add(tf.keras.layers.Dropout(.8))

        if mv == "1":
            model.add(tf.keras.layers.LSTM(128))
            model.add(tf.keras.layers.Dropout(.8))
            model.add(tf.keras.layers.Dense(64, activation='relu'))
            model.add(tf.keras.layers.Dropout(.5))

        if mv == "2":
            model.add(tf.keras.layers.LSTM(256))
            model.add(tf.keras.layers.Dropout(.8))

        if mv == "3":
            model.add(tf.keras.layers.LSTM(256))
            model.add(tf.keras.layers.Dropout(.8))
            model.add(tf.keras.layers.Dense(128, activation='relu'))
            model.add(tf.keras.layers.Dropout(.5))

        model.add(tf.keras.layers.Dense(self.DATASET.DATA["N_RST"], activation='softmax'))
        metrics = ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5'), tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10')]
        model.compile(optimizer=tf.keras.optimizers.Adam(self.CONFIG["model"]["learning_rate"]), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=metrics,)

        print(model.summary())

        return model

    def get_train_dev_sequences(self):
        train = MySequence(self, is_dev=0)
        dev = MySequence(self, is_dev=1)

        return train, dev

    def evaluate(self, test=False):

        if test:
            test_set = MySequence(self, set_name="TEST")
        else:
            test_set = MySequence(self, is_dev=1)

        ret = self.MODEL.evaluate(test_set, verbose=0)

        print_g(dict(zip(self.MODEL.metrics_names, ret)))


class MySequence(BaseSequence):

    def __init__(self, model, set_name="TRAIN_DEV", is_dev=-1):
        self.IS_DEV = is_dev
        self.SET_NAME = set_name
        BaseSequence.__init__(self, parent_model=model)
        self.KHOT = MultiLabelBinarizer(classes=list(range(self.MODEL.DATASET.DATA["N_RST"])))

    def init_data(self):
        ret = self.MODEL.DATASET.DATA[self.SET_NAME]

        if self.IS_DEV >= 0:
            ret = ret.loc[ret["dev"] == self.IS_DEV]

        return ret

    def preprocess_input(self, batch_data):
        return np.row_stack(batch_data.seq)

    def preprocess_output(self, batch_data):
        return self.KHOT.fit_transform(np.expand_dims(batch_data.id_restaurant.values, -1))
