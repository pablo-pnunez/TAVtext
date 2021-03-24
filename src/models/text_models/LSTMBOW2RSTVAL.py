# -*- coding: utf-8 -*-
from src.Common import print_g
from src.models.KerasModelClass import KerasModelClass
from src.sequences.BaseSequence import BaseSequence

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.models import KeyedVectors


class LSTMFBOW2RSTVAL(KerasModelClass):
    """FIJANDO MODELO BOW: Predecir, a partir de una review codificada mendiante una LSTM utilizando las palabras del W2V y la review BOW, la nota de dicha review y el restaurante """

    def __init__(self, config, dataset, w2v_model, bow_model):
        self.W2V_MODEL = w2v_model
        self.BOW_MODEL = bow_model
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

        print_g("Loading bow...")
        bow_model = self.BOW_MODEL.MODEL
        # Fijar pesos aprendidos previamente
        for l in bow_model.layers: l.trainable = False

        input_val = tf.keras.layers.Input(shape=(self.DATASET.DATA["MAX_LEN_PADDING"],))
        h = tf.keras.layers.Embedding(self.DATASET.DATA["VOCAB_SIZE"], w2v_emb_size, weights=[embedding_matrix], trainable=False, mask_zero=True)(input_val)
        h = tf.keras.layers.LSTM(128)(h)
        h = tf.keras.layers.Concatenate(axis=1)([bow_model.output, h])
        h = tf.keras.layers.Dense(128, activation='relu')(h)
        h = tf.keras.layers.Dense(32, activation='relu')(h)
        output_val = tf.keras.layers.Dense(1)(h)

        model = tf.keras.models.Model(inputs=[bow_model.input, input_val], outputs=[output_val])
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=self.CONFIG["model"]["learning_rate"]), metrics=['mean_absolute_error'])

        return model

    def get_train_dev_sequences(self):
        train = LSTMFBOW2RSTVALsequence(self, is_dev=0)
        dev = LSTMFBOW2RSTVALsequence(self, is_dev=1)

        return train, dev


class LSTMBOW2RSTVAL(KerasModelClass):
    """Predecir, a partir de una review codificada mendiante una LSTM utilizando las palabras del W2V y la review BOW, la nota de dicha review y el restaurante """

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

        input_rst = tf.keras.layers.Input(shape=(self.DATASET.CONFIG["num_palabras"],), name="input_rest")
        x = tf.keras.layers.Dense(self.DATASET.DATA["N_RST"], name="output_layer")(input_rst)
        x = tf.keras.layers.Dropout(.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output_rst = tf.keras.layers.Activation("softmax", name="output_rst")(x)

        input_val = tf.keras.layers.Input(shape=(self.DATASET.DATA["MAX_LEN_PADDING"],))
        h = tf.keras.layers.Embedding(self.DATASET.DATA["VOCAB_SIZE"], w2v_emb_size, weights=[embedding_matrix], trainable=False, mask_zero=True)(input_val)
        h = tf.keras.layers.LSTM(128)(h)
        h = tf.keras.layers.Concatenate(axis=1)([output_rst, h])
        h = tf.keras.layers.Dense(128, activation='relu')(h)
        h = tf.keras.layers.Dense(32, activation='relu')(h)
        output_val = tf.keras.layers.Dense(1, name="output_val")(h)

        model = tf.keras.models.Model(inputs=[input_rst, input_val], outputs=[output_rst, output_val])

        losses = {
            "output_rst": "categorical_crossentropy",
            "output_val": "mean_squared_error",
        }

        metrics = {
            "output_rst": ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5'), tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10')],
            "output_val": ["mean_absolute_error"]
        }

        model.compile(loss=losses, optimizer=tf.keras.optimizers.Adam(lr=self.CONFIG["model"]["learning_rate"]), metrics=metrics)

        return model

    def get_train_dev_sequences(self):
        train = LSTMBOW2RSTVALsequence(self, is_dev=0)
        dev = LSTMBOW2RSTVALsequence(self, is_dev=1)

        return train, dev


class LSTMFBOW2RSTVALsequence(BaseSequence):

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
        x1 = np.row_stack(batch_data.bow)
        x2 = np.row_stack(batch_data.seq)
        return [x1, x2]

    def preprocess_output(self, batch_data):
        y = batch_data["rating"].values
        return y


class LSTMBOW2RSTVALsequence(LSTMFBOW2RSTVALsequence):

    def __init__(self, model, set_name="TRAIN_DEV", is_dev=-1):
        LSTMFBOW2RSTVALsequence.__init__(self, model=model, set_name=set_name, is_dev=is_dev)
        self.KHOT = MultiLabelBinarizer(classes=list(range(self.MODEL.DATASET.DATA["N_RST"])))

    def preprocess_output(self, batch_data):
        y1 = self.KHOT.fit_transform(np.expand_dims(batch_data.id_restaurant.values, -1))
        y2 = batch_data["rating"].values
        return [y1, y2]
