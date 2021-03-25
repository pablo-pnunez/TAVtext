# -*- coding: utf-8 -*-
from src.Common import print_g
from src.models.KerasModelClass import KerasModelClass
from src.sequences.BaseSequence import BaseSequence

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer


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

        input_bow = tf.keras.layers.Input(shape=(self.DATASET.CONFIG["num_palabras"],), name="input_bow")
        x = tf.keras.layers.Dense(self.DATASET.DATA["N_RST"], name="output_layer")(input_bow)
        x = tf.keras.layers.Dropout(.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output_rst = tf.keras.layers.Activation("softmax", name="output_rst")(x)
        model_one = tf.keras.models.Model(inputs=[input_bow], outputs=[output_rst])

        input_w2v = tf.keras.layers.Input(shape=(self.DATASET.DATA["MAX_LEN_PADDING"],), name="input_w2v")
        h = tf.keras.layers.Embedding(self.DATASET.DATA["VOCAB_SIZE"], w2v_emb_size, weights=[embedding_matrix], trainable=False, mask_zero=True)(input_w2v)
        output_lstm = tf.keras.layers.LSTM(128, name="output_lstm")(h)
        model_two = tf.keras.models.Model(inputs=[input_w2v], outputs=[output_lstm])

        input_rst = tf.keras.layers.Input(shape=(self.DATASET.DATA["N_RST"],), name="input_rst")
        input_lstm = tf.keras.layers.Input(shape=(128,), name="input_lstm")
        h = tf.keras.layers.Concatenate(axis=1)([input_rst, input_lstm])
        h = tf.keras.layers.Dense(128, activation='relu')(h)
        h = tf.keras.layers.Dense(32, activation='relu')(h)
        output_val = tf.keras.layers.Dense(1, name="output_val")(h)
        model_three = tf.keras.models.Model(inputs=[input_rst, input_lstm], outputs=[output_val])

        model = tf.keras.models.Model(inputs=[input_bow, input_w2v], outputs=[output_rst, model_three([output_rst, output_lstm])])

        losses = {
            "output_rst": "categorical_crossentropy",
            "model_3": "mean_squared_error",
        }

        metrics = {
            "output_rst": ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5'), tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10')],
            "model_3": ["mean_absolute_error"]
        }

        model.compile(loss=losses, optimizer=tf.keras.optimizers.Adam(lr=self.CONFIG["model"]["learning_rate"]), metrics=metrics)

        return model

    def get_train_dev_sequences(self):
        train = LSTMBOW2RSTVALsequence(self, is_dev=0)
        dev = LSTMBOW2RSTVALsequence(self, is_dev=1)

        return train, dev

    def evaluate(self):
        test_data = self.DATASET.DATA["TEST"]

        rst_model = tf.keras.models.Model(inputs=[self.MODEL.get_layer("input_bow").input], outputs=[self.MODEL.get_layer("output_rst").output])
        lstm_model = tf.keras.models.Model(inputs=[self.MODEL.get_layer("input_w2v").input], outputs=[self.MODEL.get_layer("output_lstm").output])
        val_model = self.MODEL.get_layer("model_3")

        rst_model_weights = rst_model.get_layer("output_layer").get_weights()[0]

        for _, rev in test_data.iterrows():
            # Predecir 5 restaurantes con la parte correspondiente del modelo
            rev_rst_pred = rst_model.predict(np.expand_dims(rev.bow, 0)).flatten()
            rev_rst_pred = (-rev_rst_pred).argsort()[:5]

            # Obtener el de mayor valoración de todos ellos
            all_vals = []
            rev_lst_pred = lstm_model.predict(np.expand_dims(rev.seq, 0))
            for rst in rev_rst_pred:
                rst_vec = np.zeros((1, self.DATASET.DATA["N_RST"]))
                rst_vec[0][rst] = 1
                val_pred = val_model.predict([rst_vec, rev_lst_pred]).flatten()
                all_vals.append((rst, test_data.loc[test_data.id_restaurant == rst]["name"].values[0], val_pred[0]))

            all_vals = pd.DataFrame(all_vals, columns=["id_restaurant", "name", "pred_val"])
            selected_restaurant = all_vals.pred_val.argmax()
            selected_restaurant = all_vals.iloc[selected_restaurant]
            print("%s => %s" % (rev["name"], selected_restaurant["name"]))

            # X Palabras relevantes del restaurante seleccionado
            word_weights = rst_model_weights[:, selected_restaurant.id_restaurant]
            word_ids = np.argsort(-word_weights)[:5]

            print("%s => %s" % (rev["name"], selected_restaurant["name"]))

class LSTMFBOW2RSTVAL(LSTMBOW2RSTVAL):
    """FIJANDO MODELO BOW: Predecir, a partir de una review codificada mendiante una LSTM utilizando las palabras del W2V y la review BOW, la nota de dicha review y el restaurante """

    def __init__(self, config, dataset, w2v_model, bow_model):
        self.BOW_MODEL = bow_model
        LSTMBOW2RSTVAL.__init__(self, config=config, dataset=dataset, w2v_model=w2v_model)

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

        input_val = tf.keras.layers.Input(shape=(self.DATASET.DATA["MAX_LEN_PADDING"],), name="input_val")
        h = tf.keras.layers.Embedding(self.DATASET.DATA["VOCAB_SIZE"], w2v_emb_size, weights=[embedding_matrix], trainable=False, mask_zero=True)(input_val)
        h = tf.keras.layers.LSTM(128)(h)
        h = tf.keras.layers.Concatenate(axis=1)([bow_model.output, h])
        h = tf.keras.layers.Dense(128, activation='relu')(h)
        h = tf.keras.layers.Dense(32, activation='relu')(h)
        output_val = tf.keras.layers.Dense(1, name="output_val")(h)

        model = tf.keras.models.Model(inputs=[bow_model.input, input_val], outputs=[output_val])
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=self.CONFIG["model"]["learning_rate"]), metrics=['mean_absolute_error'])

        return model

    def get_train_dev_sequences(self):
        train = LSTMFBOW2RSTVALsequence(self, is_dev=0)
        dev = LSTMFBOW2RSTVALsequence(self, is_dev=1)

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
