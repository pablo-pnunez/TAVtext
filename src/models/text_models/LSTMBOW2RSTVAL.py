# -*- coding: utf-8 -*-
from src.Common import print_g
from src.models.KerasModelClass import KerasModelClass
from src.sequences.BaseSequence import BaseSequence

import numpy as np
import pandas as pd
from tqdm import tqdm
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
        x = tf.keras.layers.Dense(self.DATASET.DATA["N_RST"], name="bow_2_rst", kernel_initializer=tf.keras.initializers.Ones())(input_bow)
        x = tf.keras.layers.Dropout(.2)(x)
        x = tf.keras.layers.BatchNormalization(name="bow_2_rst_bn")(x)
        output_rst = tf.keras.layers.Activation("softmax", name="output_rst")(x)

        input_w2v = tf.keras.layers.Input(shape=(self.DATASET.DATA["MAX_LEN_PADDING"],), name="input_w2v")
        h = tf.keras.layers.Embedding(self.DATASET.DATA["VOCAB_SIZE"], w2v_emb_size, weights=[embedding_matrix], trainable=False, mask_zero=True)(input_w2v)
        output_lstm = tf.keras.layers.LSTM(128, name="output_lstm")(h)

        input_rst = tf.keras.layers.Input(shape=(self.DATASET.DATA["N_RST"],), name="input_rst")
        input_lstm = tf.keras.layers.Input(shape=(128,), name="input_lstm")
        h = tf.keras.layers.Concatenate(axis=1)([input_rst, input_lstm])
        h = tf.keras.layers.Dense(128, activation='relu')(h)
        h = tf.keras.layers.Dense(32, activation='relu')(h)
        output_val = tf.keras.layers.Dense(1, name="output_val")(h)
        val_model = tf.keras.models.Model(inputs=[input_rst, input_lstm], outputs=[output_val], name="val_model")

        model = tf.keras.models.Model(inputs=[input_bow, input_w2v], outputs=[output_rst, val_model([output_rst, output_lstm])])

        losses = {
            "output_rst": "categorical_crossentropy",
            "val_model": "mean_squared_error",
        }

        metrics = {
            "output_rst": ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5'), tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10')],
            "val_model": ["mean_absolute_error"]
        }

        model.compile(loss=losses, optimizer=tf.keras.optimizers.Adam(lr=self.CONFIG["model"]["learning_rate"]), metrics=metrics)

        return model

    def get_train_dev_sequences(self):
        train = LSTMBOW2RSTVALsequence(self, is_dev=0)
        dev = LSTMBOW2RSTVALsequence(self, is_dev=1)

        return train, dev

    def evaluate(self, verbose=0):
        test_data = self.DATASET.DATA["TEST"].copy()

        # Obtener el modelo que predice restaurantes, junto con la matriz de pesos relevante
        rst_model = tf.keras.models.Model(inputs=[self.MODEL.get_layer("input_bow").input], outputs=[self.MODEL.get_layer("output_rst").output])
        rst_model_weights = rst_model.get_layer("bow_2_rst").get_weights()[0]

        # Predecir, para cada elemeneto de test, la nota que se le daría
        test_data["experience_rating"] = self.MODEL.predict([np.row_stack(test_data.bow.values), np.row_stack(test_data.seq.values)])

        # Predecir 5 restaurantes con la parte correspondiente del modelo
        rev_rst_pred = rst_model.predict(np.row_stack(test_data.bow.values))
        rev_rst_pred = np.apply_along_axis(lambda x: (-x).argsort()[:5], 1, rev_rst_pred)
        test_data["recommended_rests"] = rev_rst_pred.tolist()

        eval_data = []
        for _, rev in tqdm(test_data.iterrows(), total=len(test_data)):

            usr_bow_words = np.asarray(self.DATASET.DATA["FEATURES_NAME"])[np.argwhere(np.asarray(rev.bow) > 0).flatten()]

            for rst in rev["recommended_rests"]:

                # X Palabras más relevantes para predecir el restaurante seleccionado
                word_weights = rst_model_weights[:, rst]  # + rst_model_weights_bias
                word_ids = np.argsort(-word_weights)[:5]
                most_relevant_w = np.asarray(self.DATASET.DATA["FEATURES_NAME"])[word_ids]

                # Intersección entre palabras del usuario y del restaurante
                usr_rst_intr = list(set(np.where(word_weights > 0)[0]).intersection(set(np.argwhere(np.asarray(rev.bow) > 0).flatten())))
                usr_rst_intr = np.asarray(self.DATASET.DATA["FEATURES_NAME"])[usr_rst_intr]

                eval_data.append((rev.reviewId, rev["id_restaurant"], rev["name"], rst, test_data.loc[test_data.id_restaurant == rst]["name"].values[0], rev["experience_rating"],
                                  most_relevant_w, usr_rst_intr, usr_bow_words, len(usr_bow_words), len(usr_rst_intr)))

        eval_data = pd.DataFrame(eval_data, columns=["reviewId", "id_restaurant", "restaurant_name", "id_restaurant_rec", "restaurant_name_rec", "experience_rating",
                                                     "most_relevant_words", "intersection", "bow_words", "bow_words_len", "intersection_len"])

        final_val = []
        for rv, rv_dt in eval_data.groupby("reviewId"):
            intrs = np.average(rv_dt["intersection_len"] / rv_dt["bow_words_len"])
            final_val.append((rv, intrs))

            if verbose == 1:
                print("\n%s [%.2f]" % (rv_dt["restaurant_name"].values[0], rv_dt["experience_rating"].values[0]))
                print("\tBOW: %s" % (",".join(rv_dt["bow_words"].values[0])))
                for _, rst in rv_dt.iterrows():
                    print("\t- %s" % (rst["restaurant_name_rec"]))
                    print("\t\t▲ %s" % (",".join(rst["most_relevant_words"])))
                    print("\t\t∩ %s" % (",".join(rst["intersection"])))

                input("Press Enter to continue...")

        final_val = pd.DataFrame(final_val, columns=["reviewId", "intersection"])
        print("Test average intersection pctg: %f" % final_val["intersection"].mean())

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
        for lyr in bow_model.layers:
            lyr.trainable = False

        input_w2v = tf.keras.layers.Input(shape=(self.DATASET.DATA["MAX_LEN_PADDING"],), name="input_w2v")
        h = tf.keras.layers.Embedding(self.DATASET.DATA["VOCAB_SIZE"], w2v_emb_size, weights=[embedding_matrix], trainable=False, mask_zero=True)(input_w2v)
        output_lstm = tf.keras.layers.LSTM(128, name="output_lstm")(h)

        input_rst = tf.keras.layers.Input(shape=(self.DATASET.DATA["N_RST"],), name="input_rst")
        input_lstm = tf.keras.layers.Input(shape=(128,), name="input_lstm")
        h = tf.keras.layers.Concatenate(axis=1)([input_rst, input_lstm])
        h = tf.keras.layers.Dense(128, activation='relu')(h)
        h = tf.keras.layers.Dense(32, activation='relu')(h)
        output_val = tf.keras.layers.Dense(1, name="output_val")(h)
        val_model = tf.keras.models.Model(inputs=[input_rst, input_lstm], outputs=[output_val], name="val_model")

        model = tf.keras.models.Model(inputs=[bow_model.input, input_w2v], outputs=[val_model([bow_model.output, output_lstm])])
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
