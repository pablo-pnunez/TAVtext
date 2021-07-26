# -*- coding: utf-8 -*-
from src.models.text_models.RSTModel import RSTModel
from src.sequences.BaseSequence import BaseSequence
from src.Common import print_g

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MultiLabelBinarizer


class BOW2RST(RSTModel):
    """ Predecir, a partir de una review codificada mendiante BOW, el restaurante de la review """
    def __init__(self, config, dataset):
        RSTModel.__init__(self, config=config, dataset=dataset)

    def get_model(self):

        mv = self.CONFIG["model"]["model_version"]
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.DATASET.CONFIG["num_palabras"],), name="input_bow"))

        if mv == "0":
            model.add(tf.keras.layers.Dense(self.DATASET.DATA["N_RST"], name="bow_2_rst", kernel_initializer=tf.keras.initializers.Ones()))

        if mv == "1":
            model.add(tf.keras.layers.Dense(self.DATASET.DATA["N_RST"], name="bow_2_rst", kernel_initializer=tf.keras.initializers.Ones()))
            model.add(tf.keras.layers.Dropout(.1))

        if mv == "2":
            model.add(tf.keras.layers.Dense(self.DATASET.DATA["N_RST"], name="bow_2_rst", kernel_initializer=tf.keras.initializers.Ones()))
            model.add(tf.keras.layers.Dropout(.5))

        if mv == "3":
            model.add(tf.keras.layers.Dense(self.DATASET.DATA["N_RST"], name="bow_2_rst", kernel_initializer=tf.keras.initializers.Ones()))
            model.add(tf.keras.layers.Dropout(.1))
            model.add(tf.keras.layers.BatchNormalization(name="bow_2_rst_bn"))

        if mv == "4":
            model.add(tf.keras.layers.Dense(self.DATASET.DATA["N_RST"], name="bow_2_rst", kernel_initializer=tf.keras.initializers.Ones()))
            model.add(tf.keras.layers.Dropout(.5))
            model.add(tf.keras.layers.BatchNormalization(name="bow_2_rst_bn"))

        model.add(tf.keras.layers.Activation("softmax", name="output_rst"))
        metrics = ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5'), tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10')]
        model.compile(optimizer=tf.keras.optimizers.Adam(self.CONFIG["model"]["learning_rate"]), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=metrics)

        return model

    def get_train_dev_sequences(self):
        train = BOW2RSTsequence(self, is_dev=0)
        dev = BOW2RSTsequence(self, is_dev=1)

        return train, dev

    def evaluate(self, test=False):

        if test:
            test_set = BOW2RSTsequence(self, set_name="TEST")
        else:
            test_set = BOW2RSTsequence(self, is_dev=1)

        ret = self.MODEL.evaluate(test_set, verbose=0)

        ret = dict(zip(self.MODEL.metrics_names, ret))
        print_g(ret)
               
        return ret

    def eval_custom_text(self, text_src):

        text = self.DATASET.prerpocess_text(text_src)
        bow = self.DATASET.DATA["VECTORIZER"].transform([text])
        normed_bow = normalize(bow.todense(), axis=1, norm='l1')
        bow_words = np.asarray(self.DATASET.DATA["FEATURES_NAME"])[np.argwhere(normed_bow[0] > 0)[:, 0]]

        # Obtener el modelo que predice restaurantes, junto con la matriz de pesos relevante
        rst_model = self.MODEL
        rst_model_weights = rst_model.get_layer("bow_2_rst").get_weights()[0]

        # Predecir 5 restaurantes con la parte correspondiente del modelo
        rev_rst_pred = rst_model.predict(normed_bow)
        rev_rst_pred = np.apply_along_axis(lambda x: (-x).argsort()[:3], 1, rev_rst_pred)
        recommended_rests = rev_rst_pred.flatten()

        print("\n")
        print_g("\'%s\'" % text_src)
        print("\tBOW: %s" % (",".join(bow_words)))

        for rst in recommended_rests:
            print("\t- %s" % (self.DATASET.DATA["TEST"].loc[self.DATASET.DATA["TEST"].id_restaurant == rst]["name"].values[0]))

            # X Palabras más relevantes para predecir el restaurante seleccionado
            word_weights = rst_model_weights[:, rst]  # + rst_model_weights_bias
            word_ids = np.argsort(-word_weights)[:20]
            most_relevant_w = np.asarray(self.DATASET.DATA["FEATURES_NAME"])[word_ids]

            # Intersección entre palabras del usuario y del restaurante
            usr_rst_intr = list(set(np.where(word_weights > 0)[0]).intersection(set(np.argwhere(normed_bow[0] > 0)[:, 0])))
            usr_rst_intr = np.asarray(self.DATASET.DATA["FEATURES_NAME"])[usr_rst_intr]

            print("\t\t▲ %s" % (",".join(most_relevant_w)))
            print("\t\t∩ %s" % (",".join(usr_rst_intr)))


class BOW2RSTsequence(BaseSequence):

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
        return np.row_stack(batch_data.bow)

    def preprocess_output(self, batch_data):
        return self.KHOT.fit_transform(np.expand_dims(batch_data.id_restaurant.values, -1))
