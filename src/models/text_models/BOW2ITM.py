# -*- coding: utf-8 -*-
from src.models.text_models.RSTModel import RSTModel
from src.sequences.BaseSequence import BaseSequence
from src.Common import print_g

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import normalize
import tensorflow_ranking as tfr
import tensorflow as tf
import pandas as pd
import numpy as np


class BOW2ITM(RSTModel):
    """ Predecir, a partir de una review codificada mendiante BOW, el restaurante de la review """
    def __init__(self, config, dataset):
        RSTModel.__init__(self, config=config, dataset=dataset)

    def get_model(self):

        mv = self.CONFIG["model"]["model_version"]
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(len(self.DATASET.DATA["FEATURES_NAME"]),), name="input_bow"))

        if mv == "0":  # OJO, ESTE ERA EL 3 DE ANTES
            model.add(tf.keras.layers.Dense(self.DATASET.DATA["N_ITEMS"], name="bow_2_rst", kernel_initializer=tf.keras.initializers.Ones()))
            model.add(tf.keras.layers.Dropout(.1))
            model.add(tf.keras.layers.BatchNormalization(name="bow_2_rst_bn"))

        '''
        if mv == "0":
            model.add(tf.keras.layers.Dense(self.DATASET.DATA["N_ITEMS"], name="bow_2_rst", kernel_initializer=tf.keras.initializers.Ones()))

        if mv == "1":
            model.add(tf.keras.layers.Dense(self.DATASET.DATA["N_ITEMS"], name="bow_2_rst", kernel_initializer=tf.keras.initializers.Ones()))
            model.add(tf.keras.layers.Dropout(.1))

        if mv == "2":
            model.add(tf.keras.layers.Dense(self.DATASET.DATA["N_ITEMS"], name="bow_2_rst", kernel_initializer=tf.keras.initializers.Ones()))
            model.add(tf.keras.layers.Dropout(.5))

        if mv == "3":
            model.add(tf.keras.layers.Dense(self.DATASET.DATA["N_ITEMS"], name="bow_2_rst", kernel_initializer=tf.keras.initializers.Ones()))
            model.add(tf.keras.layers.Dropout(.1))
            model.add(tf.keras.layers.BatchNormalization(name="bow_2_rst_bn"))

        if mv == "4":
            model.add(tf.keras.layers.Dense(self.DATASET.DATA["N_ITEMS"], name="bow_2_rst", kernel_initializer=tf.keras.initializers.Ones()))
            model.add(tf.keras.layers.Dropout(.5))
            model.add(tf.keras.layers.BatchNormalization(name="bow_2_rst_bn"))
        '''
        activation_function = "sigmoid"

        if activation_function == "sigmoid":
            # loss = tf.keras.losses.BinaryCrossentropy()
            loss = tf.keras.losses.CategoricalCrossentropy()
            # loss = tf.keras.losses.BinaryFocalCrossentropy()

        elif activation_function == "softmax":
            loss = tf.keras.losses.CategoricalCrossentropy()

        model.add(tf.keras.layers.Activation(activation_function, name="output_rst"))
        metrics = ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5'), tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10')]
        metrics = [tfr.keras.metrics.RecallMetric(topn=1, name='r1'), tfr.keras.metrics.RecallMetric(topn=5, name='r5'), tfr.keras.metrics.RecallMetric(topn=10, name='r10'),
                   tfr.keras.metrics.PrecisionMetric(topn=5, name='p5'), tfr.keras.metrics.PrecisionMetric(topn=10, name='p10')]

        model.build(self.CONFIG["model"]["batch_size"])

        optimizer = tf.keras.optimizers.legacy.Adam(self.CONFIG["model"]["learning_rate"])
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model

    def get_train_dev_sequences(self, dev):
        
        if dev:
            train = BOW2RSTsequence(self, is_dev=0)
            dev = BOW2RSTsequence(self, is_dev=1)
        else:
            raise NotImplemented

        return train, dev

    def evaluate(self, test=False):

        if test:
            test_set = BOW2RSTsequence(self, set_name="TEST")
        else:
            test_set = BOW2RSTsequence(self, is_dev=1)

        metrics = [
            tf.keras.metrics.Precision(top_k=1, name="Precision@1"),
            tf.keras.metrics.Precision(top_k=5, name="Precision@5"),
            tf.keras.metrics.Precision(top_k=10, name="Precision@10"),
            tf.keras.metrics.Recall(top_k=1, name="Recall@1"),
            tf.keras.metrics.Recall(top_k=5, name="Recall@5"),
            tf.keras.metrics.Recall(top_k=10, name="Recall@10")]

        self.MODEL.compile(loss=self.MODEL.loss, optimizer=self.MODEL.optimizer, metrics=metrics)
        ret = self.MODEL.evaluate(test_set, verbose=0)
        ret = dict(zip(self.MODEL.metrics_names, ret))
        
        for r in [1, 5, 10]:
            r_at = ret[f"Recall@{r}"]
            p_at = ret[f"Precision@{r}"]
            f1_at = 2 * ((r_at * p_at) / (r_at + p_at))
            ret[f"F1@{r}"] = f1_at

        ret = pd.DataFrame([ret.values()], columns=ret.keys())
        print_g(ret, title=False)

        return ret  

        # ret = self.MODEL.evaluate(test_set, verbose=0)
        # ret = dict(zip(self.MODEL.metrics_names, ret))
        # print_g(ret)
               
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
        print("\LEMM: %s" % (text))
        print("\tBOW: %s" % (",".join(bow_words)))

        for rst in recommended_rests:
            print("\t- %s" % (self.DATASET.DATA["TEST"].loc[self.DATASET.DATA["TEST"].id_item == rst]["name"].values[0]))

            # X Palabras más relevantes para predecir el restaurante seleccionado
            word_weights = rst_model_weights[:, rst]  # + rst_model_weights_bias
            word_ids = np.argsort(-word_weights)[:20]
            most_relevant_w = np.asarray(self.DATASET.DATA["FEATURES_NAME"])[word_ids]

            # Intersección entre palabras del usuario y del restaurante
            usr_rst_intr = list(set(np.where(word_weights > 0)[0]).intersection(set(np.argwhere(normed_bow[0] > 0)[:, 0])))
            usr_rst_intr = np.asarray(self.DATASET.DATA["FEATURES_NAME"])[usr_rst_intr]

            print("\t\t▲ %s" % (",".join(most_relevant_w)))
            print("\t\t∩ %s" % (",".join(usr_rst_intr)))

    def explain_test_sample(self, test_real_sample):
        # Obtener el texto de la fila dataframe y codificar en formato adecuado (al ser un caso real, ya se ha codificado).
        bow_encoding = self.DATASET.DATA["BOW_SEQUENCES"][test_real_sample["bow"]].todense()

        # Datos del rastaurante real
        restaurant_name = test_real_sample["name"]

        # Obtener la matriz de pesos relevante
        rst_model = self.MODEL
        rst_model_weights = rst_model.get_layer("bow_2_rst").get_weights()[0]
        
        # Importancia de todas las palabras del vocabulario en este restaurante
        rst_word_weights = rst_model_weights[:, test_real_sample.id_item]

        # Palabras de la reseña (el único filtrado que se hace es el POS)
        rvw_words = np.array(list(np.argwhere(bow_encoding[0] > 0)[:, 1]))
        rvw_word_names = np.asarray(self.DATASET.DATA["FEATURES_NAME"])[rvw_words]

        print({w_nm: rst_word_weights[w_id] for w_id, w_nm in zip(rvw_words, rvw_word_names)})

        return {"max": rst_model_weights.max(), "min": rst_model_weights.min(), "values": {w_nm: rst_word_weights[w_id] for w_id, w_nm in zip(rvw_words, rvw_word_names)}}


class BOW2RSTsequence(BaseSequence):

    def __init__(self, model, set_name="TRAIN_DEV", is_dev=-1):
        self.IS_DEV = is_dev
        self.SET_NAME = set_name
        BaseSequence.__init__(self, parent_model=model)
        self.KHOT = MultiLabelBinarizer(classes=list(range(self.MODEL.DATASET.DATA["N_ITEMS"])))

    def init_data(self):
        ret = self.MODEL.DATASET.DATA[self.SET_NAME]

        if self.IS_DEV >= 0:
            ret = ret.loc[ret["dev"] == self.IS_DEV]

        return ret

    def preprocess_input(self, batch_data):
        # return np.row_stack(batch_data.bow)
        # return np.row_stack(batch_data.bow.apply(lambda x: x.todense().tolist()[0]))
        return self.MODEL.DATASET.DATA["BOW_SEQUENCES"][batch_data.bow].todense()

    def preprocess_output(self, batch_data):
        return self.KHOT.fit_transform(np.expand_dims(batch_data.id_item.values, -1))
