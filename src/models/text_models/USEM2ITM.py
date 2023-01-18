# -*- coding: utf-8 -*-
from src.Common import print_g
from src.models.text_models.RSTModel import RSTModel
from src.sequences.BaseSequence import BaseSequence

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text 
import tensorflow_ranking as tfr

from sklearn.preprocessing import MultiLabelBinarizer


class USEM2ITM(RSTModel):
    """ Predecir, a partir de una review codificada mendiante USEM, el restaurante de la review """

    def __init__(self, config, dataset):
        RSTModel.__init__(self, config=config, dataset=dataset)

    def get_model(self):
        model = self.get_sub_model()
        return model

    def get_sub_model(self):
        mv = self.CONFIG["model"]["model_version"]
        self.MODEL_VERSION = mv

        inputs = tf.keras.layers.Input(shape=(), dtype=tf.string)

        # preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3")
        # encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/mobilebert_multi_cased_L-24_H-128_B-512_A-4_F-4_OPT/1")
        #  encoder_inputs = preprocessor(inputs)
        # x = encoder(encoder_inputs)["pooled_output"]

        # REFERENCE: https://arxiv.org/pdf/1810.12836.pdf
        sentence_encoding_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3", trainable=True, input_shape=[], dtype=tf.string, name='USEM')
        x = sentence_encoding_layer(inputs)

        if mv == "0":
            x = tf.keras.layers.Dropout(.8)(x)
            # x = tf.keras.layers.Dense(512, activation='tanh')(x)
            # x = tf.keras.layers.Dropout(.5)(x)
            # x = tf.keras.layers.Dense(256, activation='tanh')(x)

        outputs = tf.keras.layers.Dense(self.DATASET.DATA["N_ITEMS"], activation="sigmoid", name="out", dtype='float32')(x)

        model = tf.keras.Model(inputs, outputs)

        metrics = [tfr.keras.metrics.RecallMetric(topn=1, name='r1'), tfr.keras.metrics.RecallMetric(topn=5, name='r5'), tfr.keras.metrics.RecallMetric(topn=10, name='r10'),
                   tfr.keras.metrics.PrecisionMetric(topn=5, name='p5'), tfr.keras.metrics.PrecisionMetric(topn=10, name='p10')]

        model.compile(optimizer=tf.keras.optimizers.Adam(self.CONFIG["model"]["learning_rate"]), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=metrics,)

        print(model.summary())

        return model

    def __create_dataset(self, dataframe):
        
        # Seleccionar del conjunto donde están todos los textos, los pertenecientes al cojunto "dataframe"
        all_data = pd.read_pickle(self.DATASET.DATASET_PATH+"ALL_DATA")[["reviewId", "text"]]
        text_data = all_data.set_index("reviewId").loc[dataframe.reviewId.values]["text"].values

        seq_data = text_data
        rst_data = dataframe.id_item.values
        
        data_x = tf.data.Dataset.from_tensor_slices(seq_data)
        data_y = tf.data.Dataset.from_tensor_slices(rst_data)
        data_y = data_y.map(lambda x: tf.one_hot(x, self.DATASET.DATA["N_ITEMS"]), num_parallel_calls=tf.data.AUTOTUNE)
        
        return tf.data.Dataset.zip((data_x, data_y))

    def get_train_dev_sequences(self, dev):

        all_data = self.DATASET.DATA["TRAIN_DEV"]

        if dev:
            train_data = all_data[all_data["dev"] == 0]
            dev_data = all_data[all_data["dev"] == 1]
            train_gn = self.__create_dataset(train_data)
            dev_gn = self.__create_dataset(dev_data)
            return train_gn, dev_gn
        else:
            train_dev_gn = self.__create_dataset(all_data)
            return train_dev_gn

    def evaluate(self, test=False):

        if test:
            test_set = MySequence(self, set_name="TEST")
        else:
            test_set = MySequence(self, is_dev=1)

        ret = self.MODEL.evaluate(test_set, verbose=0)

        print_g(dict(zip(self.MODEL.metrics_names, ret)))

    def evaluate_text(self, text):

        print("\n")
        print_g("\'%s\'" % text)

        n_rsts = 3
        lstm_text = self.DATASET.DATA["TEXT_TOKENIZER"].texts_to_sequences([self.DATASET.prerpocess_text(text)])
        lstm_text_pad = tf.keras.preprocessing.sequence.pad_sequences(lstm_text, maxlen=self.DATASET.DATA["MAX_LEN_PADDING"])

        # Obtener para la review al completo, el top 3 de restaurantes predichos por el modelo
        preds_rst = self.MODEL.predict(lstm_text_pad)
        preds_rst = np.argsort(-preds_rst.flatten())[:n_rsts]
        for rst in preds_rst:
            print("\t- %s" % (self.DATASET.DATA["TRAIN_DEV"].loc[self.DATASET.DATA["TRAIN_DEV"].id_restaurant == rst]["name"].values[0]))

        print("\t-------------")

        # Para cada una de las palabras de la review, obtener el restaurante que más importancia le da
        for w in lstm_text[0]:
            preds_wrd = self.MODEL.predict(tf.keras.preprocessing.sequence.pad_sequences([[w]], maxlen=self.DATASET.DATA["MAX_LEN_PADDING"])).flatten()
            restr_wrd = np.argsort(-preds_wrd.flatten())[:3]
            restr_wrd = self.DATASET.DATA["TRAIN_DEV"].loc[self.DATASET.DATA["TRAIN_DEV"].id_restaurant.isin(restr_wrd)]["name"].unique().tolist()
            print("\t ·%s => %s " % (list(self.DATASET.DATA["WORD_INDEX"].keys())[w-1], ", ".join(restr_wrd)))
