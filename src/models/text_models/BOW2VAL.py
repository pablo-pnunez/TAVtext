# -*- coding: utf-8 -*-
from src.Common import print_g
from src.models.text_models.VALModel import VALModel
from src.sequences.BaseSequence import BaseSequence

import numpy as np
import tensorflow as tf
from scipy.sparse import csc_matrix


class BOW2VAL(VALModel):
    """ Predecir, a partir de una review codificada mendiante BOW, la nota de la review """
    def __init__(self, config, dataset):
        VALModel.__init__(self, config=config, dataset=dataset)

    def get_model(self):

        mv = self.CONFIG["model"]["model_version"]
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(len(self.DATASET.DATA["FEATURES_NAME"]),), name="input_bow"))

        if mv == "0":
            model.add(tf.keras.layers.Dense(128, activation='relu'))
            model.add(tf.keras.layers.Dense(64, activation='relu'))
            model.add(tf.keras.layers.Dense(32, activation='relu'))
        if mv == "1":
            model.add(tf.keras.layers.Dropout(.4))
            model.add(tf.keras.layers.Dense(128, activation='relu'))
            model.add(tf.keras.layers.Dropout(.3))
            model.add(tf.keras.layers.Dense(64, activation='relu'))
            model.add(tf.keras.layers.Dropout(.2))
            model.add(tf.keras.layers.Dense(32, activation='relu'))
        if mv == "2":
            model.add(tf.keras.layers.Dense(128, activation='relu'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dense(64, activation='relu'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dense(32, activation='relu'))
        if mv == "3":
            model.add(tf.keras.layers.Dense(512, activation='relu'))
            model.add(tf.keras.layers.Dense(256, activation='relu'))
            model.add(tf.keras.layers.Dense(128, activation='relu'))
            model.add(tf.keras.layers.Dense(64, activation='relu'))
            model.add(tf.keras.layers.Dense(32, activation='relu'))

        model.add(tf.keras.layers.Dense(1))
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=self.CONFIG["model"]["learning_rate"]), metrics=['mean_absolute_error'])

        return model

    def get_train_dev_sequences(self):
        train = BOW2VALsequence(self, is_dev=0)
        dev = BOW2VALsequence(self, is_dev=1)

        return train, dev

    def evaluate(self, test=False):

        if test:
            test_set = BOW2VALsequence(self, set_name="TEST")
        else:
            test_set = BOW2VALsequence(self, is_dev=1)

        ret = self.MODEL.evaluate(test_set, verbose=0)
        ret = dict(zip(self.MODEL.metrics_names, ret))
        print_g(ret)
        
        return ret


class BOW2VALsequence(BaseSequence):

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
        # return np.row_stack(batch_data.bow)
        return np.row_stack(batch_data.bow.apply(lambda x: x.todense().tolist()[0]))

    def preprocess_output(self, batch_data):
        return batch_data["rating"].values
