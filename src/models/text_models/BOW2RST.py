# -*- coding: utf-8 -*-
from src.models.text_models.RSTModel import RSTModel
from src.sequences.BaseSequence import BaseSequence
from src.Common import print_g

import numpy as np
import tensorflow as tf
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
