# -*- coding: utf-8 -*-
from src.Common import print_g
from src.models.KerasModelClass import KerasModelClass
from src.sequences.BaseSequence import BaseSequence

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer


class BOW2RST(KerasModelClass):

    def __init__(self, config, dataset):
        KerasModelClass.__init__(self, config=config, dataset=dataset)

    def get_model(self):

        rest_input = tf.keras.layers.Input(shape=(self.DATASET.CONFIG["num_palabras"],), name="input_rest")
        # output = tf.keras.layers.Dense(self.DATASET.DATA["N_RST"], activation='softmax', name="output_layer")(rest_input)
        x = tf.keras.layers.Dense(self.DATASET.DATA["N_RST"], name="output_layer")(rest_input)
        x = tf.keras.layers.Dropout(.5)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        output = tf.keras.layers.Activation("softmax")(x)

        model = tf.keras.Model(inputs=[rest_input], outputs=[output])
        # model.summary()

        metrics = [
            # tf.keras.metrics.Accuracy(name='accuracy'),
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10'),
        ]
        # si utilizo este 'metrics' con tf.keras.metrics.Accuracy(name='accuracy'), no me calcula bien la accuracy ¿?

        model.compile(optimizer=tf.keras.optimizers.Adam(self.CONFIG["model"]["learning_rate"]), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=metrics,)

        return model

    def get_train_dev_sequences(self):
        train = BOW2RSTsequence(self, is_dev=0)
        dev = BOW2RSTsequence(self, is_dev=1)

        return train, dev


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
