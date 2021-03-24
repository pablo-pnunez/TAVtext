# -*- coding: utf-8 -*-
from src.models.KerasModelClass import KerasModelClass
from src.sequences.BaseSequence import BaseSequence

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer


class BOW2RST(KerasModelClass):
    """ Predecir, a partir de una review codificada mendiante BOW, el restaurante de la review """
    def __init__(self, config, dataset):
        KerasModelClass.__init__(self, config=config, dataset=dataset)

    def get_model(self):

        rest_input = tf.keras.layers.Input(shape=(self.DATASET.CONFIG["num_palabras"],), name="input_rest")
        # output = tf.keras.layers.Dense(self.DATASET.DATA["N_RST"], activation='softmax', name="output_layer")(rest_input)
        x = tf.keras.layers.Dense(self.DATASET.DATA["N_RST"], name="output_layer")(rest_input)
        x = tf.keras.layers.Dropout(.2)(x)
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

    def baseline(self, test=False):
        """ Se calcula la popularidad para que actúe como baseline y se convierte a probabilidad """
        if test:
            dataset = self.DATASET.DATA["TRAIN_DEV"]
        else:
            dataset = self.DATASET.DATA["TRAIN_DEV"].loc[self.DATASET.DATA["TRAIN_DEV"]["dev"] == 0]

        y_out = np.zeros((len(dataset), self.DATASET.DATA["N_RST"]))
        y_out[np.expand_dims(list(range(len(y_out))), -1), np.expand_dims(dataset.id_restaurant, -1)] = 1

        popularidad_vec = y_out.sum(axis=0) / sum(y_out.sum(axis=0))
        pred_popularidad = np.row_stack([popularidad_vec]*len(dataset))

        with tf.device('/cpu:0'):

            m1 = tf.keras.metrics.Accuracy()
            m1.update_state(y_true=y_out.argmax(axis=1), y_pred=pred_popularidad.argmax(axis=1))
            # print_g("ACCURACY por popularidad: %.4f" % (m1.result().numpy()))
            # m1.reset_states()

            m2 = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
            m2.update_state(y_true=y_out, y_pred=pred_popularidad)
            # print_g("TOP5ACC por popularidad:  %.4f" % (m2.result().numpy()))
            # m2.reset_states()

            m3 = tf.keras.metrics.TopKCategoricalAccuracy(k=10)
            m3.update_state(y_true=y_out, y_pred=pred_popularidad)
            # print_g("TOP10ACC por popularidad:  %.4f" % (m3.result().numpy()))
            # m3.reset_states()

        return m1.result().numpy(), m2.result().numpy(), m3.result().numpy()

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

        return ret[1:]


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
