# -*- coding: utf-8 -*-
from src.Common import print_g
from src.models.KerasModelClass import KerasModelClass
from src.sequences.BaseSequence import BaseSequence

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.models import KeyedVectors


class RSTModel(KerasModelClass):
    """ Métodos comunes para todos los modelos VAL """
    def __init__(self, config, dataset):
        KerasModelClass.__init__(self, config=config, dataset=dataset)

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

        res = dict(zip(["ACCURACY", "TOP5ACC", "TOP10ACC"], [m1.result().numpy(), m2.result().numpy(), m3.result().numpy()]))
        print_g(res)
        return res
