# -*- coding: utf-8 -*-
from src.Common import print_g
from src.models.KerasModelClass import KerasModelClass
from src.sequences.BaseSequence import BaseSequence

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.models import KeyedVectors


class VALModel(KerasModelClass):
    """ Métodos comunes para todos los modelos VAL """
    def __init__(self, config, dataset):
        KerasModelClass.__init__(self, config=config, dataset=dataset)

    def baseline(self, test=False):
        """ Predecir la media """

        if not test:
            the_mean = self.DATASET.DATA["TRAIN_DEV"].loc[self.DATASET.DATA["TRAIN_DEV"].dev == 0].rating.mean()
            mae = np.abs(the_mean - self.DATASET.DATA["TRAIN_DEV"].loc[self.DATASET.DATA["TRAIN_DEV"].dev == 1].rating.values).mean()
            mse = np.power(the_mean - self.DATASET.DATA["TRAIN_DEV"].loc[self.DATASET.DATA["TRAIN_DEV"].dev == 1].rating.values, 2).mean()
        else:
            the_mean = self.DATASET.DATA["TRAIN_DEV"].rating.mean()
            mae = np.abs(the_mean - self.DATASET.DATA["TEST"].rating.values).mean()
            mse = np.power(the_mean - self.DATASET.DATA["TEST"].rating.values, 2).mean()

        ttl = "TEST" if test else "DEV"

        print_g("%s baseline MSE: %.4f  MAE: %.4f" % (ttl, mse, mae))
