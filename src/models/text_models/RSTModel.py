# -*- coding: utf-8 -*-
from src.Common import print_g
from src.models.KerasModelClass import KerasModelClass
from src.sequences.BaseSequence import BaseSequence

import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import tensorflow_ranking as tfr
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

    def explain_test_sample(self, data_row):
        raise NotImplementedError

    def __create_tfdata__(self, data):
        raise NotImplementedError

    def get_train_dev_sequences(self, dev):
        all_data = self.DATASET.DATA["TRAIN_DEV"]

        if dev:
            train_data = all_data[all_data["dev"] == 0]
            dev_data = all_data[all_data["dev"] == 1]
            train_gn = self.__create_tfdata__(train_data)
            dev_gn = self.__create_tfdata__(dev_data)
            return train_gn, dev_gn
        else:
            train_dev_gn = self.__create_tfdata__(all_data)
            return train_dev_gn

    def evaluate(self, test=False):
        if test:
            test_data = self.DATASET.DATA["TEST"]
        else:
            test_data = self.DATASET.DATA["TRAIN_DEV"][self.DATASET.DATA["TRAIN_DEV"]["dev"] == 1]

        test_gn = self.__create_tfdata__(test_data)

        metrics = [
            tfr.keras.metrics.NDCGMetric(topn=1, name="NDCG@1"),
            tfr.keras.metrics.NDCGMetric(topn=10, name="NDCG@10"),
            tfr.keras.metrics.NDCGMetric(name="NDCG@-1"),
            tf.keras.metrics.Precision(top_k=1, name="Precision@1"),
            tf.keras.metrics.Precision(top_k=5, name="Precision@5"),
            tf.keras.metrics.Precision(top_k=10, name="Precision@10"),
            tf.keras.metrics.Recall(top_k=1, name="Recall@1"),
            tf.keras.metrics.Recall(top_k=5, name="Recall@5"),
            tf.keras.metrics.Recall(top_k=10, name="Recall@10")]

        print_g(f"There are {len(test_data)} evaluation examples.")

        self.MODEL.compile(loss=self.MODEL.loss, optimizer=self.MODEL.optimizer, metrics=metrics)
        ret = self.MODEL.evaluate(test_gn.cache().batch(self.CONFIG["model"]['batch_size']).prefetch(tf.data.AUTOTUNE), verbose=0, return_dict=True)
        
        preds = self.MODEL.predict(test_gn.cache().batch(self.CONFIG["model"]['batch_size']).prefetch(tf.data.AUTOTUNE), verbose=0)
        sm_all = []
        # metrics = [tfr.keras.metrics.NDCGMetric(name="NDCG@-1")]
        for idx in tqdm(range(len(preds))):
            y_pred = preds[idx]
            y_true = np.zeros(len(y_pred))
            y_true[test_data.iloc[idx].id_item] = 1
            ndcg = tfr.keras.metrics.NDCGMetric()
            sm_all.append([metric([y_true], [y_pred]).numpy() for metric in metrics])

        sm_all = pd.DataFrame(sm_all, columns=[f.name for f in metrics])
        sm_all = pd.concat([test_data.reset_index(drop=True), sm_all], axis=1)

        for r in [1, 5, 10]:
            r_at = ret[f"Recall@{r}"]
            p_at = ret[f"Precision@{r}"]
            f1_at = 2 * ((r_at * p_at) / (r_at + p_at))
            ret[f"F1@{r}"] = f1_at

        ret = pd.DataFrame([ret.values()], columns=ret.keys())
        print_g(ret, title=False)

        return ret  