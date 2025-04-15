# -*- coding: utf-8 -*-
from bokeh.plotting import ColumnDataSource, figure, output_file, save
from scipy.spatial.distance import cdist, pdist, squareform
from bokeh.models import LinearColorMapper
from sklearn.manifold import TSNE
import tensorflow_ranking as tfr
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from bokeh.models import Span
from scipy.stats import skew
import tensorflow as tf
import seaborn as sns
import pickle as pkl
import pandas as pd
import numpy as np
import re

from src.Common import print_g, print_e
from src.Metrics import f1 as f1_score
from src.models.Common import create_weighted_binary_crossentropy, weighted_binary_crossentropy
from src.models.text_models.RSTModel import RSTModel
from src.sequences.BaseSequence import BaseSequence

import keras_nlp


class ATT2ITM(RSTModel):
    """ Predecir, a partir de los embeddings de una review y los de los restaruantes, el restaurante de la review """

    def __init__(self, config, dataset):
        RSTModel.__init__(self, config=config, dataset=dataset)

    def get_model(self):
        model = self.get_sub_model()
        return model

    def get_sub_model(self):

        mv = self.CONFIG["model"]["model_version"]
        self.MODEL_VERSION = mv

        rst_no = self.DATASET.DATA["N_ITEMS"]
        pad_len = self.DATASET.DATA["MAX_LEN_PADDING"]
        vocab_size = self.DATASET.DATA["VOCAB_SIZE"]

        text_in = tf.keras.Input(shape=(pad_len), dtype='int32')
        rest_in = tf.keras.Input(shape=(rst_no), dtype='int32')
        
        model_out = None
        model_out2 = None

        if mv == "0" or mv == "1":  # Modelo básico sin heads, solo una capa oculta

            emb_size = 128

            # init = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
            use_bias = True
            
            # word_importance = tf.keras.layers.Embedding(vocab_size, 1, name="word_importance", embeddings_initializer="ones", mask_zero=True)(text_in)

            query_emb = tf.keras.layers.Embedding(vocab_size, emb_size * 3, mask_zero=True)
            mask_query = query_emb.compute_mask(text_in)
            mask_query = tf.expand_dims(tf.cast(mask_query, dtype=tf.float32), -1)
            mask_query = tf.tile(mask_query, [1, 1, rst_no])
            ht_emb = query_emb(text_in)
            # ht_emb = tf.keras.layers.Activation("tanh")(ht_emb)
            ht_emb = tf.keras.layers.Dense(emb_size * 2, use_bias=use_bias)(ht_emb)
            # ht_emb = tf.keras.layers.Activation("tanh")(ht_emb)
            ht_emb = tf.keras.layers.Dense(emb_size, use_bias=use_bias)(ht_emb)
            # ht_emb = tf.keras.layers.Activation("tanh")(ht_emb)
           
            '''
            wr_att_embs = tf.keras.layers.Embedding(vocab_size, 32, mask_zero=True, name="wr_att_embs")
            wr_att_embs = wr_att_embs(text_in)
            wr_att_embs = tf.keras.layers.Dropout(.8)(wr_att_embs)     
            wr_attention_scores = tf.matmul(wr_att_embs, wr_att_embs, transpose_b=True)
            ht_emb = tf.einsum("abc,acd->abd", wr_attention_scores, ht_emb)
            '''            

            ht_emb = tf.keras.layers.Lambda(lambda x: x, name="word_emb")(ht_emb)

            '''
            position_embeddings = keras_nlp.layers.SinePositionEncoding()(ht_emb)
            ht_emb = ht_emb + position_embeddings
            '''
            
            rests_emb = tf.keras.layers.Embedding(rst_no, emb_size * 3, name=f"in_rsts")
            hr_emb = rests_emb(rest_in)
            # hr_emb = tf.keras.layers.Activation("tanh")(hr_emb)
            hr_emb = tf.keras.layers.Dense(emb_size * 2, use_bias=use_bias)(hr_emb)
            # hr_emb = tf.keras.layers.Activation("tanh")(hr_emb)
            hr_emb = tf.keras.layers.Dense(emb_size, use_bias=use_bias)(hr_emb)
            # hr_emb = tf.keras.layers.Activation("tanh")(hr_emb)
            hr_emb = tf.keras.layers.Lambda(lambda x: x, name="rest_emb")(hr_emb)

            model = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([ht_emb, hr_emb])
            # model = tf.keras.layers.BatchNormalization()(model)
            # model = tf.keras.layers.GaussianNoise(.5)(model)
            
            model = tf.keras.layers.Lambda(lambda x: x[0]*x[1])([model, mask_query])
            model = tf.keras.layers.Activation("tanh", name="dotprod")(model)
            # model = tf.keras.layers.Lambda(lambda x: x[0]*x[1], name="importance")([model, word_importance])
            # model = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x[0], 2), name="dotprod")([model])

            model = tf.keras.layers.Dropout(.4)(model)

            model = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1), name="sum")(model)
            # model = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x[0], 1)/tf.math.reduce_sum(x[1], 1), name="sum")([model, mask_query])
            ##  model = tf.keras.layers.Dropout(.1)(model)

            model_out = tf.keras.layers.Activation("sigmoid", name="out", dtype='float32')(model)
            # model_out = tf.keras.layers.Activation("linear", name="out", dtype='float32')(model)

            model = tf.keras.models.Model(inputs=[text_in, rest_in], outputs=[model_out], name=f"{self.MODEL_NAME}_{self.MODEL_VERSION}")

            optimizer = tf.keras.optimizers.legacy.Adam(self.CONFIG["model"]["learning_rate"])

            metrics = [tfr.keras.metrics.NDCGMetric(topn=10, name="NDCG@10"), tfr.keras.metrics.RecallMetric(topn=1, name='r1'), tfr.keras.metrics.RecallMetric(topn=5, name='r5'), tfr.keras.metrics.RecallMetric(topn=10, name='r10'),
                       tfr.keras.metrics.PrecisionMetric(topn=5, name='p5'), tfr.keras.metrics.PrecisionMetric(topn=10, name='p10'), f1_score]

            if mv == "0":
                loss = tf.keras.losses.CategoricalCrossentropy()  # Prueba
            elif mv == "1":
                loss = tf.keras.losses.BinaryFocalCrossentropy()

        if mv == "2": # Se añade la estandarización a la salida y se simplifica el modelo
            emb_size = 45  # 128

            use_bias = True
            
            query_emb = tf.keras.layers.Embedding(vocab_size, emb_size, mask_zero=True)
            mask_query = query_emb.compute_mask(text_in)
            mask_query = tf.expand_dims(tf.cast(mask_query, dtype=tf.float32), -1)
            mask_query = tf.tile(mask_query, [1, 1, rst_no])
            ht_emb = query_emb(text_in)
            # ht_emb = tf.keras.layers.Activation("relu")(ht_emb)
            # ht_emb = tf.keras.layers.Dense(emb_size, use_bias=use_bias)(ht_emb)
            ht_emb = tf.keras.layers.Lambda(lambda x: x, name="word_emb")(ht_emb)
            ht_emb = tf.keras.layers.Dropout(.1)(ht_emb)

            rests_emb = tf.keras.layers.Embedding(rst_no, emb_size, name=f"in_rsts")
            hr_emb = rests_emb(rest_in)
            # hr_emb = tf.keras.layers.Activation("relu")(hr_emb)
            # hr_emb = tf.keras.layers.Dense(emb_size, use_bias=use_bias)(hr_emb)
            hr_emb = tf.keras.layers.Lambda(lambda x: x, name="rest_emb")(hr_emb)
            hr_emb = tf.keras.layers.Dropout(.1)(hr_emb)

            model = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([ht_emb, hr_emb])
            
            model = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([model, mask_query])
            model = tf.keras.layers.Activation("tanh", name="dotprod")(model)

            model = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1), name="sum")(model)
            # model = tfa.layers.InstanceNormalization(axis=-1)(model)
            model = tf.keras.layers.Normalization()(model)
            # model = tf.keras.layers.LayerNormalization()(model)  # mean 0 std 1
            # model = tf.keras.layers.BatchNormalization()(model)
        
            model_out = tf.keras.layers.Activation("sigmoid", name="out", dtype='float32')(model)
            model = tf.keras.models.Model(inputs=[text_in, rest_in], outputs=[model_out], name=f"{self.MODEL_NAME}_{self.MODEL_VERSION}")

            optimizer = tf.keras.optimizers.legacy.Adam(self.CONFIG["model"]["learning_rate"])

            metrics = [tfr.keras.metrics.NDCGMetric(topn=10, name="NDCG@10"), tf.keras.metrics.AUC(name="AUC"), tfr.keras.metrics.RecallMetric(topn=1, name='r1'), 
                       tfr.keras.metrics.PrecisionMetric(topn=1, name='p1'), tfa.metrics.F1Score(num_classes=rst_no, name="F1")]

            loss = tf.keras.losses.CategoricalCrossentropy()
 
        model.compile(loss=loss, metrics=metrics, optimizer=optimizer)


        '''
        if mv == "0" or mv == "1":  # Modelo básico sin heads, solo una capa oculta

            emb_size = 128

            # init = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
            use_bias = True
            
            # word_importance = tf.keras.layers.Embedding(vocab_size, 1, name="word_importance", embeddings_initializer="ones", mask_zero=True)(text_in)

            query_emb = tf.keras.layers.Embedding(vocab_size, emb_size*3, mask_zero=True)
            mask_query = query_emb.compute_mask(text_in)
            mask_query = tf.expand_dims(tf.cast(mask_query, dtype=tf.float32), -1)
            mask_query = tf.tile(mask_query, [1, 1, rst_no])
            ht_emb = query_emb(text_in)
            # ht_emb = tf.keras.layers.Activation("tanh")(ht_emb)
            ht_emb = tf.keras.layers.Dense(emb_size*2, use_bias=use_bias)(ht_emb)
            # ht_emb = tf.keras.layers.Activation("tanh")(ht_emb)
            ht_emb = tf.keras.layers.Dense(emb_size, use_bias=use_bias)(ht_emb)

            # ht_emb = tf.keras.layers.Activation("tanh")(ht_emb)
            ht_emb = tf.keras.layers.Lambda(lambda x: x, name="word_emb")(ht_emb)

            rests_emb = tf.keras.layers.Embedding(rst_no, emb_size*3, name=f"in_rsts")
            hr_emb = rests_emb(rest_in)
            # hr_emb = tf.keras.layers.Activation("tanh")(hr_emb)
            hr_emb = tf.keras.layers.Dense(emb_size*2, use_bias=use_bias)(hr_emb)
            # hr_emb = tf.keras.layers.Activation("tanh")(hr_emb)
            hr_emb = tf.keras.layers.Dense(emb_size, use_bias=use_bias)(hr_emb)
            # hr_emb = tf.keras.layers.Activation("tanh")(hr_emb)
            hr_emb = tf.keras.layers.Lambda(lambda x: x, name="rest_emb")(hr_emb)

            model = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([ht_emb, hr_emb])
            # model = tf.keras.layers.BatchNormalization()(model)
            # model = tf.keras.layers.GaussianNoise(.5)(model)
            
            model = tf.keras.layers.Lambda(lambda x: x[0]*x[1])([model, mask_query])
            model = tf.keras.layers.Activation("tanh", name="dotprod")(model)
            # model = tf.keras.layers.Lambda(lambda x: x[0]*x[1], name="importance")([model, word_importance])
            # model = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x[0], 2), name="dotprod")([model])

            model = tf.keras.layers.Dropout(.4)(model)

            model = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1),  name="sum")(model)
            if mv == "0":  # Modelo básico con softmax
                model_out = tf.keras.layers.Activation("sigmoid", name="out", dtype='float32')(model)
            else:
                model_out = tf.keras.layers.Activation("softmax", name="out", dtype='float32')(model)

        if mv == "2" or mv == "3":  # Modelo básico sin heads, solo una capa oculta

            emb_size = 512
            h_size = 128

            # init = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
            use_bias = True
            
            # word_importance = tf.keras.layers.Embedding(vocab_size, 1, name="word_importance", embeddings_initializer="ones", mask_zero=True)(text_in)

            query_emb = tf.keras.layers.Embedding(vocab_size, emb_size, mask_zero=True)
            mask_query = query_emb.compute_mask(text_in)
            mask_query = tf.expand_dims(tf.cast(mask_query, dtype=tf.float32), -1)
            mask_query = tf.tile(mask_query, [1, 1, rst_no])
            ht_emb = query_emb(text_in)
            ht_emb = tf.keras.layers.Activation("tanh")(ht_emb)
            ht_emb = tf.keras.layers.Dense(h_size*2, use_bias=use_bias)(ht_emb)
            ht_emb = tf.keras.layers.Activation("tanh")(ht_emb)
            ht_emb = tf.keras.layers.Dense(h_size, name="word_emb", use_bias=use_bias)(ht_emb)

            rests_emb = tf.keras.layers.Embedding(rst_no, emb_size, name=f"in_rsts")
            hr_emb = rests_emb(rest_in)
            hr_emb = tf.keras.layers.Activation("tanh")(hr_emb)
            hr_emb = tf.keras.layers.Dense(h_size*2, use_bias=use_bias)(hr_emb)
            hr_emb = tf.keras.layers.Activation("tanh")(hr_emb)
            hr_emb = tf.keras.layers.Dense(h_size, name="rest_emb", use_bias=use_bias)(hr_emb)

            model = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([ht_emb, hr_emb])
            # model = tf.keras.layers.BatchNormalization()(model)
            # model = tf.keras.layers.GaussianNoise(.5)(model)
            
            # model = tf.keras.layers.Lambda(lambda x: x[0]*x[1])([model, mask_query])
            model = tf.keras.layers.Activation("tanh", name="dotprod")(model)
            # model = tf.keras.layers.Lambda(lambda x: x[0]*x[1], name="importance")([model, word_importance])
            # model = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x[0], 2), name="dotprod")([model])

            model = tf.keras.layers.Dropout(.2)(model)

            model = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1),  name="sum")(model)
            if mv == "2":  # Modelo básico con softmax
                model_out = tf.keras.layers.Activation("sigmoid", name="out", dtype='float32')(model)
            else:
                model_out = tf.keras.layers.Activation("softmax", name="out", dtype='float32')(model)

        if mv == "4":  # Modelo de Oscar con máscaras

            def compute_attention_mask(query_emb_mask, key_emb_mask=None):
                if key_emb_mask is None: 
                    key_emb_mask = query_emb_mask
                att_mask = tf.logical_and(tf.expand_dims(query_emb_mask, 2), tf.expand_dims(key_emb_mask, 1))
                return att_mask

            key_emb_size = 128
            value_emb_size = key_emb_size

            query_emb = tf.keras.layers.Embedding(vocab_size, w2v_emb_size, name="emb_text", weights=[embedding_matrix], trainable=False, mask_zero=True)
            rests_emb = tf.keras.layers.Embedding(rst_no, rst_no, name="emb_rests", embeddings_initializer=tf.constant_initializer(np.eye(rst_no)), trainable=False, mask_zero=True)

            attention_mask = compute_attention_mask(query_emb.compute_mask(text_in), rests_emb.compute_mask(rest_in))

            query_emb = query_emb(text_in)
            rests_emb = rests_emb(rest_in)

            att_ly = tf.keras.layers.MultiHeadAttention(num_heads=6, output_shape=rst_no, key_dim=key_emb_size, value_dim=value_emb_size, name="att", dropout=0.0)
            model, attention = att_ly(query=query_emb, value=rests_emb, return_attention_scores=True, attention_mask=attention_mask)
            model = tf.keras.layers.GlobalAveragePooling1D()(model)
            model_out = tf.keras.layers.Activation("softmax", name="out", dtype='float32')(model)

        if mv == "5":  # El modelo de Oscar cambiando el orden de la entrada

            key_emb_size = 128
            value_emb_size = key_emb_size

            query_emb = tf.keras.layers.Embedding(vocab_size, w2v_emb_size, name="emb_text", weights=[embedding_matrix], trainable=False, mask_zero=True)
            rests_emb = tf.keras.layers.Embedding(rst_no, rst_no, name="emb_rests", embeddings_initializer=tf.constant_initializer(np.eye(rst_no)), trainable=False, mask_zero=True)

            query_emb = query_emb(text_in)
            rests_emb = rests_emb(rest_in)

            att_ly = tf.keras.layers.MultiHeadAttention(num_heads=4, output_shape=rst_no, key_dim=key_emb_size, value_dim=value_emb_size, name="att", dropout=0.5)
            model, attention = att_ly(query=rests_emb, value=query_emb, return_attention_scores=True)
            model = tf.keras.layers.GlobalAveragePooling1D()(model)
            model_out = tf.keras.layers.Activation("softmax", name="out", dtype='float32')(model)

        if mv == "6":  # Modelo básico sin heads, solo una capa oculta

            emb_size = 512
            h_size = 128

            # init = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
            use_bias = False
            
            # word_importance = tf.keras.layers.Embedding(vocab_size, 1, name="word_importance", embeddings_initializer="ones", mask_zero=True)(text_in)

            # kernel_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05)
            # bias_initializer = tf.keras.initializers.Zeros()

            embedding_matrix = np.random.normal(size=(vocab_size, emb_size))
            embedding_matrix[0, :] = np.zeros(emb_size)

            #  Embeddings de palabras ---------------------------------------------------------
            query_emb = tf.keras.layers.Embedding(vocab_size, emb_size, mask_zero=True, weights=[embedding_matrix])
            mask_query = query_emb.compute_mask(text_in)
            mask_query = tf.expand_dims(tf.cast(mask_query, dtype=tf.float32), -1)
            mask_query = tf.tile(mask_query, [1, 1, rst_no])
            ht_emb = query_emb(text_in)
            # ht_emb = tf.keras.layers.Activation("tanh")(ht_emb)
            ht_emb = tf.keras.layers.Dense(h_size*2, use_bias=use_bias)(ht_emb)
            # ht_emb = tf.keras.layers.Activation("tanh")(ht_emb)
            ht_emb = tf.keras.layers.Dense(h_size, use_bias=use_bias)(ht_emb)
            # ht_emb = tf.keras.layers.Activation("tanh")(ht_emb)
            # ht_emb = tf.keras.layers.Dropout(.5)(ht_emb)
            ht_emb = tf.keras.layers.Lambda(lambda x: x, name="word_emb")(ht_emb)

            #  Embeddings de restaurantes ------------------------------------------------------

            rests_emb = tf.keras.layers.Embedding(rst_no, emb_size, name=f"in_rsts")
            # hr_emb = rests_emb(np.expand_dims(np.arange(rst_no),0))
            hr_emb = rests_emb(rest_in)
            # hr_emb = tf.Variable(tf.random.normal([rst_no, emb_size], dtype="float16"))
            # hr_emb = tf.keras.layers.Activation("tanh")(hr_emb)
            hr_emb = tf.keras.layers.Dense(h_size*2, use_bias=use_bias)(hr_emb)
            # hr_emb = tf.keras.layers.Activation("tanh")(hr_emb)
            hr_emb = tf.keras.layers.Dense(h_size, use_bias=use_bias)(hr_emb)
            # hr_emb = tf.keras.layers.Activation("tanh")(hr_emb)
            # hr_emb = tf.keras.layers.Dropout(.5)(hr_emb)
            hr_emb = tf.keras.layers.Lambda(lambda x: x, name="rest_emb")(hr_emb)

            #  DOT y final ---------------------------------------------------------------------
            model = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([ht_emb, hr_emb])
            model = tf.keras.layers.Activation("tanh")(model)
            # model = tf.keras.layers.Lambda(lambda x: x[0]*x[1])([model, mask_query])
            model = tf.keras.layers.Lambda(lambda x: x, name="dotprod")(model)

            model = tf.keras.layers.Dropout(.2)(model)

            suma_r = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1),  name="sum_r")(model)

            # model_out2 = tf.keras.layers.Activation("softmax", name="out_smx", dtype='float32')(suma_r)
            model_out = tf.keras.layers.Activation("sigmoid", name="out_smd", dtype='float32')(suma_r)

        if mv == "2D":  # Modelo básico que va a 2D

            emb_size = 64
            h_size = 2

            query_emb = tf.keras.layers.Embedding(vocab_size, emb_size, name="w2v_text", mask_zero=True)
            mask_query = query_emb.compute_mask(text_in)
            mask_query = tf.expand_dims(tf.cast(mask_query, dtype=tf.float32), -1)
            mask_query = tf.tile(mask_query, [1, 1, rst_no])
            ht_emb = query_emb(text_in)
            ht_emb = tf.keras.layers.Activation("tanh")(ht_emb)
            ht_emb = tf.keras.layers.Dense(h_size*2)(ht_emb)
            ht_emb = tf.keras.layers.Activation("tanh")(ht_emb)
            ht_emb = tf.keras.layers.Dense(h_size, name="word_emb")(ht_emb)

            rests_emb = tf.keras.layers.Embedding(rst_no, emb_size, name=f"in_rsts")
            hr_emb = rests_emb(rest_in)
            hr_emb = tf.keras.layers.Activation("tanh")(hr_emb)
            hr_emb = tf.keras.layers.Dense(h_size*2)(hr_emb)
            hr_emb = tf.keras.layers.Activation("tanh")(hr_emb)
            hr_emb = tf.keras.layers.Dense(h_size, name="rest_emb")(hr_emb)

            model = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([ht_emb, hr_emb])
            # model = tf.keras.layers.BatchNormalization()(model)
            # model = tf.keras.layers.GaussianNoise(.5)(model)

            model = tf.keras.layers.Lambda(lambda x: x[0]*x[1])([model, mask_query])
            model = tf.keras.layers.Activation("tanh", name="dotprod")(model)

            model = tf.keras.layers.Dropout(.2)(model)

            model = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1),  name="sum")(model)
            model_out = tf.keras.layers.Activation("softmax", name="out", dtype='float32')(model)

        if model_out2 is not None:
                
            model = tf.keras.models.Model(inputs=[text_in, rest_in], outputs=[model_out, model_out2], name=f"{self.MODEL_NAME}_{self.MODEL_VERSION}")
            metrics = [tfr.keras.metrics.RecallMetric(topn=1, name='r1')]
            optimizer = tf.keras.optimizers.Adam(self.CONFIG["model"]["learning_rate"])
            loss = [tf.keras.losses.BinaryFocalCrossentropy(), tf.keras.losses.CategoricalCrossentropy()]
            model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
        
        else:
            model = tf.keras.models.Model(inputs=[text_in, rest_in], outputs=[model_out], name=f"{self.MODEL_NAME}_{self.MODEL_VERSION}")
            activation_function = model.layers[-1].activation.__name__
            optimizer = tf.keras.optimizers.Adam(self.CONFIG["model"]["learning_rate"])
            metrics = [tfr.keras.metrics.RecallMetric(topn=1, name='r1'), tfr.keras.metrics.RecallMetric(topn=5, name='r5'), tfr.keras.metrics.RecallMetric(topn=10, name='r10'),
                    tfr.keras.metrics.PrecisionMetric(topn=5, name='p5'), tfr.keras.metrics.PrecisionMetric(topn=10, name='p10')]

            if activation_function == "sigmoid":
                # loss = create_weighted_binary_crossentropy(1, 1000)
                # loss = weighted_binary_crossentropy(.1, 100)
                # loss = tf.keras.losses.BinaryCrossentropy()
                # loss = tf.keras.losses.BinaryFocalCrossentropy()
                loss = tf.keras.losses.CategoricalCrossentropy()  # Prueba

            elif activation_function == "softmax":
                loss = tf.keras.losses.CategoricalCrossentropy()

            model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
        '''
        
        # print(model.summary())
        return model

    def __create_tfdata__(self, dataframe):

        seq_data = self.DATASET.DATA["TEXT_SEQUENCES"][dataframe.seq.values]
        rst_data = dataframe.id_item.values
        
        data_y = tf.data.Dataset.from_tensor_slices(rst_data)
        data_y = data_y.map(lambda x: tf.one_hot(x, self.DATASET.DATA["N_ITEMS"]), num_parallel_calls=tf.data.AUTOTUNE)
        
        data_x_text = tf.data.Dataset.from_tensor_slices(seq_data)
        data_x_item = tf.data.Dataset.from_tensor_slices([range(self.DATASET.DATA["N_ITEMS"])]).repeat(len(dataframe))
        data_x = tf.data.Dataset.zip((data_x_text, data_x_item))

        return tf.data.Dataset.zip((data_x, data_y))

    def emb_tsne(self):

        wrd_embs = tf.keras.models.Model(inputs=[self.MODEL.input[0]], outputs=[self.MODEL.get_layer("word_emb").output])
        rst_embs = tf.keras.models.Model(inputs=[self.MODEL.input[1]], outputs=[self.MODEL.get_layer("rest_emb").output])

        rst_embs = rst_embs.predict([list(range(self.DATASET.DATA["N_ITEMS"]))], verbose=0).squeeze()
        rest_names = self.DATASET.DATA["TRAIN_DEV"][["id_item", "name"]].sort_values("id_item").drop_duplicates().name.values.tolist()
        
        word_names = np.array(["UNK"]+list(self.DATASET.DATA["WORD_INDEX"].keys()))
        wrd_embs = wrd_embs.predict(np.expand_dims(list(range(self.DATASET.DATA["VOCAB_SIZE"])),-1), verbose=0).squeeze()
        
        ''' 
        # BUSCAR PALABRAS CERCANAS USANDO DISTANCIA COSENO
        wrd = "barbecue" 
        emb = wrd_embs[np.where(word_names==wrd)[0][0]]
        dists = pd.DataFrame(zip(word_names,cdist([emb], wrd_embs, "cosine")[0]), columns=["word", "dist"])
        print(dists.sort_values("dist").head(10))

        # BUSCAR ITEMS CERCANOS USANDO DISTANCIA COSENO
        itm = "McDonald's" 
        emb = rst_embs[np.where(np.array(rest_names)==itm)[0][0]]
        dists = pd.DataFrame(zip(rest_names,cdist([emb], rst_embs, "cosine")[0]), columns=["rest", "dist"])
        print(dists.sort_values("dist").head(10))

        # BUSCAR ITEMS CERCANOS A UNA PALABRA (COSENO)
        wrd = "pizza" 
        emb = wrd_embs[np.where(word_names==wrd)[0][0]]
        dists = pd.DataFrame(zip(rest_names,cdist([emb], rst_embs, "cosine")[0]), columns=["rest", "dist"])
        print(dists.sort_values("dist").head(10))
        
        # BUSCAR ITEMS CERCANOS A UNA PALABRA (DOT)
        wrd = "pizza" 
        emb = wrd_embs[np.where(word_names==wrd)[0][0]]
        dists = pd.DataFrame(zip(rest_names,cdist([emb], rst_embs, lambda x,y: -np.dot(x,y))[0]), columns=["rest", "dist"])
        print(dists.sort_values("dist").head(10))
        '''

        if wrd_embs.shape[1] > 2:
            tsne_r = TSNE(n_components=2, learning_rate="auto", init="pca", metric="cosine")
            tsne_w = TSNE(n_components=2, learning_rate="auto", init="pca", metric="cosine")
            rst_tsne = tsne_r.fit_transform(rst_embs)
            wrd_tsne = tsne_w.fit_transform(wrd_embs)
        else:
            rst_tsne = rst_embs
            wrd_tsne = wrd_embs

        source_w = ColumnDataSource(data=dict(x=wrd_tsne[:, 0], y=wrd_tsne[:, 1], desc=word_names))
        source_r = ColumnDataSource(data=dict(x=rst_tsne[:, 0], y=rst_tsne[:, 1], desc=rest_names))

        # TOOLTIPS = [("index", "$index"), ("(x,y)", "($x, $y)"), ("desc", "@desc")]
        TOOLTIPS = [("Name", "@desc")]
        p = figure(width=800, height=800, tooltips=TOOLTIPS)
        p.scatter('x', 'y', size=5, source=source_w, color="red")
        output_file(filename=self.MODEL_PATH+"tsne_w.html", title="t-SNE palabras")     
        save(p)

        p = figure(width=800, height=800, tooltips=TOOLTIPS)
        p.scatter('x', 'y', size=5, source=source_r)
        output_file(filename=self.MODEL_PATH+"tsne_r.html", title="t-SNE restaurantes")
        save(p)

        print(rst_tsne.shape)

    def all_words_analysis(self):

        preds_rst_arg = list(range(self.DATASET.DATA["N_ITEMS"]))

        # Obtener los nombres de items y palabras relevantes
        item_relevant_words = self.item_relevant_words(preds_rst_arg, plot=True)
        longest_rest_name = max(list(map(lambda x: len(x["name"]), item_relevant_words.values())))

        for i, itm_idx in enumerate(preds_rst_arg):
            itm_data = item_relevant_words[itm_idx]
            print(f" {itm_data['name']:{longest_rest_name}s} {itm_data['words']}") 

    def word_analysis(self, word):
        
        seqs = self.DATASET.DATA["TEXT_SEQUENCES"]

        rst_embs = tf.keras.models.Model(inputs=[self.MODEL.input[1]], outputs=[self.MODEL.get_layer("rest_emb").output])
        rst_embs = rst_embs.predict([list(range(self.DATASET.DATA["N_ITEMS"]))], verbose=0).squeeze()
        rst_names = self.DATASET.DATA["TRAIN_DEV"].sort_values("id_item")[["id_item", "name"]].drop_duplicates()["name"].values

        word_prepro = word  # self.DATASET.prerpocess_text(word)
        word_index = self.DATASET.DATA["TEXT_TOKENIZER"].word_index[word_prepro]
        word_freq = self.DATASET.DATA["TEXT_TOKENIZER"].word_counts[word_prepro]
        print(f"- {word} {'-'*10}\n  index: {word_index}\n   freq: {word_freq}")

        wrd_emb = tf.keras.models.Model(inputs=[self.MODEL.input[0]], outputs=[self.MODEL.get_layer("word_emb").output])
        wrd_emb = wrd_emb.predict(list([word_index]), verbose=0).squeeze()

        wrd_vs_rsts = np.hstack((np.expand_dims(wrd_emb,-1), rst_embs.T))
        wrd_vs_rsts = pd.DataFrame(wrd_vs_rsts, columns=[word]+rst_names.tolist())
        wrd_vs_rsts.to_excel(self.MODEL_PATH+f"embs_{word}.xlsx")

        reviews_word = self.DATASET.DATA["TRAIN_DEV"][self.DATASET.DATA["TRAIN_DEV"].seq.apply(lambda x: word_index in seqs[x])]
        word_in_rest = len(reviews_word.id_item.unique())
        word_in_users = len(reviews_word.userId.unique())        
        most_times_rest = reviews_word.name.value_counts().reset_index().iloc[0]
        print(f" n_revs: {len(reviews_word)}\n n_rest: {word_in_rest}\n n_usrs: {word_in_users}\n f_rest: {most_times_rest[0]} ({most_times_rest[1]} times)")
        print(f"std_rst: {reviews_word.name.value_counts().std()}")
        att_md = tf.keras.models.Model(inputs=[self.MODEL.input], outputs=[self.MODEL.get_layer("dotprod").output])
        lstm_text_pad = tf.keras.preprocessing.sequence.pad_sequences([[word_index]], maxlen=self.DATASET.DATA["MAX_LEN_PADDING"])

        # Obtener compatibilidad de la palabra con todos los restaurantes
        word_att = att_md.predict([lstm_text_pad, np.arange(self.DATASET.DATA["N_ITEMS"])[None,:]], verbose=0).squeeze(axis=0)
        word_att = word_att[-1, :]

        # Dibujar la distribución de la palabra
        # hp = sns.histplot(data=word_att, bins=20, stat="density")
        # sns.kdeplot(word_att, color="r")
        plt.figure(figsize=(6, 3))  # Tamaño del plot
        # hp = sns.histplot(data=word_att, bins=20, stat="percent")
        hp = sns.kdeplot(word_att, color="r")

        hp.set_title("Word: "+word.title())
        hp.set_xlim([-1, 1])
        plt.savefig(f"{self.MODEL_PATH}dist_{word}.pdf")
        plt.close()

        most_prob_rest = self.DATASET.DATA["TRAIN_DEV"][self.DATASET.DATA["TRAIN_DEV"]["id_item"] == word_att.argmax()]["name"].values[0]
        less_prob_rest = self.DATASET.DATA["TRAIN_DEV"][self.DATASET.DATA["TRAIN_DEV"]["id_item"] == word_att.argmin()]["name"].values[0]

        word_att_df = pd.DataFrame(word_att, columns=[word])
        word_att_df.insert(0, "rest", rst_names)
        word_att_df.to_excel(self.MODEL_PATH+word+".xlsx")

        print(f"min_prb: {word_att.min()}\nmin_rst: {less_prob_rest}\nmax_prb: {word_att.max()}\nmax_rst: {most_prob_rest}\nstd_prb: {word_att.std()}")
        print("-"*30)

    def item_relevant_words(self, items, words_shown=10, plot=False):
  
        '''
        Dado id retorna nombre y palabras relevantes
        '''
        # Obtener todas las reviews en forma de secuencias
        seqs = self.DATASET.DATA["TEXT_SEQUENCES"]

        # Obtenemos la matriz de attention entera
        all_att, itm_names, word_names, itm_embs, word_embs = self.get_item_word_att()

        with open(self.MODEL_PATH + 'all_att.pkl', 'wb') as f: pkl.dump(all_att, f)
        with open(self.MODEL_PATH + 'itm_names.pkl', 'wb') as f: pkl.dump(itm_names, f)
        with open(self.MODEL_PATH + 'word_names.pkl', 'wb') as f: pkl.dump(word_names, f)

        att_std = np.std(all_att, -1)
        att_mean = np.mean(all_att, -1)
        att_mean_std = np.abs(att_mean) + att_std
        all_std_pct = np.argsort(-att_std)[:int(len(att_std) * .10)]

        # Hacer un gráfico 2d pintando, para cada palabra, media y std
        if plot:
            colors = att_mean_std

            # Versión dinámica
            source_w = ColumnDataSource(data=dict(x=att_mean, y=att_std, desc=word_names, col=colors))
            TOOLTIPS = [("Name", "@desc"),("Color value", "@col")]
            p = figure(width=1500, height=600, tooltips=TOOLTIPS)
            lc = LinearColorMapper(palette="Magma256", low=min(colors), high=max(colors))
            p.scatter('x', 'y', size=5, source=source_w, line_color=None, fill_color={"field": "col", "transform": lc})

            p.xaxis.axis_label = 'MEAN'
            p.yaxis.axis_label = 'STD'
            # Vertical & Horizontal lines
            vline = Span(location=0, dimension='height', line_color='black', line_width=2)
            hline = Span(location=0, dimension='width', line_color='black', line_width=2)
            vlineh = Span(location=0.5, dimension='height', line_color='gray', line_width=1)
            hlineh = Span(location=-0.5, dimension='width', line_color='gray', line_width=1)
            p.renderers.extend([vline, hline, vlineh, hlineh])
            output_file(filename=self.MODEL_PATH + "all_words.html", title="All word analysis")
            save(p)

            # Versión estática
            cm = 2 / 2.54  # centimeters in inches
            sns.set(style="ticks")
            plt.figure(figsize=(20 * cm, 8 * cm))  # Tamaño del plot
            sp = sns.scatterplot(x=att_mean, y=att_std, s=10, linewidth=0, hue=colors, palette="rocket")
            sp.axvline(0, color="black")

            sp.set_title("All words")
            sp.set_xlabel("Mean")
            sp.set_ylabel("STD")
            # sp.set_xlim([-1, 1])
            plt.tight_layout()
            plt.legend('', frameon=False)
            plt.grid()
            plt.savefig(self.MODEL_PATH + "all_words.pdf")
            plt.close()

        ret = {}

        for item_pos, item_idx in enumerate(items):
            # Para las palabras del restaurante, ordenar de mayor a menor att

            # Obtener las palabras que aparecen en el restaurante
            rst_reviews = self.DATASET.DATA["TRAIN_DEV"][self.DATASET.DATA["TRAIN_DEV"].id_item == item_idx]
            rst_seqs = seqs[rst_reviews.seq.values]
            rst_word_ids = np.unique(np.concatenate(seqs[rst_reviews.seq.values]))

            # Obtener la att para el restaurante (todas las palabras y solo las del restaurante)
            att_rst_words = all_att[rst_word_ids, item_idx]

            # Ordenar las palabras del restaurante por su attention y coger las mayores que 0
            rst_words_sorted = np.argsort(att_rst_words)[::-1]
            rst_words_sorted = rst_words_sorted[np.where(att_rst_words[rst_words_sorted] > 0)]
            rst_words_sorted = rst_word_ids[rst_words_sorted]
            rst_words_names = np.asarray(word_names)[rst_words_sorted]

            # Eliminar las que no aparecen en el restaurante
            # rst_words_names = np.delete(rst_words_names, np.argwhere(rst_words_names=="UNK")[0][0])
            # rst_words_ids = list(map(lambda x: self.DATASET.DATA["TEXT_TOKENIZER"].word_index[x], rst_words_names))
            # rst_words_names = np.array(rst_words_names)[[i in rst_seqs for i in rst_words_ids]]

            # Otros filtros posibles
            # filter_type = 4
            # if filter_type == 1:  # Están las palabras más positivas en las reviews del restaurante?
            #     rst_words_names = np.array(rst_words_names)[[i in rst_seqs for i in rst_words_sorted]]
            # elif filter_type == 2:  # 10% con mayor STD en att
            #     rst_words_names = np.array(rst_words_names)[[i in all_std_pct for i in rst_words_sorted]]
            # # elif filter_type == 3:  # 10% con mayor STD en emb
            # #     rst_words_names = np.array(rst_words_names)[[i in wrd_embs_std_pct for i in rst_words_sorted]]
            # elif filter_type == 4:  # Mayor abs(Mean)+ abs(STD) att
            #     # Palabras que poseen un valor de abs(Mean)+ abs(STD) mayor al percentil 60
            #     wrds_high_mean_std = np.where(att_mean_std > np.percentile(att_mean_std, 60))[0]
            #     rst_words_names = np.array(rst_words_names)[[i in wrds_high_mean_std for i in rst_words_sorted]]

            # Mostrar solo X elementos
            # closest = rst_words_names[:words_shown]
            closest = dict(zip(rst_words_names[:words_shown], att_rst_words[np.argsort(att_rst_words)[::-1]][:words_shown]))
            # print(list(zip(rst_words_names,np.sort(att_rst_words)[::-1]))[:words_shown])

            # most_relevant_words.extend(closest)

            # farthest = rst_words_names[-show_items:]
            # less_relevant_words.extend(farthest)

            # closest_ids = list(map(lambda x: self.DATASET.DATA["TEXT_TOKENIZER"].word_index[x], closest))
            # closest_in_reviews = all([i in rst_seqs for i in closest_ids])
            # not_in_reviews = list(np.array(closest)[[i not in rst_seqs for i in closest_ids]])
            # not_in_reviews_words.extend(not_in_reviews)

            ret[item_idx] = {"name": str(itm_names[item_idx]), "words": closest}

        # not_in_reviews_words = pd.DataFrame(np.unique(not_in_reviews_words, return_counts=True)).T.sort_values(1, ascending=False).reset_index(drop=True)
        # most_relevant_words = pd.DataFrame(np.unique(most_relevant_words, return_counts=True)).T.sort_values(1, ascending=False).reset_index(drop=True)

        # print(f"\x1b[95m{not_in_reviews_words.head(show_items)}\x1b[0m")
        # print(most_relevant_words.head(show_items*2))

        return ret

    def evaluate_text(self, text):

        '''
        Retorna la predición y explicación para un texto dado
        :param str text: Texto
        '''

        print(f"\033[92m[QUERY] '{text}'\033[0m")

        n_rsts = 4

        # Preprocesar y limpiar el texto de la consulta
        text_prepro = self.DATASET.prerpocess_text(text)

        # Esta opción solo incluye las palabras frecuentes y no siempre coincide con la longitud de la frase
        lstm_text = self.DATASET.DATA["TEXT_TOKENIZER"].texts_to_sequences([text_prepro]) 
        # Esta opción incluye todas las palabras
        lstm_text_complete = [list(map(lambda x: self.DATASET.DATA["TEXT_TOKENIZER"].word_index[x], text_prepro.split(" ")))] 

        # Not included
        not_included_words = list(set(lstm_text_complete[0]) - set(lstm_text[0]))
        if len(not_included_words) > 0:
            not_included_words = list(map(lambda x: self.DATASET.DATA["TEXT_TOKENIZER"].index_word[x], not_included_words))
            print_g(f"No se incluyen las palabras (poco frecuentes): {not_included_words}")

        # Añadir el padding para obtener una predicción de items
        lstm_text_pad = tf.keras.preprocessing.sequence.pad_sequences(lstm_text, maxlen=self.DATASET.DATA["MAX_LEN_PADDING"])
        print(f"{'[PREPR]':15s} [{text_prepro}]")
        print(f"{'[TXT2ID]':15s} {lstm_text[0]}")
        print(f'{"[WORD FREQ]":15s} {list(map(lambda x: self.DATASET.DATA["TEXT_TOKENIZER"].word_counts[x] , text_prepro.split(" ")))}')

        # Obtenemos TODA la matriz de attention (no solo la del texto)
        all_att, itm_names, word_names, _, _ = self.get_item_word_att()

        # Separamos la de la consulta para guardarla en un excel
        att_query = all_att[lstm_text[0]]
        att_query_df = pd.DataFrame(att_query, columns=itm_names)
        att_query_df.insert(0, "text", np.array(word_names)[lstm_text[0]])
        att_query_df.transpose().to_excel(self.MODEL_PATH + "att_text.xlsx")

        # Distribución KDE de cada una de las palabras
        cm = 1 / 2.54  # centimeters in inches
        plt.figure(figsize=(20 * cm, 8 * cm))  # Tamaño del plot
        for wid, word in enumerate(att_query_df["text"].values):
            hp = sns.kdeplot(att_query[wid, :], fill=True, label=word)
        hp.set_title(text)
        hp.set_xlim([all_att.min(), all_att.max()])
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{self.MODEL_PATH}dist_text.pdf")
        plt.close()

        # Obtener la abs(mean)+ str para cada palabra de la query y de todas las palabras
        att_query_df["mean"] = att_query.mean(1)
        att_query_df["std"] = att_query.std(1)
        att_query_df["mean_std"] = np.abs(att_query_df["mean"]) + att_query_df["std"]
        all_att_mean = all_att.mean(1)
        all_att_std = all_att.std(1)
        all_att_mean_std = np.abs(all_att_mean) + all_att_std
        
        print(f"El rango de valores de la matriz de atención para todas las palabras se mueve en el rango [{all_att.min()},{all_att.max()}].")
        # Determinar que palabras son relevantes en función de su "mean_std"
        '''
        global_filter = True
        if not global_filter: pct = np.percentile(att_query_df["mean_std"], 60)  # Utilizando solo palabras query
        else: pct = np.percentile(all_att_mean_std, 10)  # Utilizando todas las palabras
        relevant_query_words = att_query_df[att_query_df["mean_std"] > pct]["text"]
        '''
        # Determinar que palabras son relevantes en función del porcentaje de valores que tenga en región [-.3, .3]
        
        n_bins = 3 + 1
        bins = [-1,-0.15,0.15,1] if self.CONFIG["model"]["model_version"]=="2" else np.linspace(-1, 1, n_bins)
        histograms = np.apply_along_axis(lambda x: np.histogram(x, bins=bins)[0], 1, all_att)
        histograms = histograms/all_att.shape[1]
        att_query_df.insert(1, "pct_0", np.array(histograms[:, n_bins//2-1])[lstm_text[0]])
        relevant_query_words = att_query_df[(att_query_df["pct_0"] < .95)]["text"]
        """
        # Usar el rango?
        att_query_df.insert(1, "range", np.ptp(att_query, -1))
        all_ranges = np.ptp(all_att, -1)
        threshold =  np.mean(all_ranges) - 2 * np.std(all_ranges)  # np.percentile(all_ranges, 10)
        relevant_query_words = att_query_df[(att_query_df["range"] > threshold)]["text"]
        print(list(zip(att_query_df["text"], att_query_df["range"])), threshold )
        """

        print_e("Reparar y unificar la parte de selección de palabras relevantes")

        # Obtener, para la review al completo, el top "n_rsts" de items predichos por el modelo
        preds_rst = self.MODEL.predict([lstm_text_pad, np.arange(self.DATASET.DATA["N_ITEMS"])[None, :]], verbose=0)
        preds_rst_arg = np.argsort(-preds_rst.flatten())[:n_rsts]
        preds_rst_vls = np.sort(-preds_rst.flatten())[:n_rsts]

        # Obtener los nombres de items y palabras relevantes
        item_relevant_words = self.item_relevant_words(preds_rst_arg)
        longest_rest_name = max(list(map(lambda x: len(x["name"]), item_relevant_words.values())))

        for i, itm_idx in enumerate(preds_rst_arg):
            itm_data = item_relevant_words[itm_idx]
            print(f"\t[{-preds_rst_vls[i]:0.2f}] {itm_data['name']:{longest_rest_name}s} {itm_data['words']}")
            itm_query_att = att_query_df.iloc[:, itm_idx + 2]
            assert itm_query_att.name == itm_data['name']
            query_item_relevance = list(zip(relevant_query_words.values, itm_query_att[relevant_query_words.index]))
            for qit, qtir in query_item_relevance:
                print(f"\t\t{qit} {qtir:0.2f}")
        
        # Retornamos varios elementos por si se quiere hacer algún tipo de preprocesado
        return att_query, att_query_df, all_att

    def get_item_word_att(self, items=[], words=[]):

        '''
        Retorna la matriz ATT de los items con las palabras dadas (todos por defecto)
        :param list items: Lista de ids de los ítems
        :param list words: Lista de ids de las palabras
        :returns:
            La matriz de "Attention"
            Lista de nombres de items
            Lista de nombres de palabras
            Embeddings de items
            Embeddings de palabras
        '''

        # Todo si las listas están vacías
        if len(items) == 0: items = list(range(self.DATASET.DATA["N_ITEMS"]))
        if len(words) == 0: words = list(range(self.DATASET.DATA["VOCAB_SIZE"]))

        # Obtener submodelos que generar los embeddigs de palabras e items
        itm_embs = tf.keras.models.Model(inputs=[self.MODEL.input[1]], outputs=[self.MODEL.get_layer("rest_emb").output])
        wrd_embs = tf.keras.models.Model(inputs=[self.MODEL.input[0]], outputs=[self.MODEL.get_layer("word_emb").output])

        # Obtener la función de activación que se aplica a la attention
        act_function = self.MODEL.get_layer("dotprod").activation._tf_decorator._decorated_target._keras_api_names[0].split(".")[-1]

        # Obtener embeddings y nombres de los items "ids"
        itm_embs = itm_embs.predict([items], verbose=0).squeeze()
        itm_names = self.DATASET.DATA["TRAIN_DEV"][["id_item", "name"]].sort_values("id_item").drop_duplicates().set_index("id_item").loc[items].name.values.tolist()

        # Obtener todas las palabras (nombres y embeddings)
        word_names = np.array(["UNK"] + list(self.DATASET.DATA["WORD_INDEX"].keys()))
        word_names = pd.DataFrame(word_names, columns=["name"]).loc[words].name.tolist()
        wrd_embs = wrd_embs.predict(np.expand_dims(words,-1), verbose=0).squeeze()
        # wrd_embs_std = np.std(wrd_embs, -1)
        # wrd_embs_std_pct = np.argsort(-wrd_embs_std)[:int(len(wrd_embs_std)*.10)]

        # Obtener compatibilidad de todas las palabras con todos los items
        all_att = np.dot(wrd_embs, itm_embs.T)
        if act_function == "tanh": all_att = np.tanh(all_att)
        elif act_function == "sigmoid": all_att = tf.math.sigmoid(all_att).numpy()
        elif act_function == "linear": all_att = all_att
        else: raise ValueError
        # att_std = np.std(all_att, -1)
        # att_mean = np.mean(all_att, -1)
        # att_mean_std = np.abs(np.mean(all_att, -1))+np.std(all_att, -1)
        # all_std_pct = np.argsort(-att_std)[:int(len(att_std)*.10)]

        # Esto también se puede hacer obteniendola directamente del modelo
        # att_md = tf.keras.models.Model(inputs=[self.MODEL.input], outputs=[self.MODEL.get_layer("dotprod").output])
        # att = att_md.predict([lstm_text_pad, np.arange(self.DATASET.DATA["N_ITEMS"])[None, :]], verbose=0).squeeze(axis=0)
        # print(att.T[0, 0])  # Ver si la máscara funciona

        return all_att, itm_names, word_names, itm_embs, wrd_embs

    def explain_test_sample(self, test_real_sample):
        # Obtenemos TODA la matriz de attention (no solo la del texto)
        all_att, itm_names, word_names, _, _ = self.get_item_word_att()

        # Obtener el texto de la fila dataframe y codificar en formato adecuado (al ser un caso real, ya se ha codificado).
        seq_encoding = self.DATASET.DATA["TEXT_SEQUENCES"][test_real_sample["seq"]]
        seq_encoding_nopad = seq_encoding[seq_encoding > 0]

        # Datos del rastaurante real
        restaurant_name = test_real_sample["name"]

        # De la matriz de attention, nos quedamos con las palabras del texto y el restaurante
        att_query = all_att[seq_encoding_nopad]
        att_query_df = pd.DataFrame(att_query, columns=itm_names)
        att_query_df.insert(0, "text", np.array(word_names)[seq_encoding_nopad])

        # Obtener la abs(mean)+ str para cada palabra de la query y de todas las palabras
        # att_query_df["mean_std"] = np.abs(att_query.mean(1)) + att_query.std(1)
        att_query_df["std"] = att_query.std(1)
        att_query_df["skew"] = skew(att_query, 1)

        # all_att_mean_std = np.abs(all_att.mean(1)) + all_att.std(1)
        all_att_std = all_att.std(1)
        all_att_skew = skew(all_att, 1)

        # Obtener cual es el percentil 10 de todas las palabras que existen y quedarse con mayores de ese valor
        # pct = np.percentile(all_att_mean_std, 10)
        # relevant_query_words = att_query_df[att_query_df["mean_std"] > pct][["text", restaurant_name]]

        pct_std = np.percentile(all_att_std, 10)
        pct_skew = np.percentile(all_att_skew, 80)
        
        relevant_query_words = att_query_df[(att_query_df["std"] >= pct_std) | (att_query_df["skew"] >= pct_skew)][["text", restaurant_name]]

        # Eliminando aquellas palabras cuya totalidad de valores se encuentren en la zona central de la distribución (3 rangos, entre -0.333 y 0.333 )
        """
        n_bins = 3 + 1
        bins = [-1, -0.4, 0.4, 1]  # np.linspace(-1, 1, n_bins)
        histograms = np.apply_along_axis(lambda x: np.histogram(x, bins=bins)[0], 1, all_att)
        histograms = histograms/all_att.shape[1]
        att_query_df["pct_0"] = np.array(histograms[:, n_bins//2-1])[seq_encoding_nopad]
        relevant_query_words = att_query_df[(att_query_df["pct_0"] < .95)][["text", restaurant_name]]
        """
        
        print({it[0]: it[1] for it in relevant_query_words.values})

        return {"max": all_att.max(), "min": all_att.min(), "values": {it[0]: it[1] for it in relevant_query_words.values}}