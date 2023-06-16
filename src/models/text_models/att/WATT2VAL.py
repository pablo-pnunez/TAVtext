# -*- coding: utf-8 -*-
from bokeh.plotting import ColumnDataSource, figure, output_file, save
from bokeh.models import LinearColorMapper
from sklearn.manifold import TSNE

import tensorflow_ranking as tfr
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pickle as pkl
import pandas as pd
import numpy as np

import keras_nlp

from src.Common import print_g, print_e
from src.models.text_models.att.ATT2VAL import ATT2VAL

def custom_loss(self, y_true, y_pred):
    mse = tf.square(y_true - y_pred)

    # rst_no = self.DATASET.DATA["N_ITEMS"]

    mul_factor = int(rst_no * 0.05) ## un 5%

    true_max = tf.reduce_max(y_true, axis=1, keepdims=True)
    true_ones = (y_true/true_max) # multiplica por 1 los items no activos y por 2 el actual
    mse = mse * ((true_ones*(mul_factor-1))+1)
    
    mse = tf.cast(tf.sqrt(tf.reduce_mean(mse)), tf.float32)
    return mse  # Devolver la pérdida calculada

class WATT2VAL(ATT2VAL):
    """ Predecir, a partir de los embeddings de una review y los de los items, el restaurante de la review """

    def __init__(self, config, dataset):
        ATT2VAL.__init__(self, config=config, dataset=dataset)
   
    def get_sub_model(self):

        mv = self.CONFIG["model"]["model_version"]
        self.MODEL_VERSION = mv

        rst_no = self.DATASET.DATA["N_ITEMS"]
        pad_len = self.DATASET.DATA["MAX_LEN_PADDING"]
        vocab_size = self.DATASET.DATA["VOCAB_SIZE"]

        text_in = tf.keras.Input(shape=(None,), dtype='int32')
        rest_in = tf.keras.Input(shape=(rst_no), dtype='int32')
        
        model = None

        if mv == "0": # Se añade la estandarización a la salida y se simplifica el modelo
            emb_size = 64  # 128
            dropout = .4
                 
            # PALABRAS            
            query_emb = tf.keras.layers.Embedding(vocab_size, emb_size , mask_zero=True, name="all_words")
            ht_emb = query_emb(text_in)      

            '''
            # Crear una capa de MultiHeadAttention
            # Máscara de la query
            att_mask = tf.cast(tf.math.not_equal(text_in, 0), tf.float32) # Se obtiene la máscara del texto para un solo item
            att_mask = tf.tile(tf.expand_dims(att_mask, axis=-1),[1,1,pad_len]) # Se repite para todos los items
            att_ly = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=8, name="att", dropout=dropout)
            ht_emb, attention = att_ly(ht_emb, ht_emb, return_attention_scores=True)
            '''
            
            wr_att_embs = tf.keras.layers.Embedding(vocab_size, emb_size, mask_zero=True, name="wr_att_embs")
            wr_att_embs = wr_att_embs(text_in)
            # wr_att_embs = tf.keras.layers.Dense(8, activation="tanh")(ht_emb)
            wr_att_embs = tf.keras.layers.Dropout(dropout)(wr_att_embs)     
            wr_attention_scores = tf.matmul(wr_att_embs, wr_att_embs, transpose_b=True)
            # wr_attention_scores = tf.matmul(ht_emb, ht_emb, transpose_b=True)
            # wr_attention_scores = tf.keras.layers.Activation("tanh")(wr_attention_scores)
            # Esto está copiado y adaptado de la attention. Se supone que cada embedding se obtiene como combinación lineal
            # de los scores por los embeddings anteriores
            ht_emb = tf.einsum("abc,acd->abd", wr_attention_scores, ht_emb)

            ## Aprender un peso para cada palabra que represente su importancia
            # word_relevance = tf.keras.layers.Embedding(vocab_size, 1, mask_zero=True, name="wr_rel_weight", embeddings_initializer="ones")
            # wr_rel = word_relevance(text_in)
            # wr_rel = tf.keras.layers.Activation("tanh")(wr_rel)
            # wr_rel = tf.tile(wr_rel, [1, 1, emb_size])
            # ht_emb = tf.keras.layers.Lambda(lambda x: tf.multiply(x[0], x[1]))([ht_emb, wr_rel])  # Multiplicar el embedding de la palabra por su peso
            ## El embedding ha de ser el resultado de haber multiplicado

            ht_emb = tf.keras.layers.Lambda(lambda x: x, name="word_emb")(ht_emb)
            ht_emb = tf.keras.layers.Dropout(dropout)(ht_emb)
            
            # position_embeddings = keras_nlp.layers.PositionEmbedding(sequence_length=pad_len)(ht_emb)
            position_embeddings = keras_nlp.layers.SinePositionEncoding()(ht_emb)
            ht_emb = ht_emb + position_embeddings

            # ITEMS
            items_emb = tf.keras.layers.Embedding(rst_no, emb_size, name="all_items")
            hr_emb = items_emb(rest_in)                      
            hr_emb = tf.keras.layers.Lambda(lambda x: x, name="rest_emb")(hr_emb)
            hr_emb = tf.keras.layers.Dropout(dropout)(hr_emb)

            # ATT MATRIX
            model = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True), name="dot_mul")([ht_emb, hr_emb])
            mask_query = tf.cast(tf.math.not_equal(text_in, 0), tf.float16) # Se obtiene la máscara del texto para un solo item
            mask_query = tf.tile(tf.expand_dims(mask_query, axis=-1),[1,1,rst_no]) # Se repite para todos los items
            model = tf.keras.layers.Activation("sigmoid", name="dotprod")(model)
            
            """
            word_relevance = tf.keras.layers.Embedding(vocab_size, 1, mask_zero=True, name="wr_rel_weight", embeddings_initializer="ones")
            wr_rel = word_relevance(text_in)
            wr_rel = tf.tile(wr_rel,[1,1, rst_no]) # Se repite para todos los items
            model = tf.multiply(model, wr_rel)
            """

            model = tf.keras.layers.Lambda(lambda x: x[0] * x[1] , name="dot_mask")([model, mask_query])
            # model = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1), name="sum")(model)
            # model = tf.keras.layers.ReLU(max_value=5)(model)                                    

            model = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x[0], 1)/tf.math.reduce_sum(x[1], 1), name="sum")([model, mask_query])
            # model = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x[0], 1), name="sum")([model, mask_query])

            model = model * 5 # Esto obliga a que siempre exista uno igual a 5
            # model = tf.nn.l2_normalize(model, axis=1) * 5 # Esto obliga a que siempre exista uno igual a 5

            # model = tf.keras.layers.Activation(self.custom_activation_smoothstep)(model)
            # model = tf.keras.layers.Lambda(lambda x: x * 5.0 )(model)
            # model = tf.keras.layers.ReLU(max_value=5)(model)


            model_out = tf.keras.layers.Activation("linear", name="out", dtype=tf.float32)(model)
                        
            model = tf.keras.models.Model(inputs=[text_in, rest_in], outputs=[model_out], name=f"{self.MODEL_NAME}_{self.MODEL_VERSION}")
            optimizer = tf.keras.optimizers.legacy.Adam(self.CONFIG["model"]["learning_rate"])

            metrics = [tfr.keras.metrics.NDCGMetric(topn=10, name="NDCG@10"), tf.keras.metrics.MeanAbsoluteError(name="MAE"),
                       tfr.keras.metrics.RecallMetric(topn=5, name='RC@5'), tfr.keras.metrics.RecallMetric(topn=10, name='RC@10')]
                     
        model.compile(loss=self.custom_loss, metrics=metrics, optimizer=optimizer)

        print(model.summary())

        return model

    def emb_tsne(self):

        wrd_embs = tf.keras.models.Model(inputs=[self.MODEL.input[0]], outputs=[self.MODEL.get_layer("word_emb").output])
        rst_embs = tf.keras.models.Model(inputs=[self.MODEL.input[1]], outputs=[self.MODEL.get_layer("rest_emb").output])

        rst_embs = rst_embs.predict([list(range(self.DATASET.DATA["N_ITEMS"]))], verbose=0).squeeze()
        rest_names = self.DATASET.DATA["TRAIN_DEV"][["id_item", "name"]].sort_values("id_item").drop_duplicates().name.values.tolist()
        
        word_names = np.array(["UNK"]+list(self.DATASET.DATA["WORD_INDEX"].keys()))
        wrd_embs = wrd_embs.predict(np.expand_dims(list(range(self.DATASET.DATA["VOCAB_SIZE"])),-1), verbose=0).squeeze()
      
        if wrd_embs.shape[1] > 2:
            tsne_r = TSNE(n_components=2, learning_rate="auto", metric="cosine")
            tsne_w = TSNE(n_components=2, learning_rate="auto", metric="cosine")
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

    def get_item_word_att(self, items=[], words=[]):

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
