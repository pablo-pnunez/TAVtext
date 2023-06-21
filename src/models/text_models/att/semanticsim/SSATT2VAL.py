# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, TFAutoModel

from bokeh.plotting import ColumnDataSource, figure, output_file, save
from sklearn.manifold import TSNE

import tensorflow_ranking as tfr
import tensorflow as tf
import pandas as pd
import numpy as np
import os

from src.Common import print_g, print_e, get_pickle, to_pickle
from src.models.text_models.att.ATT2VAL import ATT2VAL

class SSATT2VAL(ATT2VAL):
    """ Predecir, a partir de los embeddings de las palabras de una review (BERT) y los de los items, el restaurante de la review """

    def __init__(self, config, dataset):
        ATT2VAL.__init__(self, config=config, dataset=dataset)

    def get_sub_model(self):

        mv = self.CONFIG["model"]["model_version"]
        self.MODEL_VERSION = mv

        rst_no = self.DATASET.DATA["N_ITEMS"]
        pad_len = self.DATASET.DATA["MAX_LEN_PADDING"]
        vocab_size = self.DATASET.DATA["VOCAB_SIZE"]

        # model_url  = "sentence-transformers/distiluse-base-multilingual-cased-v2"
        model_url  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
        self.encoding_model = TFAutoModel.from_pretrained(model_url)
        self.tokenizer = AutoTokenizer.from_pretrained(model_url)

        self.encoding_model.trainable = True

        text_in = tf.keras.Input(shape=(pad_len), dtype='int32')
        rest_in = tf.keras.Input(shape=(rst_no), dtype='int32')
        
        model = None

        if mv == "0": # Se añade la estandarización a la salida y se simplifica el modelo
            dropout = .1
           
            bert_emb = self.encoding_model(text_in)[0]
            emb_size = bert_emb.shape[-1]

            use_bert = False

            if use_bert:            
                ht_emb = tf.keras.layers.Lambda(lambda x: x, name="word_emb")(bert_emb)
            else:
                words_emb = tf.keras.layers.Embedding(self.tokenizer.vocab_size, emb_size, name="all_words")
                ht_emb = words_emb(text_in)
                ht_emb = tf.keras.layers.Lambda(lambda x: x, name="word_emb")(ht_emb)

            ht_emb = tf.keras.layers.Dropout(dropout)(ht_emb)

            mask_query = tf.cast(tf.math.not_equal(text_in, 0), tf.float32) # Se obtiene la máscara del texto para un solo item
            mask_query = tf.tile(tf.expand_dims(mask_query, axis=-1),[1,1,rst_no]) # Se repite para todos los items

            items_emb = tf.keras.layers.Embedding(rst_no, emb_size, name="all_items")
            hr_emb = items_emb(rest_in)

            # hr_emb = tf.keras.layers.Activation("tanh")(hr_emb)                               
            # hr_emb = tf.keras.layers.Dense(emb_size, activation="tanh")(hr_emb)

            hr_emb = tf.keras.layers.Lambda(lambda x: x, name="rest_emb")(hr_emb)
            hr_emb = tf.keras.layers.Dropout(dropout)(hr_emb)

            model = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True), name="dot_mul")([ht_emb, hr_emb])
            model = tf.keras.layers.Lambda(lambda x: x[0] * x[1] , name="dot_mask")([model, mask_query])
            model = tf.keras.layers.Activation("tanh", name="dotprod")(model)

            # model = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1), name="sum")(model)
            # model = tf.keras.layers.Activation(self.custom_activation_relutan)(model) # Sigmoide entre 1 y 5                                 
            model = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x[0], 1)/tf.math.reduce_sum(x[1], 1), name="sum")([model, mask_query])
            model = model * 5 # Esto obliga a que siempre exista uno igual a 5
            model_out = tf.keras.layers.Activation("linear", name="out", dtype='float32')(model)
                        
            model = tf.keras.models.Model(inputs=[text_in, rest_in], outputs=[model_out], name=f"{self.MODEL_NAME}_{self.MODEL_VERSION}")
            optimizer = tf.keras.optimizers.legacy.Adam(self.CONFIG["model"]["learning_rate"])

            metrics = [tfr.keras.metrics.NDCGMetric(topn=10, name="NDCG@10"), tf.keras.metrics.MeanAbsoluteError(name="MAE"),
                       tfr.keras.metrics.RecallMetric(topn=5, name='RC@5'), tfr.keras.metrics.RecallMetric(topn=10, name='RC@10')]
                     
        model.compile(loss=self.custom_loss, metrics=metrics, optimizer=optimizer)

        print(model.summary())

        return model

    def __create_tfdata__(self, dataframe):

        # Hay que obtener las valoraciones y quedarse con la última si hay varias
        all_data = pd.read_pickle(self.DATASET.DATASET_PATH+"ALL_DATA").sort_values("date")
        rating_data = all_data[["reviewId", "rating"]].drop_duplicates(keep='last', inplace=False)
        rating_data = rating_data.set_index("reviewId").loc[dataframe.reviewId.values]["rating"].values/10

        # Obtener el texto original para tokenizar
        # bertoken_path = "BERT_SEQUENCES"
        # if not os.path.exists(self.DATASET.DATASET_PATH + bertoken_path):
        text_data = all_data[["reviewId", "text"]]
        text_data = text_data.set_index("reviewId").loc[dataframe.reviewId.values]["text"].values
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        text_seqs = self.tokenizer.batch_encode_plus(text_data.tolist(), max_length=self.DATASET.DATA["MAX_LEN_PADDING"], truncation=True)
        text_seqs = text_seqs.data["input_ids"]
        seq_data = tf.keras.preprocessing.sequence.pad_sequences(text_seqs, maxlen=self.DATASET.DATA["MAX_LEN_PADDING"], padding='pre')
        # to_pickle(self.DATASET.DATASET_PATH, bertoken_path, seq_data)
        # else:
        #    seq_data = get_pickle(self.DATASET.DATASET_PATH, bertoken_path)
       
        rst_data = dataframe.id_item.values
        
        data_x1 = tf.data.Dataset.from_tensor_slices(seq_data)
        data_x2 = tf.data.Dataset.from_tensor_slices([range(self.DATASET.DATA["N_ITEMS"])]).repeat(len(dataframe))
        data_x = tf.data.Dataset.zip((data_x1, data_x2))

        # Hacer un onehot y multiplicar por el score cada vector -> todo ceros menos el score
        data_y = tf.one_hot(rst_data,self.DATASET.DATA["N_ITEMS"])*tf.cast(tf.reshape(rating_data,(len(dataframe),1)), "float")
        data_y = tf.data.Dataset.from_tensor_slices(data_y)

        return tf.data.Dataset.zip((data_x, data_y))
    
    def emb_tsne(self):

        wrd_embs = tf.keras.models.Model(inputs=[self.MODEL.input[0]], outputs=[self.MODEL.get_layer("word_emb").output])
        rst_embs = tf.keras.models.Model(inputs=[self.MODEL.input[1]], outputs=[self.MODEL.get_layer("rest_emb").output])

        rst_embs = rst_embs.predict([list(range(self.DATASET.DATA["N_ITEMS"]))], verbose=0).squeeze()
        rest_names = self.DATASET.DATA["TRAIN_DEV"][["id_item", "name"]].sort_values("id_item").drop_duplicates().name.values.tolist()
        
        vocab = self.tokenizer.get_vocab()
        word_names = np.array(["UNK"]+list(self.DATASET.DATA["WORD_INDEX"].keys()))
        wrd_embs = wrd_embs.predict(list(range(self.DATASET.DATA["VOCAB_SIZE"])), verbose=0).numpy()
        wrd_embs = np.concatenate(wrd_embs)

        if rst_embs.shape[1] > 2:
            tsne_r = TSNE(n_components=2, learning_rate="auto", init="pca", metric="cosine")
            tsne_w = TSNE(n_components=2, learning_rate="auto", init="pca", metric="cosine")
            rst_tsne = tsne_r.fit_transform(rst_embs)
            wrd_tsne = tsne_w.fit_transform(wrd_embs)
        else:
            rst_tsne = rst_embs
            wrd_tsne = wrd_embs

        source_w = ColumnDataSource(data=dict(x=wrd_tsne[:, 0], y=wrd_tsne[:, 1], desc=word_names))
        source_r = ColumnDataSource(data=dict(x=rst_tsne[:, 0], y=rst_tsne[:, 1], desc=rest_names))

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
        wrd_embs = wrd_embs.predict(words, verbose=0).numpy()
        wrd_embs = np.concatenate(wrd_embs)
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