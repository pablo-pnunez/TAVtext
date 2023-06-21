# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, TFAutoModel

from bokeh.plotting import ColumnDataSource, figure, output_file, save
from sklearn.manifold import TSNE

import tensorflow_ranking as tfr
import tensorflow as tf
import pandas as pd
import numpy as np
import os

import keras_nlp


from src.models.text_models.att.ATT2VAL import ATT2VAL

class SSATT2ITM(ATT2VAL):
    """ Predecir, a partir de los embeddings de las palabras de una review (BERT) y los de los items, el restaurante de la review """

    def __init__(self, config, dataset):
        ATT2VAL.__init__(self, config=config, dataset=dataset)

    def get_sub_model(self):

        mv = self.CONFIG["model"]["model_version"]
        self.MODEL_VERSION = mv

        rst_no = self.DATASET.DATA["N_ITEMS"]
        pad_len = self.DATASET.DATA["MAX_LEN_PADDING"]
        vocab_size = self.DATASET.DATA["VOCAB_SIZE"]

        model_url  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
        self.encoding_model = TFAutoModel.from_pretrained(model_url)
        self.tokenizer = AutoTokenizer.from_pretrained(model_url)

        self.encoding_model.trainable = True

        text_in = tf.keras.Input(shape=(pad_len), dtype='int32')
        rest_in = tf.keras.Input(shape=(rst_no), dtype='int32')
        
        model = None
        if mv == "0":  # Modelo básico sin heads, solo una capa oculta

            # init = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
            use_bias = True
            dropout = .3
            
            # word_importance = tf.keras.layers.Embedding(vocab_size, 1, name="word_importance", embeddings_initializer="ones", mask_zero=True)(text_in)

            mask_query = tf.cast(tf.math.not_equal(text_in, 0), tf.float32) # Se obtiene la máscara del texto para un solo item
            mask_query = tf.tile(tf.expand_dims(mask_query, axis=-1),[1,1,rst_no]) # Se repite para todos los items

            ht_emb = self.encoding_model(text_in)[0]
            emb_size = ht_emb.shape[-1]
            
            use_bert = False

            if not use_bert:            
                words_emb = tf.keras.layers.Embedding(self.tokenizer.vocab_size, emb_size, name="all_words")
                ht_emb = words_emb(text_in)

            
            """
            wr_att_embs = tf.keras.layers.Embedding(vocab_size, 2, mask_zero=True, name="wr_att_embs")
            wr_att_embs = wr_att_embs(text_in)
            # wr_att_embs = tf.keras.layers.Dropout(dropout)(wr_att_embs)     
            wr_attention_scores = tf.matmul(wr_att_embs, wr_att_embs, transpose_b=True)
            ht_emb = tf.einsum("abc,acd->abd", wr_attention_scores, ht_emb)
            """
            
            ht_emb = tf.keras.layers.Lambda(lambda x: x, name="word_emb")(ht_emb)
            #ht_emb = tf.keras.layers.Dropout(dropout)(ht_emb)
            
            #position_embeddings = keras_nlp.layers.SinePositionEncoding()(ht_emb)
            #ht_emb = ht_emb + position_embeddings


            rests_emb = tf.keras.layers.Embedding(rst_no, emb_size, name=f"in_rsts")
            hr_emb = rests_emb(rest_in)
            # hr_emb = tf.keras.layers.Activation("tanh")(hr_emb)
            # hr_emb = tf.keras.layers.Dense(emb_size * 2, use_bias=use_bias)(hr_emb)
            # hr_emb = tf.keras.layers.Activation("tanh")(hr_emb)
            # hr_emb = tf.keras.layers.Dense(emb_size, use_bias=use_bias)(hr_emb)
            # hr_emb = tf.keras.layers.Activation("tanh")(hr_emb)
            # hr_emb = tf.keras.layers.Dropout(dropout)(hr_emb)

            hr_emb = tf.keras.layers.Lambda(lambda x: x, name="rest_emb")(hr_emb)

            model = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([ht_emb, hr_emb])
            # model = tf.keras.layers.BatchNormalization()(model)
            # model = tf.keras.layers.GaussianNoise(.5)(model)
            
            model = tf.keras.layers.Lambda(lambda x: x[0]*x[1])([model, mask_query])
            model = tf.keras.layers.Activation("tanh", name="dotprod")(model)
            # model = tf.keras.layers.Lambda(lambda x: x[0]*x[1], name="importance")([model, word_importance])
            # model = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x[0], 2), name="dotprod")([model])

            model = tf.keras.layers.Dropout(dropout)(model)

            model = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1), name="sum")(model)
            
            # model = tf.keras.layers.Dropout(.1)(model)

            model_out = tf.keras.layers.Activation("sigmoid", name="out", dtype='float32')(model)

            model = tf.keras.models.Model(inputs=[text_in, rest_in], outputs=[model_out], name=f"{self.MODEL_NAME}_{self.MODEL_VERSION}")

            optimizer = tf.keras.optimizers.legacy.Adam(self.CONFIG["model"]["learning_rate"])

            metrics = [tfr.keras.metrics.NDCGMetric(topn=10, name="NDCG@10"), tfr.keras.metrics.RecallMetric(topn=1, name='r1'), tfr.keras.metrics.RecallMetric(topn=5, name='r5'), tfr.keras.metrics.RecallMetric(topn=10, name='r10'),
                       tfr.keras.metrics.PrecisionMetric(topn=5, name='p5'), tfr.keras.metrics.PrecisionMetric(topn=10, name='p10')]

            loss = tf.keras.losses.CategoricalCrossentropy()  # Prueba

        model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

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

        data_y = tf.data.Dataset.from_tensor_slices(rst_data)
        data_y = data_y.map(lambda x: tf.one_hot(x, self.DATASET.DATA["N_ITEMS"]), num_parallel_calls=tf.data.AUTOTUNE)
        
        # return tf.data.Dataset.zip((data_x, data_y)).take(int(len(dataframe)*0.1)).shuffle(1000)
        return tf.data.Dataset.zip((data_x, data_y))
    
    def emb_tsne(self):

        wrd_embs = tf.keras.models.Model(inputs=[self.MODEL.input[0]], outputs=[self.MODEL.get_layer("word_emb").output])
        rst_embs = tf.keras.models.Model(inputs=[self.MODEL.input[1]], outputs=[self.MODEL.get_layer("rest_emb").output])

        rst_embs = rst_embs.predict([list(range(self.DATASET.DATA["N_ITEMS"]))], verbose=0).squeeze()
        rest_names = self.DATASET.DATA["TRAIN_DEV"][["id_item", "name"]].sort_values("id_item").drop_duplicates().name.values.tolist()
        
        vocab = pd.DataFrame(self.tokenizer.vocab.items(), columns=["token", "id"]).sort_values("id").reset_index(drop=True).set_index("id")
        word_names = vocab.token.values
        wrd_embs = wrd_embs.predict(vocab.index.values, verbose=0).numpy()
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