# -*- coding: utf-8 -*-

import tensorflow_ranking as tfr
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pickle as pkl
import pandas as pd
import numpy as np
import binascii
import os

from src.Common import print_g, print_e, get_pickle, to_pickle
from src.models.text_models.ATT2ITM import ATT2ITM


class ATT2VAL(ATT2ITM):
    """ Predecir, a partir de los embeddings de una review y los de los items, el restaurante de la review """

    def __init__(self, config, dataset):
        ATT2ITM.__init__(self, config=config, dataset=dataset)

    def get_sub_model(self):

        mv = self.CONFIG["model"]["model_version"]
        self.MODEL_VERSION = mv

        rst_no = self.DATASET.DATA["N_ITEMS"]
        pad_len = self.DATASET.DATA["MAX_LEN_PADDING"]
        vocab_size = self.DATASET.DATA["VOCAB_SIZE"]

        def custom_loss(y_true, y_pred):
            mse = tf.square(y_true - y_pred)
            
            mul_factor = int(rst_no*0.3) ## un 10%

            true_max = tf.reduce_max(y_true, axis=1, keepdims=True)
            true_ones = (y_true/true_max) # multiplica por 1 los items no activos y por 2 el actual
            mse = mse * ((true_ones*(mul_factor-1))+1)
            
            mse = tf.sqrt(tf.reduce_mean(mse))
            return mse  # Devolver la pérdida calculada
        
        def custom_activation_sigmoid(x):
            min_value = 0.0
            max_value = 5.0
            return (max_value - min_value) * tf.sigmoid(x) + min_value

        def custom_activation_sigmoid_two(x):
            min_value = 0.0
            max_value = 10.0
            return (tf.sigmoid(x) - 0.5) * (max_value - min_value) + min_value

        def custom_activation_tanh(x):
            min_value = 0.0
            max_value = 5.0
            return (tf.nn.tanh(x) + 1) * ((max_value - min_value) / 2) + min_value

        def custom_activation_relutan(x):
            min_value = 0.0
            max_value = 5
            x = tf.cast(x, 'float32')  # Convertir a float32
            return tf.maximum(0.0, tf.nn.tanh(x)) * (max_value - min_value) + min_value

        def custom_activation_relu5(x):
            min_value = 0.0
            max_value = 5.0
            x = tf.cast(x, 'float32')  # Convertir a float32
            return tf.maximum(0.0, x) * max_value


        text_in = tf.keras.Input(shape=(pad_len), dtype='int32')
        rest_in = tf.keras.Input(shape=(rst_no), dtype='int32')
        
        model = None

        if mv == "0": # Se añade la estandarización a la salida y se simplifica el modelo
            emb_size = 64  # 128
            dropout = .2

            use_bias = True
            emb_regu = None # tf.keras.regularizers.L2()
             
            query_emb = tf.keras.layers.Embedding(vocab_size, emb_size , mask_zero=True, name="all_words", embeddings_regularizer=emb_regu)
            mask_query = tf.cast(tf.math.not_equal(text_in, 0), tf.float32) # Se obtiene la máscara del texto para un solo item
            mask_query = tf.tile(tf.expand_dims(mask_query, axis=-1),[1,1,rst_no]) # Se repite para todos los items

            ht_emb = query_emb(text_in)
            ht_emb = tf.keras.layers.Lambda(lambda x: x, name="word_emb")(ht_emb)
            ht_emb = tf.keras.layers.Dropout(dropout)(ht_emb)

            items_emb = tf.keras.layers.Embedding(rst_no, emb_size, name="all_items", embeddings_regularizer=emb_regu)
            hr_emb = items_emb(rest_in)
            hr_emb = tf.keras.layers.Lambda(lambda x: x, name="rest_emb")(hr_emb)
            hr_emb = tf.keras.layers.Dropout(dropout)(hr_emb)

            model = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True), name="dot_mul")([ht_emb, hr_emb])
            model = tf.keras.layers.Lambda(lambda x: x[0] * x[1] , name="dot_mask")([model, mask_query])
            model = tf.keras.layers.Activation("tanh", name="dotprod")(model)

            model = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1), name="sum")(model)
            model = tf.keras.layers.Activation(custom_activation_relutan)(model) # Sigmoide entre 1 y 5                                 
            model_out = tf.keras.layers.Activation("linear", name="out", dtype='float32')(model)
                        
            model = tf.keras.models.Model(inputs=[text_in, rest_in], outputs=[model_out], name=f"{self.MODEL_NAME}_{self.MODEL_VERSION}")
            optimizer = tf.keras.optimizers.legacy.Adam(self.CONFIG["model"]["learning_rate"])

            metrics = [tfr.keras.metrics.NDCGMetric(topn=10, name="NDCG@10"), tf.keras.metrics.MeanAbsoluteError(name="MAE"),
                       tfr.keras.metrics.RecallMetric(topn=5, name='RC@5'), tfr.keras.metrics.RecallMetric(topn=10, name='RC@10')]
                     
        model.compile(loss=custom_loss, metrics=metrics, optimizer=optimizer)

        print(model.summary())

        return model

    def __create_tfdata__(self, dataframe):

        # Hay que obtener las valoraciones y quedarse con la última si hay varias
        all_data = pd.read_pickle(self.DATASET.DATASET_PATH+"ALL_DATA").sort_values("date")
        rating_data = all_data[["reviewId", "rating"]].drop_duplicates(keep='last', inplace=False)
        rating_data = rating_data.set_index("reviewId").loc[dataframe.reviewId.values]["rating"].values/10

        seq_data = self.DATASET.DATA["TEXT_SEQUENCES"][dataframe.seq.values]
        rst_data = dataframe.id_item.values
        
        data_x1 = tf.data.Dataset.from_tensor_slices(seq_data)
        data_x2 = tf.data.Dataset.from_tensor_slices([range(self.DATASET.DATA["N_ITEMS"])]).repeat(len(dataframe))
        data_x = tf.data.Dataset.zip((data_x1, data_x2))

        # Hacer un onehot y multiplicar por el score cada vector -> todo ceros menos el score
        data_y = tf.one_hot(rst_data,self.DATASET.DATA["N_ITEMS"])*tf.cast(tf.reshape(rating_data,(len(dataframe),1)), "float")
        data_y = tf.data.Dataset.from_tensor_slices(data_y)

        return tf.data.Dataset.zip((data_x, data_y))
    
    def evaluate_text(self, text):

        '''
        Retorna la predición y explicación para un texto dado
        :param str text: Texto
        '''

        print(f"\033[92m[QUERY] '{text}'\033[0m")

        n_rsts = 6
        text_crc = binascii.crc32(text.encode('utf8'))
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
        att_query_df.transpose().to_excel(self.MODEL_PATH + f"att_text_{text_crc}.xlsx")

        # Distribución KDE de cada una de las palabras
        cm = 1 / 2.54  # centimeters in inches
        plt.figure(figsize=(20 * cm, 8 * cm))  # Tamaño del plot
        for wid, word in enumerate(att_query_df["text"].values):
            hp = sns.kdeplot(att_query[wid, :], fill=True, label=word)
        hp.set_title(text)
        hp.set_xlim([all_att.min(), all_att.max()])
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{self.MODEL_PATH}dist_text_{text_crc}.pdf")
        plt.close()

        # Obtener la abs(mean)+ str para cada palabra de la query y de todas las palabras
        att_query_df["mean"] = att_query.mean(1)
        att_query_df["std"] = att_query.std(1)
        att_query_df["mean_std"] = np.abs(att_query_df["mean"]) + att_query_df["std"]
        all_att_mean = all_att.mean(1)
        all_att_std = all_att.std(1)
        all_att_mean_std = np.abs(all_att_mean) + all_att_std

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