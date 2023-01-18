# -*- coding: utf-8 -*-
from src.datasets.text_datasets.amazon.AmazonDataset import *
from src.Common import to_pickle, print_g

import os
import numpy as np
import pandas as pd
import tensorflow as tf


class AmazonCategoryDataset(AmazonDataset):

    def __init__(self, config, load=None):
        AmazonDataset.__init__(self, config=config, load=load)

    def get_data(self, load=["TEXT_SEQUENCES", "TRAIN_DEV", "TEST", "WORD_INDEX", "VOCAB_SIZE", "MAX_LEN_PADDING", "TEXT_TOKENIZER", "N_ITEMS"]):

        # Cargar los datos
        dict_data = self.get_dict_data(self.DATASET_PATH, load)

        # Si ya existen, retornar
        if dict_data is not False and len(dict_data) == len(load):
            return dict_data
        # Si no existe, crear
        else:

            # Cargar las reviews
            all_data = self.load_category(self.CONFIG["category"])

            # Items con X o más reseñas
            r_mth_x = all_data.groupby("itemId").apply(lambda x: 0 if len(x) < self.CONFIG["min_reviews_rst"] else 1).reset_index()
            r_mth_x = r_mth_x.loc[r_mth_x[0] == 1]["itemId"].values
            print_g("%d items with %d or more reviews." % (len(r_mth_x), self.CONFIG["min_reviews_rst"]))
            all_data = all_data.loc[all_data["itemId"].isin(r_mth_x)]

            # Usuarios con X o más reseñas
            u_mth_x = all_data.groupby("userId").apply(lambda x: 0 if len(x) < self.CONFIG["min_reviews_usr"] else 1).reset_index()
            u_mth_x = u_mth_x.loc[u_mth_x[0] == 1]["userId"].values
            all_data = all_data.loc[all_data["userId"].isin(u_mth_x)]

            # Obtener los datos del conjunto tras el filtrado (para el paper)
            print("· Nº de ejemplos resultantes: %d" % len(all_data))
            print("· Nº de items: %d" % len(all_data.itemId.unique()))
            print("· Nº de usuarios: %d" % len(all_data.userId.unique()))
            print("· Nº items medio por usuario: %f" % all_data.groupby("userId").apply(lambda x: len(x.itemId.unique())).mean())
            
            data_raw_path = "data/raw/amazon/"
            os.makedirs(data_raw_path, exist_ok=True)
            all_data[["reviewId", "userId", "itemId", "rating", "text"]].to_pickle(f'{data_raw_path}{self.CONFIG["category"]}.pkl')

            # Crear id de items (para el ONE-HOT)
            rst_newid = pd.DataFrame(zip(r_mth_x, range(len(r_mth_x))), columns=["itemId", "id_item"])
            all_data = all_data.merge(rst_newid, on="itemId")
            all_data = all_data.drop(columns=["itemId"])

            # Mezclar las reviews
            all_data = all_data.sample(frac=1, random_state=self.CONFIG["seed"]).reset_index(drop=True)

            # Tokenizar las palabras (Asociar cada palabra a un índice [WORD_INDEX])
            if self.CONFIG["n_max_words"] == 0:
                tokenizer_txt = tf.keras.preprocessing.text.Tokenizer()
            elif self.CONFIG["n_max_words"] > 0:
                n_max_words = self.CONFIG["n_max_words"]
                tokenizer_txt = tf.keras.preprocessing.text.Tokenizer(num_words=n_max_words)
            else:
                # Se busca el número máximo de palabras en función de la frecuencia de cada una
                tokenizer_tmp = tf.keras.preprocessing.text.Tokenizer()
                tokenizer_tmp.fit_on_texts(all_data[self.CONFIG["text_column"]])
                token_freq = pd.DataFrame(zip(tokenizer_tmp.word_counts.keys(), tokenizer_tmp.word_counts.values()), columns=["token", "freq"]).sort_values("freq", ascending=False).reset_index(drop=True)
                token_freq = token_freq[token_freq.freq >= abs(self.CONFIG["n_max_words"])]
                n_max_words = len(token_freq)
                tokenizer_txt = tf.keras.preprocessing.text.Tokenizer(num_words=n_max_words)

            tokenizer_txt.fit_on_texts(all_data[self.CONFIG["text_column"]])
            word_index = tokenizer_txt.word_index

            # Si se utilizan todas
            if self.CONFIG["n_max_words"] == 0:
                print_g("Intervienen %d palabras y utilizamos todas." % len(word_index))
            # Quedarse con las "n_max_words" palabras más frecuentes
            else:
                word_counts = tokenizer_txt.word_counts
                word_counts = {k: v for k, v in sorted(word_counts.items(), key=lambda item: item[1], reverse=True)}
                mas_frecuentes = list(word_counts.keys())[0:n_max_words]
                print_g("Intervienen %d palabras, pero nos quedamos con las %d más frecuentes." % (len(word_index), n_max_words))
                word_index = {x: word_index[x] for x in mas_frecuentes}

            # Transformar frases a secuencias de números según [WORD_INDEX] y añadir al set
            text_sequences = tokenizer_txt.texts_to_sequences(all_data[self.CONFIG["text_column"]])
            all_data["seq"] = text_sequences

            # Train Test split asegurandose de que en Train están todos los items.
            def data_split(rst_rvws):
                for_dev_tst = int(len(rst_rvws) * (self.CONFIG["test_dev_split"]*2))//2
                if for_dev_tst > 0:
                    rst_rvws.iloc[-for_dev_tst:, rst_rvws.columns.get_loc("test")] = 1  # Últimas x para test
                    rst_rvws.iloc[-for_dev_tst*2:-for_dev_tst, rst_rvws.columns.get_loc("dev")] = 1  # Penúltimas x para dev
                return rst_rvws

            all_data["dev"] = 0
            all_data["test"] = 0
            all_data = all_data.groupby("id_item").apply(data_split)

            # Truncar el padding?
            max_len_padding = None
            if self.CONFIG["truncate_padding"]:
                seq_lens = all_data.loc[(all_data["dev"] == 0) & (all_data["test"] == 0)]["seq"].apply(lambda x: len(x)).values
                max_len_padding = int(seq_lens.mean() + seq_lens.std() * 2)

            # Añadir al set con el padding
            seq_w_pad = tf.keras.preprocessing.sequence.pad_sequences(all_data["seq"].values, maxlen=max_len_padding).astype(np.int32)
            
            # Añadir un id para poder almacenar las secuecias en un numpy
            all_data["seq"] = range(len(seq_w_pad))
            # all_data["seq"] = seq_w_pad.tolist()
            max_len_padding = seq_w_pad.shape[1]

            # Hacer castings para ahorrar espacio
            all_data = all_data.astype({'dev': 'int8', 'test': 'int8', 'rating': 'int8', 'seq': 'int32', 'id_item': 'int32', })
            all_data = all_data.astype({'n_words_text': 'int32', 'n_words_title': 'int32', 'n_char_text': 'int32'})

            # El texto en si no es necesario, se pone en un pandas aparte para no sobrecargar la RAM
            model_data = all_data.drop(columns=["date", "text", "title"])

            # Separar los conjuntos finales
            train_dev = model_data.loc[model_data["test"] == 0].drop(columns=["test"])
            test = model_data.loc[model_data["test"] == 1].drop(columns=["dev", "test"])

            # Almacenar pickles
            to_pickle(self.DATASET_PATH, "N_ITEMS", len(all_data["id_item"].unique()))

            to_pickle(self.DATASET_PATH, "WORD_INDEX", word_index)
            to_pickle(self.DATASET_PATH, "VOCAB_SIZE", len(word_index) + 1)
            to_pickle(self.DATASET_PATH, "MAX_LEN_PADDING", max_len_padding)
            to_pickle(self.DATASET_PATH, "TEXT_TOKENIZER", tokenizer_txt)
            to_pickle(self.DATASET_PATH, "TEXT_SEQUENCES", seq_w_pad)

            to_pickle(self.DATASET_PATH, "ALL_DATA", all_data)
            to_pickle(self.DATASET_PATH, "TRAIN_DEV", train_dev)
            to_pickle(self.DATASET_PATH, "TEST", test)

            return self.get_dict_data(self.DATASET_PATH, load)

    def get_data_stats(self):
        ALL = self.DATA["TRAIN_DEV"].append(self.DATA["TEST"])

        n_revs = len(ALL["reviewId"].unique())
        n_rests = len(ALL["id_item"].unique())
        
        revs_per_res = ALL.groupby("id_item").apply(lambda x: len(x.reviewId.unique()))
        avg_revs_rest = revs_per_res.mean()
        pctg_total_revs_rest_pop = revs_per_res.sort_values().iloc[-1]/n_revs

        avg_rating = ALL.rating.mean()/10
        std_rating = ALL.rating.std()/10

        print("\n".join(map(str, [self.CONFIG["category"],n_revs, n_rests, avg_revs_rest, pctg_total_revs_rest_pop, avg_rating, std_rating])))
