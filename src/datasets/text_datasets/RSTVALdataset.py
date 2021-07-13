# -*- coding: utf-8 -*-

from src.datasets.text_datasets.TextDataset import *
from src.Common import to_pickle, print_g, print_e

import os
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer


class RSTVALdataset(TextDataset):

    def __init__(self, config):
        TextDataset.__init__(self, config=config)

    def get_data(self, load=["TRAIN_DEV", "TEST",  "WORD_INDEX", "VOCAB_SIZE", "MAX_LEN_PADDING", "TEXT_TOKENIZER", "VECTORIZER", "FEATURES_NAME", "N_RST", "STEMMING_DICT"]):

        # Cargar los datos
        dict_data = self.get_dict_data(self.DATASET_PATH, load)

        # Si ya existen, retornar
        if dict_data:
            return dict_data
        # Si no existe, crear
        else:

            if "cities" in self.CONFIG.keys():
                # Cargar los ficheros correspondientes (múltiples ciudades)
                all_data = []
                stemming_dict = []
                for city in self.CONFIG["cities"]:
                    ct_dt, ct_st = self.load_city(city)
                    all_data = pd.concat([all_data, ct_dt])
                    stemming_dict = pd.concat([stemming_dict, ct_st])

                all_data = all_data.sample(frac=1, random_state=self.CONFIG["seed"]).reset_index(drop=True)
                stemming_dict = stemming_dict.drop_duplicates().sort_values("stemming").reset_index(drop=True)

            else:
                # Cargar las reviews
                all_data, stemming_dict = self.load_city(self.CONFIG["city"])

            # Restaurantes con X o más reseñas
            r_mth_x = all_data.groupby("restaurantId").apply(lambda x: 0 if len(x) < self.CONFIG["min_reviews_rst"] else 1).reset_index()
            r_mth_x = r_mth_x.loc[r_mth_x[0] == 1]["restaurantId"].values
            print_g("%d restaurants with %d or more reviews." % (len(r_mth_x), self.CONFIG["min_reviews_rst"]))
            all_data = all_data.loc[all_data["restaurantId"].isin(r_mth_x)]

            # Usuarios con X o más reseñas
            u_mth_x = all_data.groupby("userId").apply(lambda x: 0 if len(x) < self.CONFIG["min_reviews_usr"] else 1).reset_index()
            u_mth_x = u_mth_x.loc[u_mth_x[0] == 1]["userId"].values
            all_data = all_data.loc[all_data["userId"].isin(u_mth_x)]

            # Obtener los datos del conjunto tras el filtrado (para el paper)
            print("· Nº de ejemplos resultantes: %d" % len(all_data))
            print("· Nº de restaurantes: %d" % len(all_data.restaurantId.unique()))
            print("· Nº de usuarios: %d" % len(all_data.userId.unique()))
            print("· Nº restaurantes medio por usuario: %f" % all_data.groupby("userId").apply(lambda x: len(x.restaurantId.unique())).mean())
            os.makedirs("data/raw/", exist_ok=True)
            all_data[["reviewId", "userId", "restaurantId", "rating", "text"]].to_pickle("data/raw/%s.pkl" % self.CONFIG["city"])

            # Crear id de restaurantes (para el ONE-HOT)
            rst_newid = pd.DataFrame(zip(r_mth_x, range(len(r_mth_x))), columns=["restaurantId", "id_restaurant"])
            all_data = all_data.merge(rst_newid, on="restaurantId")
            all_data = all_data.drop(columns=["restaurantId"])

            # Mezclar las reviews
            all_data = all_data.sample(frac=1, random_state=self.CONFIG["seed"]).reset_index(drop=True)

            # Crear vectores del BOW
            if self.CONFIG["remove_stopwords"] == 0:
                vectorizer = CountVectorizer(stop_words=None, min_df=self.CONFIG["min_df"], max_features=self.CONFIG["num_palabras"], binary=self.CONFIG["presencia"])
            elif self.CONFIG["remove_stopwords"] == 1:
                vectorizer = CountVectorizer(stop_words=self.SPANISH_STOPWORDS, min_df=self.CONFIG["min_df"], max_features=self.CONFIG["num_palabras"], binary=self.CONFIG["presencia"])
            elif self.CONFIG["remove_stopwords"] == 2:
                if self.CONFIG["lemmatization"]:
                    # Se hace un countvectorizer con todas las palabras para obtener la frecuencia de cada una
                    vectorizer = CountVectorizer(stop_words=None, min_df=self.CONFIG["min_df"], max_features=None, binary=self.CONFIG["presencia"])
                    bow = vectorizer.fit_transform(all_data[self.CONFIG["text_column"]])
                    word_freq = np.asarray(bow.sum(axis=0))[0]
                    # Se obtiene, para cada palabra, su POS (part of speech)
                    word_pos = [w[0].pos_ for w in self.NLP.pipe(vectorizer.get_feature_names())]
                    # Se alamacena todo en un DF para buscar X palabras más frecuentes que cumplan las exigencias
                    word_data = pd.DataFrame(zip(vectorizer.get_feature_names(), word_freq, word_pos), columns=["feature", "freq", "pos"]).sort_values("freq", ascending=False).reset_index(drop=True)
                    selected_words = word_data.loc[word_data.pos.isin(["ADJ", "NOUN"])].iloc[:self.CONFIG["num_palabras"]].reset_index(drop=True)
                    # Todas las que no sean seleccionadas, se consideran stopwords
                    stop_words = word_data.loc[~word_data.feature.isin(selected_words.feature)].feature.tolist()
                    vectorizer = CountVectorizer(stop_words=stop_words, min_df=self.CONFIG["min_df"], max_features=self.CONFIG["num_palabras"], binary=self.CONFIG["presencia"])
                else:
                    print_e("La selección automática de palabras requiere de lemmatización.")
                    exit()

            bow = vectorizer.fit_transform(all_data[self.CONFIG["text_column"]])

            # Cada palabra corresponde con cada columna de las <self.CONFIG["num_palabras"]>
            features_name = vectorizer.get_feature_names()
            np.savetxt(self.DATASET_PATH+"features.csv", features_name, fmt="%s")

            # Normalizar vector de cada review
            normed_bow = normalize(bow.todense(), axis=1, norm='l1')

            # Incroporar BOW en los datos
            all_data["bow"] = normed_bow.tolist()

            # Tokenizar las palabras (Asociar cada palabra a un índice [WORD_INDEX])
            if self.CONFIG["n_max_words"] == 0:
                tokenizer_txt = tf.keras.preprocessing.text.Tokenizer()
            else:
                tokenizer_txt = tf.keras.preprocessing.text.Tokenizer(num_words=self.CONFIG["n_max_words"])

            tokenizer_txt.fit_on_texts(all_data[self.CONFIG["text_column"]])
            word_index = tokenizer_txt.word_index

            # Si se utilizan todas
            if self.CONFIG["n_max_words"] == 0:
                print_g("Intervienen %d palabras y utilizamos todas." % len(word_index))
            # Quedarse con las "n_max_words" palabras más frecuentes
            else:
                word_counts = tokenizer_txt.word_counts
                word_counts = {k: v for k, v in sorted(word_counts.items(), key=lambda item: item[1], reverse=True)}
                mas_frecuentes = list(word_counts.keys())[0:self.CONFIG["n_max_words"]]
                print_g("Intervienen %d palabras, pero nos quedamos con las %d más frecuentes." % (len(word_index), self.CONFIG["n_max_words"]))
                word_index = {x: word_index[x] for x in mas_frecuentes}

            # Transformar frases a secuencias de números según [WORD_INDEX] y añadir al set
            text_sequences = tokenizer_txt.texts_to_sequences(all_data[self.CONFIG["text_column"]])
            all_data["seq"] = text_sequences

            # Train Test split asegurandose de que en Train están todos los restaurantes.
            def data_split(rst_rvws):
                for_dev_tst = int(len(rst_rvws) * (self.CONFIG["test_dev_split"]*2))//2
                if for_dev_tst > 0:
                    rst_rvws.iloc[-for_dev_tst:, rst_rvws.columns.get_loc("test")] = 1  # Últimas x para test
                    rst_rvws.iloc[-for_dev_tst*2:-for_dev_tst, rst_rvws.columns.get_loc("dev")] = 1  # Penúltimas x para dev
                return rst_rvws

            all_data["dev"] = 0
            all_data["test"] = 0
            all_data = all_data.groupby("id_restaurant").apply(data_split)

            # Truncar el padding?
            max_len_padding = None
            if self.CONFIG["truncate_padding"]:
                seq_lens = all_data.loc[(all_data["dev"] == 0) & (all_data["test"] == 0)]["seq"].apply(lambda x: len(x)).values
                max_len_padding = int(seq_lens.mean() + seq_lens.std() * 2)

            # Añadir al set con el padding
            seq_w_pad = tf.keras.preprocessing.sequence.pad_sequences(all_data["seq"].values, maxlen=max_len_padding)
            all_data["seq"] = seq_w_pad.tolist()
            max_len_padding = seq_w_pad.shape[1]

            # Separar los conjuntos finales
            train_dev = all_data.loc[all_data["test"] == 0].drop(columns=["test"])
            test = all_data.loc[all_data["test"] == 1].drop(columns=["dev", "test"])

            # Almacenar pickles
            to_pickle(self.DATASET_PATH, "VECTORIZER", vectorizer)
            to_pickle(self.DATASET_PATH, "FEATURES_NAME", features_name)
            to_pickle(self.DATASET_PATH, "N_RST", len(all_data["id_restaurant"].unique()))
            to_pickle(self.DATASET_PATH, "STEMMING_DICT", stemming_dict)

            to_pickle(self.DATASET_PATH, "WORD_INDEX", word_index)
            to_pickle(self.DATASET_PATH, "VOCAB_SIZE", len(word_index) + 1)
            to_pickle(self.DATASET_PATH, "MAX_LEN_PADDING", max_len_padding)
            to_pickle(self.DATASET_PATH, "TEXT_TOKENIZER", tokenizer_txt)

            to_pickle(self.DATASET_PATH, "TRAIN_DEV", train_dev)
            to_pickle(self.DATASET_PATH, "TEST", test)

            return self.get_dict_data(self.DATASET_PATH, load)
