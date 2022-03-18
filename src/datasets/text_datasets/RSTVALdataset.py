# -*- coding: utf-8 -*-

from src.datasets.text_datasets.TextDataset import *
from src.Common import to_pickle, print_g, print_e

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from functools import partial
from multiprocessing import Pool
from scipy.sparse import csc_matrix
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer

'''
def features_pos(w, RVW, NLP, text_column, seed):
    # Para cada una de las palabras más frecuentes, se miran las reviews que la contienen
    rvs_with_w = RVW.loc[RVW[text_column].apply(lambda x: w in x)]
    # Para evitar sobrecarga, nos quedamos con 20 reviews como máximo
    rvs_with_w = rvs_with_w.sample(min(len(rvs_with_w), 20), random_state=seed)
    # Buscamos la posición concreta de la palabra dentro de la review
    rvs_with_w["w_loc"] = rvs_with_w[text_column].apply(lambda x: np.argwhere(np.asarray(x) == w).flatten()[0])
    # Obtenemos un vector de POS para cada palabra de las reviews y nos quedamos con el POS de la que nos interesa
    rvs_with_w["w_pos"] = rvs_with_w.apply(lambda x: [n.pos_ for n in NLP(" ".join(x[text_column]))][x.w_loc],  1)
    # Finalmente, como habrá varios POS, nos quedamos con el que aparezca en la mayoría de reviews
    poses, p_counts = np.unique(rvs_with_w.w_pos, return_counts=True)
    # Retornar resultado
    print(w)
    return poses[p_counts.argmax()]
'''

class RSTVALdataset(TextDataset):

    def __init__(self, config, load=None):
        TextDataset.__init__(self, config=config, load=load)

    def get_data(self, load=["TRAIN_DEV", "TEST", "WORD_INDEX", "VOCAB_SIZE", "MAX_LEN_PADDING", "TEXT_TOKENIZER", "VECTORIZER", "FEATURES_NAME", "N_RST", "STEMMING_DICT"]):

        # Cargar los datos
        dict_data = self.get_dict_data(self.DATASET_PATH, load)

        # Si ya existen, retornar
        if dict_data is not False and len(dict_data) == len(load):
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
            if self.CONFIG["remove_stopwords"] == 0:  # Solo más frecuentes
                vectorizer = CountVectorizer(stop_words=None, min_df=self.CONFIG["min_df"], max_features=self.CONFIG["bow_pct_words"], binary=self.CONFIG["presencia"])
            elif self.CONFIG["remove_stopwords"] == 1:  # Más frecuentes + stopwords manual
                print("Actualizar código para otros idiomas"); exit()
                vectorizer = CountVectorizer(stop_words=self.STOPWORDS, min_df=self.CONFIG["min_df"], max_features=self.CONFIG["bow_pct_words"], binary=self.CONFIG["presencia"])
            elif self.CONFIG["remove_stopwords"] == 2:  # Más frecuentes + stopwords automático
                if self.CONFIG["lemmatization"]:
                    # Se hace un countvectorizer con todas las palabras para obtener la frecuencia de cada una
                    vectorizer = CountVectorizer(stop_words=None, min_df=self.CONFIG["min_df"], max_features=None, binary=self.CONFIG["presencia"])
                    bow = vectorizer.fit_transform(all_data[self.CONFIG["text_column"]])
                    word_freq = np.asarray(bow.sum(axis=0))[0]

                    # Hay que obtener el POS (part of speech) de cada palabra en su contexto (si hay varios, el más habitual)
                    pos_values = np.array(["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"])

                    # · Obtenemos los conjuntos relevantes
                    RVW_CP = all_data[[self.CONFIG["text_column"]]].copy()
                    # RVW_CP[self.CONFIG["text_column"]] = RVW_CP[self.CONFIG["text_column"]].apply(lambda x: x.split())                
                    features = vectorizer.get_feature_names()
                    
                    # · Luego se crear varias estructuras de datos para almacenar los valores de POS
                    features_df = pd.DataFrame(zip(range(len(features)), features), columns=["id_feature", "feature"])
                    features_df = features_df.set_index("feature")
                    pos_df = pd.DataFrame(zip(range(len(pos_values)), pos_values), columns=["id_pos", "pos"])
                    pos_df = pos_df.set_index("pos")
                    mtrx = np.zeros((len(features), len(pos_values)), dtype=int)

                    # · Hacemos 32 batches para evitar sobrecarga de RAM
                    batches = np.array_split(RVW_CP, 32)

                    # · Para cada batch, obtener POS de sus palabras y almacenar valores en mtrx
                    for idx, b in tqdm(enumerate(batches), desc="Features POS"):
                        POS = list(self.NLP.pipe(b.text, n_process=8))  # 8 es lo mejor

                        all_words = np.concatenate(list(map(lambda x: [(str(w), w.pos_) for w in x], POS)))
                        all_df = pd.DataFrame(all_words, columns=["feature", "pos"])
                        all_df = all_df.loc[all_df.feature.isin(features)].reset_index(drop=True)
                        all_df = all_df.merge(features_df, on="feature").merge(pos_df, on="pos")
                        all_df = all_df.groupby(["id_feature", "id_pos"]).apply(len).reset_index()

                        mtrx[all_df.id_feature, all_df.id_pos] += all_df[0]

                        del all_df, b, POS

                    # · Obtener los POS de las features (más común)
                    word_pos = pos_values[np.apply_along_axis(np.argmax, 1, mtrx)]

                    '''
                    word_pos = []

                    fn_partial = partial(features_pos, RVW=RVW_CP, NLP=self.NLP, text_column=self.CONFIG["text_column"], seed=self.CONFIG["seed"])  # Se fijan los parametros que no varian

                    nppc = 4
                    pool = Pool(processes=nppc)

                    ret = pool.map_async(fn_partial, features)

                    total = int(np.ceil(len(features)/ret._chunksize))
                    pbar = tqdm(total=total)

                    while not ret.ready():
                        pbar.n = total-ret._number_left
                        pbar.last_print_n = total-ret._number_left
                        pbar.refresh()
                        ret.wait(timeout=1)
                    pbar.n = total
                    pbar.last_print_n = total
                    pbar.refresh()
                    pbar.close()

                    word_pos = ret.get()
                    '''

                    '''

                    for ret in tqdm(pool.imap(fn_partial, features, chunksize=200), total=len(features)):
                        word_pos.append(ret)

                    pool.close()
                    pool.join()
                    '''

                    '''
                    for w in tqdm(vectorizer.get_feature_names()):
                        # · Para cada una de las palabras más frecuentes, se miran las reviews que la contienen
                        rvs_with_w = rvw_cpy.loc[rvw_cpy[self.CONFIG["text_column"]].apply(lambda x: w in x)]
                        # · Para evitar sobrecarga, nos quedamos con 20 reviews como máximo
                        rvs_with_w = rvs_with_w.sample(min(len(rvs_with_w), 20), random_state=self.CONFIG["seed"])
                        # · Buscamos la posición concreta de la palabra dentro de la review
                        rvs_with_w["w_loc"] = rvs_with_w[self.CONFIG["text_column"]].apply(lambda x: np.argwhere(np.asarray(x) == w).flatten()[0])
                        # · Obtenemos un vector de POS para cada palabra de las reviews y nos quedamos con el POS de la que nos interesa
                        rvs_with_w["w_pos"] = rvs_with_w.apply(lambda x: [n.pos_ for n in self.NLP(" ".join(x[self.CONFIG["text_column"]]))][x.w_loc],  1)
                        # · Finalmente, como habrá varios POS, nos quedamos con el que aparezca en la mayoría de reviews
                        poses, p_counts = np.unique(rvs_with_w.w_pos, return_counts=True)
                        word_pos.append(poses[p_counts.argmax()])
                    '''

                    # Se alamacena todo en un DF para buscar X palabras más frecuentes que cumplan las exigencias
                    word_data = pd.DataFrame(zip(features, word_freq, word_pos), columns=["feature", "freq", "pos"]).sort_values("freq", ascending=False).reset_index(drop=True)
                    word_data.to_excel(self.DATASET_PATH+"all_features.xlsx")

                    selected_words = word_data.loc[word_data.pos.isin(["ADJ", "NOUN"])]
                    num_selected_words = int(len(selected_words)*(self.CONFIG["bow_pct_words"]/100))
                    selected_words = selected_words.iloc[:num_selected_words].reset_index(drop=True)
                    # Todas las que no sean seleccionadas, se consideran stopwords
                    stop_words = word_data.loc[~word_data.feature.isin(selected_words.feature)].feature.tolist()
                    vectorizer = CountVectorizer(stop_words=stop_words, min_df=self.CONFIG["min_df"], max_features=num_selected_words, binary=self.CONFIG["presencia"])
                else:
                    print_e("La selección automática de palabras requiere de lemmatización.")
                    exit()

            bow = vectorizer.fit_transform(all_data[self.CONFIG["text_column"]])

            # Cada palabra corresponde con cada columna de las <self.CONFIG["bow_pct_words"]>
            features_name = vectorizer.get_feature_names()
            np.savetxt(self.DATASET_PATH+"features.csv", features_name, fmt="%s")

            # Normalizar vector de cada review
            # normed_bow = normalize(bow.todense(), axis=1, norm='l1')
            normed_bow = normalize(bow, axis=1, norm='l1')

            # Incroporar BOW en los datos
            # all_data["bow"] = normed_bow.tolist()
            all_data["bow"] = list(map(csc_matrix, normed_bow))

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

    def get_data_stats(self):
        ALL = self.DATA["TRAIN_DEV"].append(self.DATA["TEST"])

        n_revs = len(ALL["reviewId"].unique())
        n_rests = len(ALL["id_restaurant"].unique())
        
        revs_per_res = ALL.groupby("id_restaurant").apply(lambda x: len(x.reviewId.unique()))
        avg_revs_rest = revs_per_res.mean()
        pctg_total_revs_rest_pop = revs_per_res.sort_values().iloc[-1]/n_revs

        avg_rating = ALL.rating.mean()/10
        std_rating = ALL.rating.std()/10

        print("\n".join(map(str, [self.CONFIG["city"],n_revs, n_rests, avg_revs_rest, pctg_total_revs_rest_pop, avg_rating, std_rating])))
