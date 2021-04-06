# -*- coding: utf-8 -*-

from src.datasets.text_datasets.TextDataset import *
from src.Common import to_pickle, print_g

import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer


class BOW2RSTdataset(TextDataset):

    def __init__(self, config):
        TextDataset.__init__(self, config=config)

    def get_data(self, load=["TRAIN_DEV", "TEST",  "VECTORIZER", "FEATURES_NAME", "N_RST"]):

        # Cargar los datos
        dict_data = self.get_dict_data(self.DATASET_PATH, load)

        # Si ya existen, retornar
        if dict_data:
            return dict_data
        # Si no existe, crear
        else:
            # Cargar las reviews
            all_data = self.load_city(self.CONFIG["city"])

            # Restaurantes con X o más reseñas
            r_mth_x = all_data.groupby("restaurantId").apply(lambda x: 0 if len(x) < self.CONFIG["min_reviews_rst"] else 1).reset_index()
            r_mth_x = r_mth_x.loc[r_mth_x[0] == 1]["restaurantId"].values
            print_g("%d restaurants with %d or more reviews." % (len(r_mth_x), self.CONFIG["min_reviews_rst"]))
            all_data = all_data.loc[all_data["restaurantId"].isin(r_mth_x)]

            # Usuarios con X o más reseñas
            u_mth_x = all_data.groupby("userId").apply(lambda x: 0 if len(x) < self.CONFIG["min_reviews_usr"] else 1).reset_index()
            u_mth_x = u_mth_x.loc[u_mth_x[0] == 1]["userId"].values
            all_data = all_data.loc[all_data["userId"].isin(u_mth_x)]

            # Crear id de restaurantes (para el ONE-HOT)
            rst_newid = pd.DataFrame(zip(r_mth_x, range(len(r_mth_x))), columns=["restaurantId", "id_restaurant"])
            all_data = all_data.merge(rst_newid, on="restaurantId")
            all_data = all_data.drop(columns=["restaurantId"])

            # Crear vectores del BOW
            vectorizer = CountVectorizer(stop_words=self.SPANISH_STOPWORDS, min_df=self.CONFIG["min_df"], max_features=self.CONFIG["num_palabras"], binary=self.CONFIG["presencia"])  # Frecuencia
            bow = vectorizer.fit_transform(all_data[self.CONFIG["text_column"]])

            # El vocabulary_ es un diccionario {palabra:idx_columna}. Se ordena para saber a que palabra corresponde cada columna de las <self.CONFIG["num_palabras"]>
            features_name = sorted(vectorizer.vocabulary_)

            # Normalizar vector de cada review
            normed_bow = normalize(bow.todense(), axis=1, norm='l1')

            # Incroporar BOW en los datos
            all_data["bow"] = normed_bow.tolist()

            # Mezclar las reviews
            all_data = all_data.sample(frac=1, random_state=self.CONFIG["seed"]).reset_index(drop=True)

            # Train Test split asegurandose de que en Train están todos los restaurantes.
            def data_split(rst_rvws):
                for_dev_tst = int(len(rst_rvws) * (self.CONFIG["test_dev_split"]*2))//2
                if for_dev_tst > 0:
                    rst_rvws.iloc[-for_dev_tst:, rst_rvws.columns.get_loc("test")] = 1  # Últimas x para test
                    rst_rvws.iloc[-for_dev_tst*2:-for_dev_tst, rst_rvws.columns.get_loc("dev")] = 1  # Penúltimas x para dev
                return rst_rvws

            all_data["dev"] = 0; all_data["test"] = 0
            all_data = all_data.groupby("id_restaurant").apply(data_split)

            train_dev = all_data.loc[all_data["test"] == 0].drop(columns=["test"])
            test = all_data.loc[all_data["test"] == 1].drop(columns=["dev", "test"])

            # Almacenar pickles
            to_pickle(self.DATASET_PATH, "VECTORIZER", vectorizer)
            to_pickle(self.DATASET_PATH, "FEATURES_NAME", features_name)
            to_pickle(self.DATASET_PATH, "N_RST", len(all_data["id_restaurant"].unique()))
            to_pickle(self.DATASET_PATH, "TRAIN_DEV", train_dev)
            to_pickle(self.DATASET_PATH, "TEST", test)

            return self.get_dict_data(self.DATASET_PATH, load)


