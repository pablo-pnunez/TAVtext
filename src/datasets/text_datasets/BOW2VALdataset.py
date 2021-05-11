# -*- coding: utf-8 -*-

from src.datasets.text_datasets.TextDataset import TextDataset
from src.Common import to_pickle, print_g

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer

class BOW2VALdataset(TextDataset):

    def __init__(self, config):
        TextDataset.__init__(self, config=config)

    def get_data(self, load=["TRAIN_DEV", "TEST", "VECTORIZER", "FEATURES_NAME"]):


        # Cargar los datos
        dict_data = self.get_dict_data(self.DATASET_PATH, load)

        # Si ya existen, retornar
        if dict_data:
            return dict_data
        # Si no existe, crear
        else:
            # Cargar las reviews
            all_data = self.load_city(self.CONFIG["city"])

            # Mezclar las reviews
            all_data = all_data.sample(frac=1, random_state=self.CONFIG["seed"]).reset_index(drop=True)

            # Crear vectores del BOW
            vectorizer = CountVectorizer(stop_words=self.SPANISH_STOPWORDS, min_df=self.CONFIG["min_df"], max_features=self.CONFIG["num_palabras"], binary=self.CONFIG["presencia"])  # Frecuencia
            bow = vectorizer.fit_transform(all_data[self.CONFIG["text_column"]])

            # El vocabulary_ es un diccionario {palabra:idx_columna}. Se ordena para saber a que palabra corresponde cada columna de las <self.CONFIG["num_palabras"]>
            features_name = sorted(vectorizer.vocabulary_)

            # Normalizar vector de cada review
            normed_bow = normalize(bow.todense(), axis=1, norm='l1')

            # Incroporar BOW en los datos
            all_data["bow"] = normed_bow.tolist()

            # Train-Dev, Test split
            for_dev_tst = int(len(all_data) * (self.CONFIG["test_dev_split"] * 2)) // 2
            all_data["dev"] = 0
            all_data["test"] = 0
            all_data.iloc[-for_dev_tst:, all_data.columns.get_loc("test")] = 1  # Últimas x para test
            all_data.iloc[-for_dev_tst * 2:-for_dev_tst, all_data.columns.get_loc("dev")] = 1  # Penúltimas x para dev

            # Separar los conjuntos finales
            train_dev = all_data.loc[all_data["test"] == 0].drop(columns=["test"])
            test = all_data.loc[all_data["test"] == 1].drop(columns=["dev", "test"])

            # Almacenar pickles
            to_pickle(self.DATASET_PATH, "VECTORIZER", vectorizer)
            to_pickle(self.DATASET_PATH, "FEATURES_NAME", features_name)
            to_pickle(self.DATASET_PATH, "TRAIN_DEV", train_dev)
            to_pickle(self.DATASET_PATH, "TEST", test)
            
            return self.get_dict_data(self.DATASET_PATH, load)
