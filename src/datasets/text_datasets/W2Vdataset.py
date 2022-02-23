# -*- coding: utf-8 -*-

from src.datasets.text_datasets.TextDataset import TextDataset
from src.Common import to_pickle

import pandas as pd


class W2Vdataset(TextDataset):

    def __init__(self, config, load=None):
        TextDataset.__init__(self, config=config, load=load)

    def get_data(self, load=["ALL_TEXTS", "ALL_TITLES", "STEMMING_DICT"]):

        # Cargar los datos
        dict_data = self.get_dict_data(self.DATASET_PATH, load)

        # Si ya existen, retornar
        if dict_data is not False and len(dict_data) == len(load):
            return dict_data
        # Si no existe, crear
        else:
            # Cargar los ficheros correspondientes (múltiples ciudades)
            all_data = pd.DataFrame()
            stemming_dict = pd.DataFrame()
            for city in self.CONFIG["cities"]:
                ct_dt, ct_st = self.load_city(city)
                all_data = pd.concat([all_data, ct_dt])
                stemming_dict = pd.concat([stemming_dict, ct_st])

            # Mezclar las reviews
            all_data = all_data.sample(frac=1, random_state=self.CONFIG["seed"]).reset_index(drop=True)
            stemming_dict = stemming_dict.drop_duplicates().sort_values("stemming").reset_index(drop=True)

            # Almacenar pickles
            to_pickle(self.DATASET_PATH, "ALL_TEXTS", all_data["text"].str.split())
            to_pickle(self.DATASET_PATH, "ALL_TITLES", all_data["title"].str.split())
            to_pickle(self.DATASET_PATH, "STEMMING_DICT", stemming_dict)

            return self.get_dict_data(self.DATASET_PATH, load)
