# -*- coding: utf-8 -*-

from src.datasets.text_datasets.TextDataset import TextDataset
from src.Common import to_pickle

import pandas as pd


class W2Vdataset(TextDataset):

    def __init__(self, config):
        TextDataset.__init__(self, config=config)

    def get_data(self, load=["ALL_TEXTS", "ALL_TITLES"]):

        # Cargar los datos
        dict_data = self.get_dict_data(self.DATASET_PATH, load)

        # Si ya existen, retornar
        if dict_data:
            return dict_data
        # Si no existe, crear
        else:
            # Cargar los ficheros correspondientes (múltiples ciudades)
            all_data = pd.concat([self.load_city(city) for city in self.CONFIG["cities"]])

            # Mezclar las reviews
            all_data = all_data.sample(frac=1, random_state=self.CONFIG["seed"]).reset_index(drop=True)

            # Almacenar pickles
            to_pickle(self.DATASET_PATH, "ALL_TEXTS", all_data["text"].str.split())
            to_pickle(self.DATASET_PATH, "ALL_TITLES", all_data["title"].str.split())

            return self.get_dict_data(self.DATASET_PATH, load)
