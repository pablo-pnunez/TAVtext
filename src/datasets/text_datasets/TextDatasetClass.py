# -*- coding: utf-8 -*-
from src.datasets.DatasetClass import *

from unicodedata import normalize
import pandas as pd
import re


class TextDatasetClass(DatasetClass):

    def __init__(self, config):
        DatasetClass.__init__(self, config=config)

    def load_city(self, city):
        """Carga los datos de una ciudad, quedandose con las columnas relevantes"""

        # Cargar restaurantes
        res = pd.read_pickle(self.CONFIG["data_path"] + city + "_data/restaurants.pkl")

        # Cargar reviews
        rev = pd.read_pickle(self.CONFIG["data_path"] + city + "_data/reviews.pkl")
        rev = rev[['reviewId', 'userId', 'restaurantId', 'rating', 'date', 'language', 'text', 'title', 'url']]
        rev["city"] = city

        # Concatenar restaurantes y reviews
        rev = rev.merge(res[["id", "name"]], left_on="restaurantId", right_on="id", how="left")
        rev = rev.drop(columns=["id"])

        # Eliminar reviews vacías (que tengan NAN, pero sigue habiendo reviews con texto=="")
        rev = rev.loc[(~rev["text"].isna()) & (~rev["title"].isna())]

        # Casting a int de algunas columnas
        rev = rev.astype({'reviewId': 'int64', 'restaurantId': 'int64', 'rating': 'int64'})

        # A minusculas
        rev['text'] = rev['text'].apply(lambda x: '%r' % x.lower())
        rev['title'] = rev['title'].apply(lambda x: '%r' % x.lower())

        # Eliminar formatos (/n /t ...) y pasa a minuscula
        rgx_b = r'(\\.)'
        rev['text'] = rev['text'].apply(lambda x: re.sub(rgx_b, '', x).strip())
        rev['title'] = rev['title'].apply(lambda x: re.sub(rgx_b, '', x).strip())

        # Cambiar signos de puntuación por espacios
        rgx_a = r'\s*[^\w\s]+\s*'
        rev['text'] = rev['text'].apply(lambda x: re.sub(rgx_a, ' ', x).strip())
        rev['title'] = rev['title'].apply(lambda x: re.sub(rgx_a, ' ', x).strip())

        # Eliminar accentos?
        if self.CONFIG["remove_accents"]:
            rgx_c = r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+"
            rev['text'] = rev['text'].apply(lambda x: normalize('NFC', re.sub(rgx_c, r"\1", normalize("NFD", x), 0, re.I)))
            rev['title'] = rev['title'].apply(lambda x: normalize('NFC', re.sub(rgx_c, r"\1", normalize("NFD", x), 0, re.I)))

        # Eliminar números?
        if self.CONFIG["remove_numbers"]:
            rgx_d = r"\s*\d+\s*"
            rev['text'] = rev['text'].apply(lambda x: re.sub(rgx_d, ' ', x).strip())
            rev['title'] = rev['title'].apply(lambda x: re.sub(rgx_d, ' ', x).strip())

        # Obtener número de palabras de las reviews y del texto
        rev["n_words_text"] = rev["text"].apply(lambda x: 0 if len(x) == 0 else len(x.split(" ")))
        rev["n_words_title"] = rev["title"].apply(lambda x: 0 if len(x) == 0 else len(x.split(" ")))

        # Eliminar reviews de tamaño 0 en texto y titulo
        rev = rev.loc[(rev["n_words_text"] > 0) & (rev["n_words_title"] > 0)]

        return rev




