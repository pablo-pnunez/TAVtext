# -*- coding: utf-8 -*-
from src.datasets.DatasetClass import DatasetClass

import re
import gzip
import json
import mapply
import pandas as pd
from unicodedata import normalize
import en_core_web_sm as spacy_en_model
from nltk.stem import SnowballStemmer, PorterStemmer


class AmazonDataset(DatasetClass):

    def __init__(self, config, load):
        # Solo hay datos en inglés
        self.NLP = spacy_en_model.load(disable=["parser", "ner"])
        DatasetClass.__init__(self, config=config, load=load)

    def prerpocess_text(self, text):

        # A minusculas, eliminar números, acentos, stemming etc...
        text = self.__preprocess_text_base__(text)

        return text

    def __preprocess_text_base__(self, text):

        # A minusculas
        text = text.lower()

        # Eliminar formatos (/n /t ...)
        rgx_b = r'(\\.)+'
        text = re.sub(rgx_b, ' ', text).strip()

        # Cambiar signos de puntuación por espacios
        rgx_a = r'\s*[^\w\s]+\s*'
        text = re.sub(rgx_a, ' ', text).strip()

        # Tagging & Lemmatización
        if self.CONFIG["lemmatization"]:
            text = " ".join([e.lemma_ for e in self.NLP(text)]).lower()  # alguna palabra la pone mayúscula

        # Eliminar accentos?
        if self.CONFIG["remove_accents"]:
            rgx_c = r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+"
            text = normalize('NFC', re.sub(rgx_c, r"\1", normalize("NFD", text), 0, re.I))

        # Eliminar números?
        if self.CONFIG["remove_numbers"]:
            rgx_d = r"\s*\d+\s*"
            text = re.sub(rgx_d, ' ', text).strip()

        # Eliminar plurales?
        if self.CONFIG["remove_plurals"]:
            stemmer = PorterStemmer()
            text = " ".join([stemmer.stem(w) for w in text.split(" ")])

        # Stemming?
        if self.CONFIG["stemming"]:
            print("Actualizar código para otros idiomas") 
            exit()
            stemmer = SnowballStemmer('spanish')
            text = " ".join([stemmer.stem(w) for w in text.split(" ")])
     
        return text

    def __parse_gzip_file__(self, path):
        g = gzip.open(path, 'rb')
        for line in g:
            yield json.loads(line)

    def __get_pandas_from_gzip__(self, path):
        i = 0
        df = {}
        for d in self.__parse_gzip_file__(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')

    def load_category(self, category):
        """Carga los datos de una categoría, quedandose con las columnas relevantes"""

        # Cargar reviews, renombrar columnas y ordenar por fecha para crear "reviewId"
        rev = self.__get_pandas_from_gzip__(f'{self.CONFIG["data_path"]}{category}/data.json.gz')
        columns_name_dict = {"reviewerID": "userId", "overall": "rating", "asin": "itemId", "reviewText": "text", "summary": "title", "unixReviewTime": "date"}
        rev = rev.rename(columns=columns_name_dict).sort_values("date").reset_index(drop=True)
        rev.insert(0, "reviewId", range(len(rev)))

        # Cargar metadata
        meta = self.__get_pandas_from_gzip__(f'{self.CONFIG["data_path"]}{category}/metadata.json.gz')
        columns_name_dict = {"asin": "itemId", "title": "name"}
        meta = meta.rename(columns=columns_name_dict).sort_values("date").reset_index(drop=True)
        meta = meta.astype({'name': 'str'})
        meta = meta[~meta["name"].str.contains('getTime')]  # Eliminar algunos que tienen HTML en el título por error

        # Quedarse con columnas relevantes y casting de algunas
        rev = rev[['reviewId', 'userId', 'itemId', 'rating', 'date', 'text', 'title']]
        rev["rating"] = rev.rating*10
        rev = rev.astype({'reviewId': 'int64', 'rating': 'int64'})

        # Concatenar items y reviews
        rev = rev.merge(meta[["itemId", "name"]], left_on="itemId", right_on="itemId", how="left")

        # Eliminar reviews vacías (que tengan NAN, pero sigue habiendo reviews con texto=="")
        rev = rev.loc[(~rev["text"].isna()) & (~rev["title"].isna())]

        # Eliminar reviews de usuarios que escriben lo mismo en muchos ítems con la misma fecha
        #  users_reviews_same_date = rev.groupby(["userId", "date"]).reviewId.count().reset_index()
        #  good_users = users_reviews_same_date[users_reviews_same_date.reviewId==1].userId.unique()

        # Preprocesar los textos
        mapply.init(
            n_workers=-1,
            chunk_size=100,
            max_chunks_per_worker=8,
            progressbar=True
        )

        # Preprocesamos el texto (a minusculas, quitar números ...)
        rev["text"] = rev["text"].mapply(self.__preprocess_text_base__)

        # Obtener número de palabras de las reviews y del título
        rev["n_words_text"] = rev["text"].apply(lambda x: 0 if len(x) == 0 else len(x.split(" ")))
        rev["n_words_title"] = rev["title"].apply(lambda x: 0 if len(x) == 0 else len(x.split(" ")))

        # Eliminar reviews de tamaño 0 en texto y titulo
        rev = rev.loc[(rev["n_words_text"] > 0) & (rev["n_words_title"] > 0)]

        # Obtener número de caracteres de las reviews y eliminar aquellas con más de 2000
        rev["n_char_text"] = rev["text"].apply(lambda x: len(x))
        rev = rev.loc[rev["n_char_text"] <= 2000]

        return rev
