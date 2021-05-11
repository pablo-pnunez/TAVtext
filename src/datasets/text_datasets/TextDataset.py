# -*- coding: utf-8 -*-
from src.datasets.DatasetClass import DatasetClass

import re
import nltk
import pandas as pd
from tqdm import tqdm
import numpy as np
from unicodedata import normalize
from nltk.corpus import stopwords
import mapply
from nltk.stem import SnowballStemmer, PorterStemmer


class TextDataset(DatasetClass):

    def __init__(self, config):
        self.SPANISH_STOPWORDS = self.__get_es_stopwords__()
        DatasetClass.__init__(self, config=config)

    def prerpocess_text(self, text):

        # A minusculas
        text = text.lower()

        # Eliminar formatos (/n /t ...)
        rgx_b = r'(\\.)+'
        text = re.sub(rgx_b, ' ', text).strip()

        # Cambiar signos de puntuación por espacios
        rgx_a = r'\s*[^\w\s]+\s*'
        text = re.sub(rgx_a, ' ', text).strip()

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
            self.SPANISH_STOPWORDS = [stemmer.stem(w) for w in self.SPANISH_STOPWORDS]

        # Stemming?
        if self.CONFIG["stemming"]:
            stemmer = SnowballStemmer('spanish')
            text = " ".join([stemmer.stem(w) for w in text.split(" ")])
            self.SPANISH_STOPWORDS = [stemmer.stem(w) for w in self.SPANISH_STOPWORDS]

        return text


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
        
        # Preprocesar los textos
        mapply.init(
            n_workers=-1,
            chunk_size=100,
            max_chunks_per_worker=8,
            progressbar=True
        )   

        rev["text"] = rev["text"].mapply(self.prerpocess_text)
        rev["title"] = rev["title"].mapply(self.prerpocess_text)

        # Obtener número de palabras de las reviews y del título
        rev["n_words_text"] = rev["text"].apply(lambda x: 0 if len(x) == 0 else len(x.split(" ")))
        rev["n_words_title"] = rev["title"].apply(lambda x: 0 if len(x) == 0 else len(x.split(" ")))

        # Eliminar reviews de tamaño 0 en texto y titulo
        rev = rev.loc[(rev["n_words_text"] > 0) & (rev["n_words_title"] > 0)]

        # Obtener número de caracteres de las reviews y eliminar aquellas con más de 2000
        rev["n_char_text"] = rev["text"].apply(lambda x: len(x))
        rev = rev.loc[rev["n_char_text"] <= 2000]

        return rev

    def __get_es_stopwords__(self):
        # nltk.download('stopwords')

        spanish_stopwords = stopwords.words('spanish')
        spanish_stopwords += ['ademas', 'alli', 'aqui', 'asturias', 'asi', 'aunque', 'cada', 'casa', 'casi',
                              'comido', 'comimos', 'cosas', 'creo', 'decir', 'despues', 'dos', 'dia', 'fin', 'gijon',
                              'gijon', 'hace', 'hacer', 'hora', 'ido', 'igual', 'ir', 'lado', 'luego', 'mas', 'merece',
                              'mismo', 'momento', 'mucha', 'muchas', 'parece', 'parte', 'pedimos', 'pedir', 'probar',
                              'puede', 'puedes', 'pues', 'punto', 'relacion', 'reservar', 'seguro', 'semana', 'ser',
                              'si',
                              'sido', 'siempre', 'sitio', 'sitios', 'solo', 'si', 'tan', 'tener', 'toda', 'tomar',
                              'tres',
                              'unas', 'varias', 'veces', 'ver', 'verdad', 'vez', 'visita', 'bastante', 'duda', 'gran',
                              'menos', 'no', 'nunca', 'opinion', 'primera', 'primero', 'segundo', 'mejor',
                              'mejores']
        spanish_stopwords += ['alguna', 'asturiana', 'caso', 'centro', 'cierto', 'comentario',
                              'cosa',
                              'cualquier', 'cuanto', 'cuenta', 'da', 'decidimos', 'demasiado', 'dentro', 'destacar',
                              'detalle',
                              'dia', 'dias', 'esperamos', 'esperar', 'general', 'gracias', 'haber', 'hacen', 'hecho',
                              'lleno',
                              'media', 'minutos', 'noche', 'nota', 'poder', 'ponen', 'probado', 'puedo', 'reserva',
                              'resto',
                              'sabor', 'solo', 'tiempo', 'todas', 'tomamos', 'totalmente', 'vamos', 'varios', 'vida',
                              'unico']
        spanish_stopwords += ['ahora', 'aun', 'cerca', 'ciudad', 'cuatro', 'elegir', 'encima', 'falta', 'final',
                              'ganas',
                              'hoy', 'llegamos', 'medio', 'mundo', 'nuevo', 'ocasiones', 'opcion', 'parecio', 'pasar',
                              'pedido',
                              'pesar', 'poner', 'probamos', 'pronto', 'realmente', 'salimos', 'sirven', 'situado',
                              'tampoco',
                              'tarde', 'tipo', 'va', 'vas', 'voy']
        spanish_stopwords += ['come', 'demas', 'ello', 'etc', 'incluso', 'llegar', 'pasado', 'primer', 'pusieron',
                              'quedamos', 'quieres', 'saludo', 'tambien', 'trabajo', 'tras', 'verano']
        spanish_stopwords += ['algun', 'cenamos', 'comentarios', 'comiendo', 'dan', 'dice', 'domingo', 'ofrecen',
                              'razonable',
                              'tamano']
        spanish_stopwords += ['nadie', 'ningun', 'opiniones', 'quizas', 'san', 'sino']
        spanish_stopwords += ['atendio', 'pega', 'sabado']
        spanish_stopwords += ['dicho', 'par', 'total']
        spanish_stopwords += ['años', 'año', 'ultima', 'comer']


        return spanish_stopwords
