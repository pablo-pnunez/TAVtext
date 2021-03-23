# -*- coding: utf-8 -*-
from src.datasets.DatasetClass import *

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from unicodedata import normalize


class TextDataset(DatasetClass):

    def __init__(self, config):
        self.SPANISH_STOPWORDS = self.__get_es_stopwords__()
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

        # Eliminar formatos (/n /t ...)
        rgx_b = r'(\\.)+'
        rev['text'] = rev['text'].apply(lambda x: re.sub(rgx_b, ' ', x).strip())
        rev['title'] = rev['title'].apply(lambda x: re.sub(rgx_b, ' ', x).strip())

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
        spanish_stopwords += ['además', 'allí', 'aquí', 'asturias', 'así', 'aunque', 'años', 'cada', 'casa', 'casi',
                              'comido', 'comimos', 'cosas', 'creo', 'decir', 'después', 'dos', 'día', 'fin', 'gijon',
                              'gijón', 'hace', 'hacer', 'hora', 'ido', 'igual', 'ir', 'lado', 'luego', 'mas', 'merece',
                              'mismo', 'momento', 'mucha', 'muchas', 'parece', 'parte', 'pedimos', 'pedir', 'probar',
                              'puede', 'puedes', 'pues', 'punto', 'relación', 'reservar', 'seguro', 'semana', 'ser',
                              'si',
                              'sido', 'siempre', 'sitio', 'sitios', 'solo', 'sí', 'tan', 'tener', 'toda', 'tomar',
                              'tres',
                              'unas', 'varias', 'veces', 'ver', 'verdad', 'vez', 'visita', 'bastante', 'duda', 'gran',
                              'menos', 'no', 'nunca', 'opinión', 'primera', 'primero', 'segundo', '10', 'mejor',
                              'mejores']
        spanish_stopwords += ['100', '15', '20', '30', 'alguna', 'asturiana', 'caso', 'centro', 'cierto', 'comentario',
                              'cosa',
                              'cualquier', 'cuanto', 'cuenta', 'da', 'decidimos', 'demasiado', 'dentro', 'destacar',
                              'detalle',
                              'dia', 'días', 'esperamos', 'esperar', 'general', 'gracias', 'haber', 'hacen', 'hecho',
                              'lleno',
                              'media', 'minutos', 'noche', 'nota', 'poder', 'ponen', 'probado', 'puedo', 'reserva',
                              'resto',
                              'sabor', 'sólo', 'tiempo', 'todas', 'tomamos', 'totalmente', 'vamos', 'varios', 'vida',
                              'único']
        spanish_stopwords += ['50', 'ahora', 'aún', 'cerca', 'ciudad', 'cuatro', 'elegir', 'encima', 'falta', 'final',
                              'ganas',
                              'hoy', 'llegamos', 'medio', 'mundo', 'nuevo', 'ocasiones', 'opción', 'pareció', 'pasar',
                              'pedido',
                              'pesar', 'poner', 'probamos', 'pronto', 'realmente', 'salimos', 'sirven', 'situado',
                              'tampoco',
                              'tarde', 'tipo', 'va', 'vas', 'voy']
        spanish_stopwords += ['12', 'come', 'demás', 'ello', 'etc', 'incluso', 'llegar', 'pasado', 'primer', 'pusieron',
                              'quedamos', 'quieres', 'saludo', 'tambien', 'trabajo', 'tras', 'verano']
        spanish_stopwords += ['algún', 'cenamos', 'comentarios', 'comiendo', 'dan', 'dice', 'domingo', 'ofrecen',
                              'razonable',
                              'tamaño']
        spanish_stopwords += ['nadie', 'ningún', 'opiniones', 'quizás', 'san', 'sino']
        spanish_stopwords += ['atendió', 'pega', 'sábado']
        spanish_stopwords += ['dicho', 'par', 'total']

        return spanish_stopwords
