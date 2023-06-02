# -*- coding: utf-8 -*-
from src.datasets.DatasetClass import DatasetClass

import os
import re
import json
import nltk
import spacy
import mapply
import numpy as np
import pandas as pd
from tqdm import tqdm
from unicodedata import normalize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, PorterStemmer


class TripAdvisorDataset(DatasetClass):

    def __init__(self, config, load):
        nltk.download('stopwords', quiet=True)


        if config["city"] in ["gijon", "madrid", "barcelona"]:
            import es_core_news_sm as spacy_es_model  # python -m spacy download es_core_news_sm
            self.NLP = spacy_es_model.load(disable=["parser", "ner", "attribute_ruler"])
            self.STOPWORDS = self.__get_stopwords__(lang="es")

        elif config["city"] in ["newyorkcity", "london"]:
            import en_core_web_sm as spacy_en_model  # python -m spacy download en_core_web_sm
            self.NLP = spacy_en_model.load(disable=["parser", "ner"])
            self.STOPWORDS = self.__get_stopwords__(lang="en")

        else:
            import fr_core_news_sm as spacy_fr_model  # python -m spacy download fr_core_news_sm
            self.NLP = spacy_fr_model.load(disable=["parser", "ner", "attribute_ruler"])
            self.STOPWORDS = self.__get_stopwords__(lang="fr")

        DatasetClass.__init__(self, config=config, load=load)

    def prerpocess_text(self, text):

        # A minusculas, eliminar números, acentos etc...
        text = self.__preprocess_text_base__(text)

        # Hacer stemming si está activado
        text = self.__preprocess_text_stemming__(text)

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
            text = " ".join([e.lemma_ for e in self.NLP(text)]).lower() # alguna palabra la pone mayúscula

        # Eliminar accentos?
        if self.CONFIG["remove_accents"]:
            rgx_c = r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+"
            text = normalize('NFC', re.sub(rgx_c, r"\1", normalize("NFD", text), 0, re.I))

        # Eliminar números?
        if self.CONFIG["remove_numbers"]:
            rgx_d = r"\s*\d+\s*"
            text = re.sub(rgx_d, ' ', text).strip()

        return text

    def __preprocess_text_stemming__(self, text):

        # Eliminar plurales?
        if self.CONFIG["remove_plurals"]:
            stemmer = PorterStemmer()
            text = " ".join([stemmer.stem(w) for w in text.split(" ")])

        # Stemming?
        if self.CONFIG["stemming"]:
            print("Actualizar código para otros idiomas");exit()
            stemmer = SnowballStemmer('spanish')
            text = " ".join([stemmer.stem(w) for w in text.split(" ")])
     
        return text

    def load_city(self, city):
        """Carga los datos de una ciudad, quedandose con las columnas relevantes"""

        # Cargar restaurantes
        res = pd.read_pickle(self.CONFIG["data_path"] + city + "/restaurants.pkl")

        # Cargar reviews
        rev = pd.read_pickle(self.CONFIG["data_path"] + city + "/reviews.pkl")
        rev = rev[['reviewId', 'userId', 'restaurantId', 'rating', 'date', 'language', 'text', 'title', 'url']]
        rev["city"] = city

        # Casting a int de algunas columnas
        res = res.astype({'id': 'int64'})
        rev = rev.astype({'reviewId': 'int64', 'restaurantId': 'int64', 'rating': 'int64'})

        # Concatenar restaurantes y reviews
        rev = rev.merge(res[["id", "name"]], left_on="restaurantId", right_on="id", how="left")
        rev = rev.drop(columns=["id"])

        # Eliminar reviews vacías (que tengan NAN, pero sigue habiendo reviews con texto=="")
        rev = rev.loc[(~rev["text"].isna()) & (~rev["title"].isna())]

        # Preprocesar las StopWords
        self.STOPWORDS = self.prerpocess_text(" ".join(self.STOPWORDS)).split(" ")

        # Preprocesar los textos
        mapply.init(
            n_workers=-1,
            chunk_size=100,
            max_chunks_per_worker=8,
            progressbar=True
        )

        # Primero preprocesamos el texto eliminando elementos básicos (a minusculas, quitar números)
        rev["text"] = rev["text"].mapply(self.__preprocess_text_base__)
        rev["title"] = rev["title"].mapply(self.__preprocess_text_base__)

        # Almancenamos una copia de lo anterior y hacemos stemming (si está activado)
        rev["text_base"] = rev["text"]
        rev["title_base"] = rev["title"]

        rev["text"] = rev["text"].mapply(self.__preprocess_text_stemming__)
        rev["title"] = rev["title"].mapply(self.__preprocess_text_stemming__)

        # Obtenemos un diccionario palabra_base -> stemming (para evitar error de Allocation hay que hacerlo así de lento)
                
        stemming_dict = pd.DataFrame(columns=["stemming", "real"])
        batches = np.array_split(rev, len(rev) // 10000)

        for b in tqdm(batches, desc="Stemming dict", total=len(batches)):
            base = np.concatenate((b.title_base+" "+b.text_base).str.split(" ").values)
            stmg = np.concatenate((b.title+" "+b.text).str.split(" ").values)
            stemming_dict = pd.concat([stemming_dict, pd.DataFrame(zip(stmg, base), columns=stemming_dict.columns)])
            stemming_dict = stemming_dict.drop_duplicates()
     
        stemming_dict = stemming_dict.sort_values("stemming").reset_index(drop=True)

        # Obtener número de palabras de las reviews y del título
        rev["n_words_text"] = rev["text"].apply(lambda x: 0 if len(x) == 0 else len(x.split(" ")))
        rev["n_words_title"] = rev["title"].apply(lambda x: 0 if len(x) == 0 else len(x.split(" ")))

        # Eliminar reviews de tamaño 0 en texto y titulo
        rev = rev.loc[(rev["n_words_text"] > 0) & (rev["n_words_title"] > 0)]

        # Obtener número de caracteres de las reviews y eliminar aquellas con más de 2000
        rev["n_char_text"] = rev["text"].apply(lambda x: len(x))
        rev = rev.loc[rev["n_char_text"] <= 2000]

        return rev, stemming_dict

    def __get_stopwords__(self, lang="es"):
        # nltk.download('stopwords')
        ret_stgrs = []

        if lang == "es":
            ret_stgrs = stopwords.words('spanish')
            ret_stgrs += ["gijon", "asturiano", "asturias"]
            ret_stgrs += ['ademas', 'alli', 'aqui', 'asturias', 'asi', 'aunque', 'cada', 'casa', 'casi',
                          'comido', 'comimos', 'cosas', 'creo', 'decir', 'despues', 'dos', 'dia', 'fin',
                          'hace', 'hacer', 'hora', 'ido', 'igual', 'ir', 'lado', 'luego', 'mas', 'merece',
                          'mismo', 'momento', 'mucha', 'muchas', 'parece', 'parte', 'pedimos', 'pedir', 'probar',
                          'puede', 'puedes', 'pues', 'punto', 'relacion', 'reservar', 'seguro', 'semana', 'ser',
                          'si',
                          'sido', 'siempre', 'sitio', 'sitios', 'solo', 'si', 'tan', 'tener', 'toda', 'tomar',
                          'tres',
                          'unas', 'varias', 'veces', 'ver', 'verdad', 'vez', 'visita', 'bastante', 'duda', 'gran',
                          'menos', 'no', 'nunca', 'opinion', 'primera', 'primero', 'segundo', 'mejor',
                          'mejores']
            ret_stgrs += ['alguna', 'caso', 'centro', 'cierto', 'comentario',
                          'cosa',
                          'cualquier', 'cuanto', 'cuenta', 'da', 'decidimos', 'demasiado', 'dentro', 'destacar',
                          'detalle',
                          'dia', 'dias', 'esperamos', 'esperar', 'general', 'gracias', 'haber', 'hacen', 'hecho',
                          'lleno',
                          'media', 'minutos', 'noche', 'nota', 'poder', 'ponen', 'probado', 'puedo', 'reserva',
                          'resto',
                          'sabor', 'solo', 'tiempo', 'todas', 'tomamos', 'totalmente', 'vamos', 'varios', 'vida',
                          'unico']
            ret_stgrs += ['ahora', 'aun', 'cerca', 'ciudad', 'cuatro', 'elegir', 'encima', 'falta', 'final',
                          'ganas',
                          'hoy', 'llegamos', 'medio', 'mundo', 'nuevo', 'ocasiones', 'opcion', 'parecio', 'pasar',
                          'pedido',
                          'pesar', 'poner', 'probamos', 'pronto', 'realmente', 'salimos', 'sirven', 'situado',
                          'tampoco',
                          'tarde', 'tipo', 'va', 'vas', 'voy']
            ret_stgrs += ['come', 'demas', 'ello', 'etc', 'incluso', 'llegar', 'pasado', 'primer', 'pusieron',
                          'quedamos', 'quieres', 'saludo', 'tambien', 'trabajo', 'tras', 'verano']
            ret_stgrs += ['algun', 'cenamos', 'comentarios', 'comiendo', 'dan', 'dice', 'domingo', 'ofrecen',
                          'razonable', 'tamaño']
            ret_stgrs += ['nadie', 'ningun', 'opiniones', 'quizas', 'san', 'sino']
            ret_stgrs += ['atendio', 'pega', 'sabado', 'dicho', 'par', 'total', 'años', 'año', 'ultima', 'comer']
            ret_stgrs += ['ahi', 'restaurante']
            ret_stgrs += ["claro", "dar", "dieron", "dijo", "entrar", "equipo", "establecimiento", "forma", "hacia", "ibamos", "local", "mayor", "mientras", "misma", "ninguna", "paso", "pedi", "pudimos", "pueden", "resumen", "seguir", "segunda", "siendo", "suele", "supuesto", "ultimo"]
            ret_stgrs += ["pagamos", "tal", "saber", "deja", "toque", "puesto"]
            ret_stgrs += ["segun", "iba", "manera", "arriba"]
            ret_stgrs += ["queda", "parecia", "imposible", "proxima"]

            # ToDo: Quitar "saludos", "lorenzo" en el futuro

        elif lang == "en":
            ret_stgrs = stopwords.words("english")

        elif lang == "fr":
            ret_stgrs = stopwords.words("french")

        return ret_stgrs
