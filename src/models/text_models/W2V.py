# -*- coding: utf-8 -*-
from src.Common import print_g
from src.models.ModelClass import ModelClass

import os
import gensim
from gensim.models import KeyedVectors


class W2V(ModelClass):
    """ Aprender embeddigs de palabras mediante W2V de Gensim """
    def __init__(self, config, dataset):
        self.MODEL_FILE_PATH = None
        ModelClass.__init__(self, config=config, dataset=dataset)

    def get_model(self):

        self.MODEL_FILE_PATH = self.MODEL_PATH+"word2vec.model"

        if not os.path.exists(self.MODEL_FILE_PATH):
            model = gensim.models.Word2Vec(self.DATASET.DATA[self.CONFIG["model"]["train_set"]],
                                           min_count=self.CONFIG["model"]["min_count"],  # ignoramos las palabras que aparezcan menos de MINC veces
                                           window=self.CONFIG["model"]["window"],  # máxima distancia entre la palabra actual y la predicha
                                           size=self.CONFIG["model"]["n_dimensions"],  # dimensión del espacio
                                           seed=self.CONFIG["model"]["seed"],
                                           workers=10)
        else:
            print_g("Loading existing model...")
            model = gensim.models.Word2Vec.load(self.MODEL_FILE_PATH)  # Word2Vec propio

        return model

    def train(self):

        if not os.path.exists(self.MODEL_FILE_PATH):
            print_g("Training model...")
            self.MODEL.train(self.DATASET.DATA[self.CONFIG["model"]["train_set"]], total_examples=self.MODEL.corpus_count, total_words=self.MODEL.corpus_total_words, epochs=100)
            self.MODEL.save(self.MODEL_FILE_PATH)
        else:
            print_g("Model already trained!")

    def test(self, word):
        print(self.MODEL.wv.most_similar(positive=word))
