# -*- coding: utf-8 -*-

from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec

import pandas as pd
import numpy as np
import os, re

from src.models.text_models.att.ATT2VAL import ATT2VAL


class TrainingCallback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.loss_previous_step = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print(f"Epoch {self.epoch:03d} => Loss:", loss-self.loss_previous_step)
        self.epoch+=1
        self.loss_previous_step = loss

class W2VATT(ATT2VAL):
    """ Predecir, a partir de los embeddings W2V de una review y los de los items, el restaurante de la review """

    def __init__(self, config, dataset):
        ATT2VAL.__init__(self, config=config, dataset=dataset)

    def get_w2v_embeddings(self, emb_size):
        w2v_embeddings_path = self.DATASET.DATASET_PATH+f"W2V_{emb_size}.npy"

        if os.path.exists(w2v_embeddings_path):
            w2v_embeddings = np.load(w2v_embeddings_path)
        else:
            all_data = pd.read_pickle(self.DATASET.DATASET_PATH+"ALL_DATA")[["reviewId", "text"]]
            text_sequences = [re.split(r'[\s|_]', sentence) for sentence in all_data.text.values] 
            w2v_model = Word2Vec(sentences=text_sequences, vector_size=emb_size, min_count=0, epochs=100, window=5, 
                                callbacks=[TrainingCallback()], compute_loss=True, workers=20, seed=self.CONFIG["model"]["seed"])

            # Los index del word2vec son diferentes a los del tokenizador de keras
            # Hay que obtener una matriz con los embeddings del w2v en orden keras y con el padding
            new_order = [w2v_model.wv.key_to_index[wrd] for wrd in list(self.DATASET.DATA["TEXT_TOKENIZER"].word_index.keys())]
            assert len(new_order) == len(self.DATASET.DATA["TEXT_TOKENIZER"].index_word)
            w2v_embeddings = w2v_model.wv.vectors[new_order,:]
            w2v_embeddings = np.vstack((np.zeros((1, w2v_embeddings.shape[1])), w2v_embeddings))
            np.save(w2v_embeddings_path, w2v_embeddings)
        
        return w2v_embeddings
