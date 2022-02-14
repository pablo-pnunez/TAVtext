from contextlib import redirect_stderr
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import fr_core_news_sm as spacy_fr_model  # python -m spacy download fr_core_news_sm
from tqdm.contrib.concurrent import process_map


def find_features_in_reviews(rw, features, NLP):
    rw, rw_tx = rw
    rw_ftr = list(map(lambda x: x in features, rw_tx))
    rw_nlp = NLP(" ".join(rw_tx))
    rw_pos = list(map(lambda x: x.pos_, rw_nlp))
    
    rw_tx = np.array(rw_tx)[rw_ftr]
    rw_pos = np.array(rw_pos)[rw_ftr]

    return list(zip(rw_tx, rw_pos, [rw]*len(rw_tx)))


def features_pos(w, RVW, NLP, text_column, seed):
    # Para cada una de las palabras más frecuentes, se miran las reviews que la contienen
    # rvs_with_w = RVW.loc[RVW[text_column].apply(lambda x: w in x)]
    rvs_with_w = np.apply_along_axis(lambda x: w in x[0], 1, RVW)
    rvs_with_w = RVW[rvs_with_w]
    # Para evitar sobrecarga, nos quedamos con 20 reviews como máximo
    # rvs_with_w = rvs_with_w.sample(min(len(rvs_with_w), 20), random_state=seed)
    np.random.seed(seed)
    rvs_with_w = np.random.choice(rvs_with_w.flatten(), min(len(rvs_with_w), 20))
    # Buscamos la posición concreta de la palabra dentro de la review
    # rvs_with_w["w_loc"] = rvs_with_w[text_column].apply(lambda x: np.argwhere(np.asarray(x) == w).flatten()[0])
    w_loc = np.apply_along_axis(lambda x: np.argwhere(np.asarray(x[0]) == w).flatten()[0], 1, np.expand_dims(rvs_with_w, -1))
    # Obtenemos un vector de POS para cada palabra de las reviews y nos quedamos con el POS de la que nos interesa
    # rvs_with_w["w_pos"] = rvs_with_w.apply(lambda x: [n.pos_ for n in NLP(" ".join(x[text_column]))][x.w_loc],  1)
    w_pos = list(map(lambda x: NLP(" ".join(x[1]))[x[0]].pos_, zip(w_loc, rvs_with_w)))
    # Finalmente, como habrá varios POS, nos quedamos con el que aparezca en la mayoría de reviews
    # poses, p_counts = np.unique(rvs_with_w.w_pos, return_counts=True)
    poses, p_counts = np.unique(w_pos, return_counts=True)
    # Retornar resultado
    res = poses[p_counts.argmax()]

    print(w, res)
    return res


def la_pos(x):
    p, f = np.unique(x.id_pos, return_counts=True)
    return pd.Series({"col": p, "val": f})

NLP = spacy_fr_model.load(disable=["parser", "ner", "attribute_ruler"])

pos_values = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]

features = np.load("FEATURES_PRS.npy")  # Palabras sobre las que hacer POS
features = list(map(str, features))

RVW_CP = np.load("RVW_CP_PRS.npy", allow_pickle=True).flatten()  # Lista de reviews (lista de palabras)

batches = np.array_split(RVW_CP, 32)

# Crear estructuras de datos para almacenar los valores de POS
features_df = pd.DataFrame(zip(range(len(features)), features), columns=["id_feature", "feature"])
features_df = features_df.set_index("feature")
pos_df = pd.DataFrame(zip(range(len(pos_values)), pos_values), columns=["id_pos", "pos"])
pos_df = pos_df.set_index("pos")
mtrx = np.zeros((len(features), len(pos_values)), dtype=int)

for idx, b in enumerate(batches):
    print(f'{idx+1} de {len(batches)}')
    b = list(map(lambda x: " ".join(x), b))
    POS = list(NLP.pipe(b, n_process=8))  # 8 es lo mejor

    all_words = np.concatenate(list(map(lambda x: [(str(w), w.pos_) for w in x], POS)))
    all_df = pd.DataFrame(all_words, columns=["feature", "pos"])
    all_df = all_df.loc[all_df.feature.isin(features)].reset_index(drop=True)
    all_df = all_df.merge(features_df, on="feature").merge(pos_df, on="pos")
    all_df = all_df.groupby(["id_feature", "id_pos"]).apply(len).reset_index()

    mtrx[all_df.id_feature, all_df.id_pos] += all_df[0]

    del all_df, b, POS


features_pos = np.asarray(pos_values)[np.apply_along_axis(np.argmax, 1, mtrx)]
res = pd.DataFrame(zip(features, features_pos), columns=["feature", "pos"])

np.save("res", res)

