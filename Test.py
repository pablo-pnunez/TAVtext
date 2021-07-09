import es_core_news_sm as spacy_es_model
from hunspell import HunSpell
import pandas as pd
import numpy as np
from tqdm import tqdm
from pattern.es import suggest, singularize, lemma, lexeme

data = pd.read_pickle("data/W2Vdataset/b6fbb3a4ba31fa0d1bd7128ff7e74869/ALL_TEXTS")

nlp = spacy_es_model.load(disable=["parser", "ner", "attribute_ruler"])

diccionario = HunSpell('/usr/share/hunspell/es_ES.dic', '/usr/share/hunspell/es_ES.aff')

bad_spell = set()
rev_err_pct = []

for row in tqdm(data):
    rev_nlp = nlp(" ".join(row))
    rev_pos = np.asarray(list(map(lambda x: x.pos_, rev_nlp)))
    rev_lemm = " ".join(list(map(lambda x: x.lemma_, rev_nlp)))

    try:
        good_spelling = np.asarray(list(map(diccionario.spell, row)))
        bad_spelling = np.argwhere(good_spelling == False).flatten()
        bad_spelling = bad_spelling[np.argwhere(rev_pos[bad_spelling] != 'NOUN').flatten()]

        rev_err_pct.append(len(bad_spelling)/len(row))
        
        bad_spelling = np.asarray(row)[bad_spelling]
        bad_spell.update(bad_spelling)
    except:
        rev_err_pct.append(1.0)


print(bad_spell)