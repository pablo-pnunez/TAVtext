from lime.lime_text import LimeTextExplainer

from src.datasets.text_datasets.RestaurantDataset import RestaurantDataset
from src.datasets.text_datasets.AmazonDataset import AmazonDataset
from src.datasets.text_datasets.POIDataset import POIDataset

from models.text_models.att.ATT2ITM import ATT2ITM
from src.models.text_models.BOW2ITM import BOW2ITM
from src.models.text_models.USEM2ITM import USEM2ITM

import os
import re
import json
import nvgpu
import imgkit
import numpy as np
import pandas as pd
import tensorflow as tf
from bs4 import BeautifulSoup

'''
 ¿Qué palabras de una reseña son las más importantes para cada modelo?
 ¿Qué puntuación se le da a cada una de esas palabras en el restaurante donde fueron escritas (no donde se predice, el restaurante real)?

'''


def plot_task1(data, file_name="out_tk1"):

    def text_prepro(text, numbers=False):
        # Eliminar los \n \t....
        text = re.sub(r"(\\\w)+", ' ', text).strip()
        # Remplazar las comas mal puestas
        text = re.sub(r"(\w)\s*([\.\,\;\:\!\?\\\-])+\s*(\w)", "\\1\\2 \\3", text)
        # Eliminar números
        if not numbers:
            text = re.sub(r"\d+[\.\,\:\/\\\-]*\s*", "", text)
        # Eliminar contracciones
        contracts = {"'aight": "alright",
                     "ain't": "am not",
                     "amn't": "am not",
                     "aren't": "are not",
                     "can't": "can not",
                     "'cause": "because",
                     "could've": "could have",
                     "couldn't": "could not",
                     "couldn't've": "could not have",
                     "daren't": "dare not",
                     "daresn't": "dare not",
                     "dasn't": "dare not",
                     "didn't": "did not",
                     "doesn't": "does not",
                     "don't": "do not",
                     "d'ye": "do you",
                     "e'er": "ever",
                     "everybody's": "everybody is",
                     "everyone's": "everyone is",
                     "finna": "going to",
                     "g'day": "good day",
                     "gimme": "give me",
                     "giv'n": "given",
                     "gonna": "going to",
                     "gon't": "go not",
                     "gotta": "got to",
                     "hadn't": "had not",
                     "had've": "had have",
                     "hasn't": "has not",
                     "haven't": "have not",
                     "he'd": "he had",
                     "he'dn't've'd": "he would not have had",
                     "he'll": "he will",
                     "he's": "he is",
                     "he've": "he have",
                     "how'd": "how did",
                     "howdy": "how do you do",
                     "how'll": "how will",
                     "how're": "how are",
                     "how's": "how is",
                     "i'd": "i had",
                     "i'd've": "i would have",
                     "i'll": "i will",
                     "i'm": "i am",
                     "i'm'a": "i am about to",
                     "i'm'o": "i am going to",
                     "innit": "is it not",
                     "i've": "i have",
                     "isn't": "is not",
                     "it'd": "it had",
                     "it'll": "it will",
                     "it's": "it is",
                     "let's": "let us",
                     "ma'am": "madam",
                     "mayn't": "may not",
                     "may've": "may have",
                     "methinks": "me thinks",
                     "mightn't": "might not",
                     "might've": "might have",
                     "mustn't": "must not",
                     "mustn't've": "must not have",
                     "must've": "must have",
                     "needn't": "need not",
                     "ne'er": "never",
                     "o'clock": "of the clock",
                     "o'er": "over",
                     "ol'": "old",
                     "oughtn't": "ought not",
                     "shalln't": "shall not",
                     "shan't": "shall not",
                     "she'd": "she had",
                     "she'll": "she will",
                     "she's": "she is",
                     "should've": "should have",
                     "shouldn't": "should not",
                     "shouldn't've": "should not have",
                     "somebody's": "somebody is",
                     "someone's": "someone is",
                     "something's": "something is",
                     "so're": "so are",
                     "that'll": "that will",
                     "that're": "that are",
                     "that's": "that is",
                     "that'd": "that had",
                     "there'd": "there had",
                     "there'll": "there will",
                     "there're": "there are",
                     "there's": "there is",
                     "these're": "these are",
                     "these've": "these have",
                     "they'd": "they had",
                     "they'll": "they will",
                     "they're": "they are",
                     "they've": "they have",
                     "this's": "this is",
                     "those're": "those are",
                     "those've": "those have",
                     "'tis": "it is",
                     "to've": "to have",
                     "'twas": "it was",
                     "wanna": "want to",
                     "wasn't": "was not",
                     "we'd": "we had",
                     "we'd've": "we would have",
                     "we'll": "we will",
                     "we're": "we are",
                     "we've": "we have",
                     "weren't": "were not",
                     "what'd": "what did",
                     "what'll": "what will",
                     "what're": "what are",
                     "what's": "what is",
                     "what've": "what have",
                     "when's": "when is",
                     "where'd": "where did",
                     "where'll": "where will",
                     "where're": "where are",
                     "where's": "where is",
                     "where've": "where have",
                     "which'd": "which had",
                     "which'll": "which will",
                     "which're": "which are",
                     "which's": "which is",
                     "which've": "which have",
                     "who'd": "who did",
                     "who'd've": "who would have",
                     "who'll": "who will",
                     "who're": "who are",
                     "who's": "who is",
                     "who've": "who have",
                     "why'd": "why did",
                     "why're": "why are",
                     "why's": "why is",
                     "won't": "will not",
                     "would've": "would have",
                     "wouldn't": "would not",
                     "wouldn't've": "would not have",
                     "y'all": "you all",
                     "y'all'd've": "you all would have",
                     "y'all'dn't've'd": "you all would not have had",
                     "y'all're": "you all are",
                     "you'd": "you had",
                     "you'll": "you will",
                     "you're": "you are",
                     "you've": "you have",}
        text = " ".join([contracts[w.lower()] if w.lower() in contracts else w for w in text.split(" ")])
        
        # Para la tercera persona del singular
        text = text.replace("'s", " 's")  

        # Para emojis y otras cosas extrañas
        text = re.sub(r"\s+[\-|\:|\)|\)|\[|\]\<|\>\{|\}]+", "", text)

        return text

    def dict_creation(text_a, text_b): 
        text_a = re.sub(r" {2,}", " ", text_a)  # Eliminar dos espacios juntos
        text_b = text_prepro(text_b)

        txt_wrds = text_a.split(" ")
        raw_txt_wrds = text_b.split(" ")

        res_dict = {}

        if len(txt_wrds) != len(raw_txt_wrds):
            raise ValueError
        else:
            res_dict = dict(zip(raw_txt_wrds, txt_wrds))

        return res_dict

    with open("explanation/template_tk1.html", "r", encoding="utf-8") as f: html_file = f.read()

    soup = BeautifulSoup(html_file, 'html.parser')

    # Creamos un diccionario palabra_preprocesada: palabra_raw
    # Para que coincidan los tamaños, hay que eliminar elementos del texto raw (como números)
    word_dict = dict_creation(data["text"], data["raw_text"])

    # Aquí si incluimos los números, es el texto que vamos a imprimir
    review_text = text_prepro(data["raw_text"], numbers=True).split(" ")

    review_items = soup.select('div.system')

    for sys_idx, sys_html in enumerate(review_items):
        sysdata = data["data"][sys_idx]["values"]
        sys_relevant_words = list(sysdata.keys())

        final_hmtl = []
        for word in review_text:
            wrd_html = word
            if word in word_dict.keys() and word_dict[word] in sys_relevant_words: wrd_html = f"<hl>{word}</hl>"
            final_hmtl.append(wrd_html)
        
        sys_html.append(BeautifulSoup(" ".join(final_hmtl), features="lxml"))

    options = {'enable-javascript': ''}

    os.makedirs("explanation/tk1", exist_ok=True)

    with open(f"explanation/tk1/{file_name}.html", "w") as f: f.write(str(soup))

    with open(f"explanation/tk1/{file_name}.txt", "w") as f: f.write(str(data))

    imgkit.from_string(str(soup), f"explanation/tk1/{file_name}.svg", options=options)


def plot_task2(data):

    with open("explanation/template_tk2.html", "r", encoding="utf-8") as f: html_file = f.read()

    soup = BeautifulSoup(html_file, 'html.parser')

    charts = soup.select('div.chart table')

    review_item = soup.select_one('div.head')
    review_item.append(data["text"])

    for cidx, c in enumerate(charts):

        data_info = data["data"][cidx]
        max_val = data_info["max"]
        min_val = data_info["min"]

        values = data_info["values"]

        for word in values:
            val = values[word]

            abs_max_val = max(abs(min_val), max_val)
            val_pct = (abs(val) / abs_max_val) * 100
            # val_pct = ((val - min_val) / (max_val - min_val)) * 100 # Normalizar entre 0 y 1

            opacity = 1  # max(.1, val_pct / 100)
            new_div = soup.new_tag("tr")

            neg = soup.new_tag("td")
            text = soup.new_tag("td")
            pos = soup.new_tag("td")

            text.string = word
            text['class'] = "text"

            if val < 0:
                bar = soup.new_tag("div", attrs={"class": "tbar neg"})
                bar['style'] = f"left: {100-val_pct+.65}%; width: {val_pct}%; opacity: {opacity}"
                neg.append(bar)
            else:
                bar = soup.new_tag("div", attrs={"class": "tbar pos"})
                bar['style'] = f"width: {val_pct}%; opacity: {opacity}"
                pos.append(bar)

            new_div.append(neg)
            new_div.append(text)
            new_div.append(pos)

            c.append(new_div)

    options = {'enable-javascript': ''}

    with open("explanation/out_tk2.html", "w") as f: f.write(str(soup))

    imgkit.from_string(str(soup), "explanation/out_tk2.svg", options=options)


gpu = int(np.argmin(list(map(lambda x: x["mem_used_percent"], nvgpu.gpu_info()))))

experiment_data = pd.read_csv("explanation/best_models.csv")
experiment_data = experiment_data.loc[(experiment_data.dataset == "restaurants") & (experiment_data.subset == "gijon")]

for (dataset, subset), data in experiment_data.groupby(["dataset", "subset"]):
    # Inicialmente se carga el dataset del primer modelo (por ejemplo, son todos iguales dentro del mismo dataset/subset)
    model_path = f"models/ATT2ITM/{dataset}/{subset}/{data['md5'].values[0]}"
    with open(f'{model_path}/cfg.json') as f: model_config = json.load(f)
    dts_cfg = model_config["dataset_config"]

    # Cargar el dataset pertinente
    if dataset == "restaurants": text_dataset = RestaurantDataset(dts_cfg)
    elif dataset == "pois": text_dataset = POIDataset(dts_cfg)
    elif dataset == "amazon": text_dataset = AmazonDataset(dts_cfg)
    else: raise ValueError

    # Seleccionar un caso de TEST aleatorio sobre el que trabajar (menor de 30 palabras??)
    test_real_sample = pd.read_pickle(text_dataset.DATASET_PATH + "ALL_DATA")
    test_real_sample = test_real_sample[(test_real_sample.test == 1) & (test_real_sample.n_words_text <= 50)]
    
    # Para cada ejemplo de test
    for _, test_sample in test_real_sample.iterrows():

        # Obtener el texto real de la reseña
        raw_data = text_dataset.load_subset(subset)
        review_text = raw_data.loc[raw_data.reviewId == test_sample.reviewId]["text"].values[0]
        del raw_data

        model_outputs = []

        # Obtener la predicción de cada modelo
        for _, model_experiment_data in data.iterrows():
            model = model_experiment_data["model"]
            md5 = model_experiment_data["md5"]

            # Cargar la configuración para los mejores modelos
            model_path = f"models/{model}/{dataset}/{subset}/{md5}"
            with open(f'{model_path}/cfg.json') as f: model_config = json.load(f)
            mdl_cfg = {"model": model_config["model"], "session": {"gpu": gpu, "mixed_precision": True, "in_md5": False}}

            # Cargar el modelo pertinente
            if "ATT2ITM" == model: model_class = ATT2ITM(mdl_cfg, text_dataset)
            elif "BOW2ITM" == model: model_class = BOW2ITM(mdl_cfg, text_dataset)
            elif "USEM2ITM" == model: model_class = USEM2ITM(mdl_cfg, text_dataset)
            else: raise NotImplementedError
            model_class.train(dev=True, save_model=True)

            # Obtener las palabras y notas del modelo
            exp_att = model_class.explain_test_sample(test_sample)
            model_outputs.append(exp_att)

        plot_dict = {"raw_text": review_text, "text": test_sample.text, "data": {0: model_outputs[0], 1: model_outputs[1], 2: model_outputs[2]}}
        plot_task1(plot_dict, file_name=f"{test_sample.reviewId}")
      
exit()
text_instance = test_real_sample.text_source
text_instance = re.sub(r"\s+", " ", text_instance, 0, re.MULTILINE)
# Codificar el texto en los formatos adecuados (al ser un ejemplo del dataset, ya está previamente calculado)
bow_encoding = text_dataset.DATA["BOW_SEQUENCES"][test_real_sample["bow"]].todense()
seq_encoding = text_dataset.DATA["TEXT_SEQUENCES"][test_real_sample["seq"]]
seq_encoding_nopad = seq_encoding[seq_encoding > 0]

# Datos del rastaurante real
restaurant_name = test_real_sample["name"]
restaurant_id = classes.loc[classes.name == restaurant_name].index[0]

# Predecir palabras y puntuación para el restaurante real
all_att, _, word_names, _, _ = model_class.get_item_word_att(items = np.array([restaurant_id]))
review_words = np.asarray(word_names)[seq_encoding_nopad]
review_words_val = all_att[seq_encoding_nopad]



preds_rst = model_class.MODEL.predict([seq_encoding[None, :], np.arange(model_class.DATASET.DATA["N_ITEMS"])[None, :]], verbose=0)


# Lime Explainer
explainer = LimeTextExplainer(class_names=class_names)


# For the multiclass case, we have to determine for which labels we will get explanations, via the 'labels' parameter.
# Below, we generate explanations for labels 0 and 17


def classifier_fn(d):
    # takes a list of d strings and outputs a (d, k) numpy array with prediction probabilities, where k is the number of classes. For ScikitClassifiers , this is classifier.predict_proba.
    ret = []
    for text in d:
        text = re.sub(r"\s+", " ", text, 0, re.MULTILINE)
        if len(text.strip()) > 0:
            text_prepro = model_class.DATASET.prerpocess_text(text)
            lstm_text = model_class.DATASET.DATA["TEXT_TOKENIZER"].texts_to_sequences([text_prepro])
            # lstm_text = [list(map(lambda x: model_class.DATASET.DATA["TEXT_TOKENIZER"].word_index[x], text_prepro.split(" ")))]
            lstm_text_pad = tf.keras.preprocessing.sequence.pad_sequences(lstm_text, maxlen=model_class.DATASET.DATA["MAX_LEN_PADDING"])
            preds_rst = model_class.MODEL.predict([lstm_text_pad, np.arange(model_class.DATASET.DATA["N_ITEMS"])[None, :]], verbose=0)
            ret.append(preds_rst.flatten())
        else:
            ret.append(np.zeros(len(classes)))

    return np.row_stack(ret)


def classifier_fn2(d):
    # takes a list of d strings and outputs a (d, k) numpy array with prediction probabilities, where k is the number of classes. For ScikitClassifiers , this is classifier.predict_proba.
    ret = []
    for text in d:
        text = re.sub(r"\s+", " ", text, 0, re.MULTILINE)
        if len(text.strip()) > 0:
            preds_rst = model_class.MODEL.predict([text], verbose=0)
            ret.append(preds_rst.flatten())
        else:
            ret.append(np.zeros(len(classes)))

    return np.row_stack(ret)


np.random.seed(None)
test_real_sample = pd.read_pickle(text_dataset.DATASET_PATH + "ALL_DATA")
test_real_sample = test_real_sample[(test_real_sample.test == 1)&(test_real_sample.n_words_text<=30)]
test_real_sample = test_real_sample.iloc[np.random.randint(0, len(test_real_sample))]
text_instance = test_real_sample.text_source
text_instance = re.sub(r"\s+", " ", text_instance, 0, re.MULTILINE)

print("[RESTAURANTE REAL]: ", test_real_sample["name"])

model_class.evaluate_text(text_instance)

if "USEM2ITM" == model:
    classifier_fn = classifier_fn2

items = classifier_fn([text_instance]).flatten().argsort()[-1:-5:-1]  # Items recomendados
exp = explainer.explain_instance(text_instance, classifier_fn, num_features=test_real_sample.n_words_text, labels=items)

for itm in items:
    print('Explanation for class \033[1m %s \033[0m \n' % class_names[itm])
    print('\n'.join(map(str, exp.as_list(label=itm))))
    print("-" * 50)
