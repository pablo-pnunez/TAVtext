# -*- coding: utf-8 -*-
from src.Common import parse_cmd_args, print_b, print_e
import pandas as pd
import numpy as np
import nvgpu
import json
import os

args = parse_cmd_args()
gpu = np.argmin([g["mem_used_percent"] for g in nvgpu.gpu_info()]) if args.gpu is None else args.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

from src.datasets.text_datasets.RestaurantDataset import RestaurantDataset
from src.datasets.text_datasets.AmazonDataset import AmazonDataset
from src.datasets.text_datasets.POIDataset import POIDataset

from src.models.text_models.att.ATT2VAL import ATT2VAL
from src.models.text_models.att.bert.BERTATT2VAL import BERTATT2VAL
from src.models.text_models.att.ATT2ITM import ATT2ITM
from src.models.text_models.att.bert.tf_BERTATT2VAL import tf_BERTATT2VAL
from src.models.text_models.att.bert.BERTATT2ITM import BERTATT2ITM

from src.models.text_models.att.semanticsim.SSATT2VAL import SSATT2VAL
from src.models.text_models.att.semanticsim.SSATT2ITM import SSATT2ITM

from src.models.text_models.att.w2v.W2VATT2VAL import W2VATT2VAL
from src.models.text_models.att.w2v.W2VATT2ITM import W2VATT2ITM

from src.models.text_models.att.WATT2VAL import WATT2VAL

import tensorflow as tf
import urllib.parse
import requests

# #######################################################################################################################


class TelegramCallback(tf.keras.callbacks.Callback):
    def __init__(self, subset, token, chat_id):
        super(TelegramCallback, self).__init__()
        self.subset = subset
        self.token = token[::-1]
        self.chat_id = chat_id

    def send(self, text):
        text = urllib.parse.quote(text)
        requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage?chat_id={self.chat_id}&text={text}")

    def on_epoch_end(self, epoch, logs=None):
        
        '''
        input_w = np.array([[0,1,1]])
        input_i = np.array([list(range(149))])
        model = tf.keras.Model(inputs=self.model.inputs, outputs=[self.model.get_layer("dotprod").output, self.model.get_layer("tf.tile_1").output])
        preds = model.predict([input_w, input_i], verbose=0)
        print(tf.math.reduce_sum(preds[0], 1) / tf.math.reduce_sum(preds[1], 1))
        '''
        message = f"{self.subset} {epoch+1:03d} TR: {logs['NDCG@10']:.3f} VAL: {logs['val_NDCG@10']:.3f}"
        self.send(message)


dataset = "restaurants".lower().replace(" ", "") if args.dst is None else args.dst
subset = "gijon".lower().replace(" ", "") if args.sst is None else args.sst

seed = 100 if args.sd is None else args.sd

min_reviews_rst = 100
min_reviews_usr = 1
bow_pct_words = 10 if args.bownws is None else args.bownws
language = "es" if subset in ["gijon", "madrid", "barcelona"] else "fr" if subset in ["paris"] else "en"

remove_stopwords = 2  # 2 # 0, 1 o 2 (No quitar, quitar manual, quitar automático)
remove_accents = True
remove_numbers = True
truncate_padding = True
lemmatization = True

'''
print_e("OJO: NO HAY LEMATIZACIÓN")
lemmatization = False
remove_stopwords = 0  # 2 # 0, 1 o 2 (No quitar, quitar manual, quitar automático)
'''

if dataset == "restaurants":
    base_path = "/media/nas/datasets/tripadvisor/restaurants/"
elif dataset == "pois":
    base_path = "/media/nas/datasets/tripadvisor/pois/"
    language = "es"  # Están todas en español
elif dataset == "amazon":
    base_path = "/media/nas/datasets/amazon/"

# DATASET CONFIG #######################################################################################################

dts_cfg = {"dataset": dataset, "subset": subset, "language": language, "seed": seed, "data_path": base_path, "save_path": "data/",  # base_path + "Datasets/",
            "remove_stopwords": remove_stopwords, "remove_accents": remove_accents, "remove_numbers": remove_numbers,
            "lemmatization": lemmatization,
            "min_reviews_rst": min_reviews_rst, "min_reviews_usr": min_reviews_usr,
            "min_df": 5, "bow_pct_words": bow_pct_words, "presencia": False, "text_column": "text",  # BOW
            "n_max_words": -50, "test_dev_split": .1, "truncate_padding": truncate_padding}

if dataset == "restaurants": text_dataset = RestaurantDataset(dts_cfg)
elif dataset == "pois": text_dataset = POIDataset(dts_cfg)
elif dataset == "amazon": text_dataset = AmazonDataset(dts_cfg)
else: raise ValueError

model = "SSATT2ITM"
model_v = "0" if args.mv is None else args.mv

l_rate = 5e-6 if args.lr is None else args.lr
n_epochs = 1000 if args.ep is None else args.eps
b_size = 128 if args.bs is None else args.bs
early_stop_patience = 10 if args.esp is None else args.esp

mdl_cfg = {"model": {"model_version": model_v, "learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": n_epochs, "batch_size": b_size, "seed": seed,
                        "early_st_first_epoch": 0, "early_st_monitor": "val_loss", "early_st_monitor_mode": "min", "early_st_patience": early_stop_patience},
            "session": {"gpu": gpu, "mixed_precision": True, "in_md5": False}}


telegram_callback = TelegramCallback(subset=subset, token="kP3QuLoUv0b_yZ28rK3GNMesAW12e9KiEAA:0245328206", chat_id=717499654)

if "ATT2VAL" == model: mdl = ATT2VAL(mdl_cfg, text_dataset)
elif "BERTATT2VAL" == model: mdl = BERTATT2VAL(mdl_cfg, text_dataset)
elif "ATT2ITM" == model: mdl = ATT2ITM(mdl_cfg, text_dataset)
elif "BERTATT2ITM" == model: mdl = BERTATT2ITM(mdl_cfg, text_dataset)
elif "tf_BERTATT2VAL" == model: mdl = tf_BERTATT2VAL(mdl_cfg, text_dataset)
elif "SSATT2VAL" == model: mdl = SSATT2VAL(mdl_cfg, text_dataset)
elif "SSATT2ITM" == model: mdl = SSATT2ITM(mdl_cfg, text_dataset)

elif "W2VATT2VAL" == model: mdl = W2VATT2VAL(mdl_cfg, text_dataset)
elif "W2VATT2ITM" == model: mdl = W2VATT2ITM(mdl_cfg, text_dataset)

elif "WATT2VAL" == model: mdl = WATT2VAL(mdl_cfg, text_dataset)
else: raise ValueError


mdl.train(dev=True, save_model=True, callbacks=[telegram_callback])

"""
train_dev_users = mdl.DATASET.DATA["TRAIN_DEV"].userId.unique()
mdl.DATASET.DATA["TEST"] = mdl.DATASET.DATA["TEST"][mdl.DATASET.DATA["TEST"]["userId"].isin(train_dev_users)]
mdl.DATASET.DATA["TEST"] = mdl.DATASET.DATA["TEST"].drop_duplicates(subset=["userId", "id_item"], keep='last', inplace=False)
mdl.evaluate(test=True)
"""

mdl.emb_tsne()
exit()

if language == "es": 
    mdl.evaluate_text("a el la yo en un con y") # HAY QUE USAR UNA RELU o RELUTAN SI NO ESTO DA VALORES ALTOS
    mdl.evaluate_text("quiero un con y con bogavante buenas arroz vistas comer")
    mdl.evaluate_text("quiero comer un arroz con bogavante y con buenas vistas")
    mdl.evaluate_text("quiero comer un arroz con bogavante y con malas vistas")
    # mdl.evaluate_text("quiero arroz y quiero marisco")
    # mdl.evaluate_text("quiero arroz y no quiero marisco")

if language == "en":
    mdl.evaluate_text("he she it am are they")
    mdl.evaluate_text("Where can not i eat the typical pastrami sandwich")

# TODO: Parece que la lemmatización depende mucho de la posición de la palabra. 
# Por ejemplo quiero puede transformarse en "querer" o "quiero". 

# TODO: Semantic similarity embeddings parece que si dan uno por cada palabra
# TODO: food2vec?
# TODO: No poner que word2vec no va y ya, añadir resultado.



