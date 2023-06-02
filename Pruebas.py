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
from src.models.text_models.att.BERTATT2VAL import BERTATT2VAL
from src.models.text_models.ATT2ITM import ATT2ITM
from src.models.text_models.att.tf_BERTATT2VAL import tf_BERTATT2VAL
from src.models.text_models.att.BERTATT2ITM import BERTATT2ITM

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
        #all_dots = tf.matmul(self.model.get_layer("all_words").weights[0], self.model.get_layer("all_items").weights[0], transpose_b=True).numpy()
        #print(all_dots.min(), all_dots.max())
        message = f"{self.subset} {epoch+1:03d} TR: {logs['NDCG@10']:.3f} VAL: {logs['val_NDCG@10']:.3f}"
        self.send(message)


dataset = "restaurants".lower().replace(" ", "") if args.dst is None else args.dst
subset = "gijon".lower().replace(" ", "") if args.sst is None else args.sst

model = "BERTATT2ITM"
model_v = "0" if args.mv is None else args.mv

seed = 100 if args.sd is None else args.sd
l_rate = 1e-4 if args.lr is None else args.lr
n_epochs = 1000 if args.ep is None else args.eps
b_size = 256 if args.bs is None else args.bs
early_stop_patience = 10 if args.esp is None else args.esp

min_reviews_rst = 100
min_reviews_usr = 1
bow_pct_words = 10 if args.bownws is None else args.bownws
language = "es" if subset in ["gijon", "madrid", "barcelona"] else "fr" if subset in ["paris"] else "en"

remove_stopwords = 2  # 0, 1 o 2 (No quitar, quitar manual, quitar automático)
lemmatization = True
remove_accents = True
remove_numbers = True
truncate_padding = True

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

mdl_cfg = {"model": {"model_version": model_v, "learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": n_epochs, "batch_size": b_size, "seed": seed,
                        "early_st_first_epoch": 0, "early_st_monitor": "val_loss", "early_st_monitor_mode": "min", "early_st_patience": early_stop_patience},
            "session": {"gpu": gpu, "mixed_precision": True, "in_md5": False}}


telegram_callback = TelegramCallback(subset=subset, token="kP3QuLoUv0b_yZ28rK3GNMesAW12e9KiEAA:0245328206", chat_id=717499654)

if "ATT2VAL" == model: mdl = ATT2VAL(mdl_cfg, text_dataset)
elif "BERTATT2VAL" == model: mdl = BERTATT2VAL(mdl_cfg, text_dataset)
elif "ATT2ITM" == model: mdl = ATT2ITM(mdl_cfg, text_dataset)
elif "BERTATT2ITM" == model: mdl = BERTATT2ITM(mdl_cfg, text_dataset)
elif "tf_BERTATT2VAL" == model: mdl = tf_BERTATT2VAL(mdl_cfg, text_dataset)
else: raise NotImplementedError

mdl.train(dev=True, save_model=True, callbacks=[telegram_callback])
mdl.emb_tsne()

if language == "es": 
    mdl.evaluate_text("quiero comer un arroz con bogavante y con buenas vistas")
    mdl.evaluate_text("a el la yo en un con y") # HAY QUE USAR UNA RELU o RELUTAN SI NO ESTO DA VALORES ALTOS
if language == "en": mdl.evaluate_text("Where can not i eat the typical pastrami sandwich")





