# -*- coding: utf-8 -*-

from src.Common import parse_cmd_args

from src.datasets.text_datasets.W2Vdataset import W2Vdataset
from src.datasets.text_datasets.RSTVALdataset import RSTVALdataset
from src.datasets.text_datasets.LSTM2VALdataset import LSTM2VALdataset
from src.datasets.text_datasets.BOW2VALdataset import BOW2VALdataset
from src.datasets.text_datasets.BOW2RSTdataset import BOW2RSTdataset
from src.datasets.text_datasets.LSTMBOW2RSTVALdataset import LSTMBOW2RSTVALdataset

from src.models.text_models.W2V import W2V
from src.models.text_models.BOW2VAL import BOW2VAL
from src.models.text_models.LSTM2VAL import LSTM2VAL

from src.models.text_models.LSTM2RST import LSTM2RST
from src.models.text_models.LSTMBOW2RSTVAL import LSTMFBOW2RSTVAL
from src.models.text_models.LSTMBOW2RSTVAL import LSTMBOW2RSTVAL

import json
import nvgpu
import numpy as np

########################################################################################################################

args = parse_cmd_args()

city = "gijon".lower().replace(" ", "") if args.ct is None else args.ct

stage = 1 if args.stg is None else args.stg
model_v = "1" if args.mv is None else args.mv

gpu = int(np.argmin(list(map(lambda x: x["mem_used_percent"], nvgpu.gpu_info()))))
seed = 100 if args.sd is None else args.sd
l_rate = 1e-3 if args.lr is None else args.lr
n_epochs = 1000 if args.ep is None else args.ep
b_size = 1024 if args.bs is None else args.bs

min_reviews_rst = 100
min_reviews_usr = 1
bow_n_words = 300 if args.bownws is None else args.bownws
w2v_dimen = 300 

stemming = False
remove_plurals = True
remove_accents = True
remove_numbers = True

base_path = "/media/nas/pperez/data/TripAdvisor/"


# W2V ##################################################################################################################

w2v_dts = W2Vdataset({"cities": ["gijon", "barcelona", "madrid"], "city": "multi", "seed": seed, "data_path": base_path, "save_path": base_path + "Datasets/",
                      "remove_plurals": remove_plurals, "stemming": stemming, "remove_accents": remove_accents, "remove_numbers": remove_numbers})

w2v_mdl = W2V({"model": {"train_set": "ALL_TEXTS", "min_count": 100, "window": 5, "n_dimensions": w2v_dimen, "seed": seed},
               "session": {"gpu": gpu, "in_md5": False}}, w2v_dts)

w2v_mdl.train()

# DATASET CONFIG #######################################################################################################

dts_cfg = {"city": city, "seed": seed, "data_path": base_path, "save_path": base_path + "Datasets/", 
                    "remove_plurals": remove_plurals, "stemming": stemming, "remove_accents": remove_accents, "remove_numbers": remove_numbers,
                    "min_reviews_rst": min_reviews_rst, "min_reviews_usr": min_reviews_usr,
                    "min_df": 5, "num_palabras": bow_n_words, "presencia": False, "text_column": "text",  # BOW
                    "n_max_words": 0, "test_dev_split": .1, "truncate_padding": True}  # LSTM

rstval = RSTVALdataset(dts_cfg)

# MODELO 1: LSTM2VAL ###################################################################################################
'''
lstm2val_mdl_cfg = {"model": {"model_version":model_v, "learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": n_epochs, "batch_size": b_size, "seed": seed,
                            "early_st_first_epoch": 0, "early_st_monitor": "val_mean_absolute_error", "early_st_monitor_mode": "min", "early_st_patience": 20},
                    "session": {"gpu": gpu, "in_md5": False}}

if stage == 0:
    lstm2val_mdl = LSTM2VAL(lstm2val_mdl_cfg, rstval, w2v_mdl)
    lstm2val_mdl.train(dev=True, save_model=True)
    lstm2val_mdl.baseline()
    lstm2val_mdl.evaluate(test=False)

if stage == 1:
    # Sobreescribir la configuración por la mejor conocida: 
    with open('models/LSTM2VAL/gijon/648042d8166ce7a02b24f9991b2778ce/cfg.json') as f: best_cfg_data = json.load(f)
    dts_cfg = best_cfg_data["dataset_config"]
    rstval = RSTVALdataset(dts_cfg)
    lstm2val_mdl_cfg["model"] = best_cfg_data["model"]
    lstm2val_mdl = LSTM2VAL(lstm2val_mdl_cfg, rstval, w2v_mdl)

    lstm2val_mdl.train(dev=False, save_model=True)
    lstm2val_mdl.baseline(test=True)
    lstm2val_mdl.evaluate(test=True)

'''
# MODELO 2: BOW2VAL  #################################################################################################
'''
bow2val_mdl_cfg = {"model": {"model_version":model_v, "learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": n_epochs, "batch_size": b_size, "seed": seed,
                            "early_st_first_epoch": 0, "early_st_monitor": "val_mean_absolute_error", "early_st_monitor_mode": "min", "early_st_patience": 20},
                    "session": {"gpu": gpu, "in_md5": False}}

if stage == 0:
    bow2val_mdl = BOW2VAL(bow2val_mdl_cfg, rstval)
    bow2val_mdl.train(dev=True, save_model=True)
    bow2val_mdl.baseline()
    bow2val_mdl.evaluate(test=False)

if stage == 1:
    # Sobreescribir la configuración por la mejor conocida: 
    with open('models/BOW2VAL/gijon/35f62204f54af44c43df8f1de1829abf/cfg.json') as f: best_cfg_data = json.load(f)
    rstval = RSTVALdataset(dts_cfg)
    bow2val_mdl_cfg["model"] = best_cfg_data["model"]
    bow2val_mdl = BOW2VAL(bow2val_mdl_cfg, rstval)

    bow2val_mdl.train(dev=False, save_model=True)
    bow2val_mdl.baseline(test=True)
    bow2val_mdl.evaluate(test=True)
'''
# MODELO 3: LSTM2RST ###################################################################################################
lstm2rst_mdl_cfg = {"model": {"model_version":model_v, "learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": n_epochs, "batch_size": b_size, "seed": seed,
                            "early_st_first_epoch": 0, "early_st_monitor": "val_accuracy", "early_st_monitor_mode": "max", "early_st_patience": 20},
                    "session": {"gpu": gpu, "in_md5": False}}

if stage == 0:
    lstm2rst_mdl = LSTM2RST(lstm2rst_mdl_cfg, rstval, w2v_mdl)
    lstm2rst_mdl.train(dev=True, save_model=True)
    lstm2rst_mdl.baseline()
    lstm2rst_mdl.evaluate(test=False)

if stage == 1:
    # Sobreescribir la configuración por la mejor conocida: 
    with open('models/LSTM2RST/gijon/42e900956c0a8cb09d72c383a143b3d2/cfg.json') as f: best_cfg_data = json.load(f)
    rstval = RSTVALdataset(dts_cfg)
    lstm2rst_mdl_cfg["model"] = best_cfg_data["model"]
    lstm2rst_mdl = LSTM2RST(lstm2rst_mdl_cfg, rstval, w2v_mdl)

    lstm2rst_mdl.train(dev=False, save_model=True)
    lstm2rst_mdl.baseline(test=True)
    lstm2rst_mdl.evaluate(test=True)

# MODELO 4: BOW2RST  ###################################################################################################
'''
bow2rst_dts = BOW2RSTdataset({"city": city, "seed": seed, "data_path": base_path, "save_path": base_path + "Datasets/",
                              "remove_plurals": remove_plurals, "stemming": stemming, "remove_accents": remove_accents, "remove_numbers": remove_numbers,
                              "min_reviews_rst": min_reviews_rst, "min_reviews_usr": min_reviews_usr,
                              "min_df": 5, "num_palabras": bow_n_words, "presencia": False, "text_column": "text",
                              "test_dev_split": .1})

bow2rst_mdl = BOW2RST({"model": {"learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": n_epochs, "batch_size": b_size, "seed": seed,
                                 "early_st_first_epoch": 0, "early_st_monitor": "val_loss", "early_st_monitor_mode": "min", "early_st_patience": 20},
                       "session": {"gpu": gpu, "in_md5": False}}, bow2rst_dts)

bow2rst_mdl.train(dev=True, save_model=True)

if stage == 0:
    bow2rst_mdl.train(dev=True, save_model=True)
    blr = bow2rst_mdl.baseline(test=False)
    mlr = bow2rst_mdl.evaluate(test=False)
    print("\t".join(list(map(lambda x: "%f\t%f" % (x[0], x[1]), list(zip(blr, mlr))))))

elif stage == 1:
    bow2rst_mdl.train(dev=False, save_model=True)
    blr = bow2rst_mdl.baseline(test=True)
    mlr = bow2rst_mdl.evaluate(test=True)
    print("\t".join(list(map(lambda x: "%f\t%f" % (x[0], x[1]), list(zip(blr, mlr))))))

# Obtener, para cada palabra, los restaurantes más afines


for wrd_idx, wrd in enumerate(bow2rst_dts.DATA["FEATURES_NAME"]):
    bow_word = np.zeros(bow_n_words)
    bow_word[wrd_idx]=1
    pred = bow2rst_mdl.MODEL.predict(np.expand_dims(bow_word, 0))
    rst_ids = np.argsort(-pred)[0][:3]
    rst_names = bow2rst_dts.DATA["TRAIN_DEV"].loc[bow2rst_dts.DATA["TRAIN_DEV"].id_restaurant.isin(rst_ids)].name.unique()

    print(wrd, " => ", ", ".join(rst_names))


# MODELO 5: LSTM&BOW2RST&VAL ###########################################################################################


lstmbow2rstval_dts = LSTMBOW2RSTVALdataset({"city": city, "seed": seed, "data_path": base_path, "save_path": base_path + "Datasets/",
                                            "remove_plurals": remove_plurals, "stemming": stemming, "remove_accents": remove_accents, "remove_numbers": remove_numbers,
                                            "min_reviews_rst": min_reviews_rst, "min_reviews_usr": min_reviews_usr,
                                            "min_df": 5, "num_palabras": bow_n_words, "presencia": False, "text_column": "text",
                                            "n_max_words": 0, "truncate_padding": True,
                                            "test_dev_split": .1})


lstmbow2rstval_mdl = LSTMBOW2RSTVAL({"model": {"learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": n_epochs, "batch_size": b_size, "seed": seed,
                                        "early_st_first_epoch": 0, "early_st_monitor": "val_loss", "early_st_monitor_mode": "min", "early_st_patience": 20},
                                        "session": {"gpu": gpu, "in_md5": False}}, lstmbow2rstval_dts, w2v_mdl)

lstmbow2rstval_mdl.train(dev=True, save_model=True)


# lstmbow2rstval_mdl.evaluate(verbose=1)
# lstmbow2rstval_mdl.eval_custom_text("Quiero comer un cachopo de cecina como una casa de grande")
# lstmbow2rstval_mdl.eval_custom_text("Quiero comer grande, barato y abundante")


'''