# -*- coding: utf-8 -*-

from src.Common import parse_cmd_args

from src.datasets.text_datasets.TextDataset import TextDataset
from src.datasets.text_datasets.W2Vdataset import W2Vdataset
from src.datasets.text_datasets.BOW2RSTdataset import BOW2RSTdataset
from src.datasets.text_datasets.LSTM2VALdataset import LSTM2VALdataset
from src.datasets.text_datasets.LSTMBOW2RSTVALdataset import LSTMBOW2RSTVALdataset

from src.models.text_models.W2V import W2V
from src.models.text_models.BOW2RST import BOW2RST
from src.models.text_models.BOW2VAL import BOW2VAL
from src.models.text_models.LSTM2VAL import LSTM2VAL
from src.models.text_models.LSTMBOW2RSTVAL import LSTMFBOW2RSTVAL
from src.models.text_models.LSTMBOW2RSTVAL import LSTMBOW2RSTVAL

import nvgpu
import numpy as np

########################################################################################################################

args = parse_cmd_args()

city = "gijon".lower().replace(" ", "") if args.ct is None else args.ct
stage = 0 if args.stg is None else args.stg

gpu = int(np.argmin(list(map(lambda x: x["mem_used_percent"], nvgpu.gpu_info()))))
seed = 100 if args.sd is None else args.sd
l_rate = 1e-4 if args.lr is None else args.lr
n_epochs = 5000 if args.ep is None else args.ep
b_size = 1024 if args.bs is None else args.bs

min_reviews_rst = 100
min_reviews_usr = 1
bow_n_words = 1024

stemming = False
remove_plurals = True
remove_accents = True
remove_numbers = True

base_path = "/media/nas/pperez/data/TripAdvisor/"

# W2V ##################################################################################################################
'''
w2v_dts = W2Vdataset({"cities": ["gijon", "barcelona", "madrid"], "city": "multi", "seed": seed, "data_path": base_path, "save_path": base_path + "Datasets/",
                      "remove_plurals": remove_plurals, "stemming": stemming, "remove_accents": remove_accents, "remove_numbers": remove_numbers})

w2v_dts = W2Vdataset({"cities": ["gijon", "barcelona", "madrid"], "city": "multi", "seed": seed, "data_path": base_path, "save_path": base_path + "Datasets/",
                      "remove_plurals": remove_plurals, "stemming": stemming, "remove_accents": remove_accents, "remove_numbers": remove_numbers})

w2v_mdl = W2V({"model": {"train_set": "ALL_TEXTS", "min_count": 100, "window": 5, "n_dimensions": 300, "seed": seed},
               "session": {"gpu": gpu, "in_md5": False}}, w2v_dts)

w2v_mdl.train()
'''
# BOW2VAL ##############################################################################################################

bow2val_dts = BOW2RSTdataset({"city": city, "seed": seed, "data_path": base_path, "save_path": base_path + "Datasets/",
                              "remove_plurals": remove_plurals, "stemming": stemming, "remove_accents": remove_accents, "remove_numbers": remove_numbers,
                              "min_reviews_rst": min_reviews_rst, "min_reviews_usr": min_reviews_usr,
                              "min_df": 5, "num_palabras": bow_n_words, "presencia": False, "text_column": "text",
                              "test_dev_split": .1})

bow2val_mdl = BOW2VAL({"model": {"learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": n_epochs, "batch_size": b_size, "seed": seed,
                                 "early_st_first_epoch": 0, "early_st_monitor": "val_loss", "early_st_monitor_mode": "min", "early_st_patience": 20},
                       "session": {"gpu": gpu, "in_md5": False}}, bow2val_dts)

bow2val_mdl.baseline()
bow2val_mdl.train(dev=True, save_model=False)

exit()

'''
# BOW2RST ##############################################################################################################

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

# LSTM2VAL ###########################################################################################################

lstm2val_dts = LSTM2VALdataset({"cities": [city], "city": city, "seed": seed, "data_path": base_path, "save_path": base_path + "Datasets/",
                                "remove_plurals": remove_plurals, "stemming": stemming, "remove_accents": remove_accents, "remove_numbers": remove_numbers,
                                "n_max_words": 0, "test_dev_split": .1, "truncate_padding": True})

lstm2val_mdl = LSTM2VAL({"model": {"learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": 500, "batch_size": b_size, "seed": seed,
                                   "early_st_first_epoch": 0, "early_st_monitor": "val_loss", "early_st_monitor_mode": "min", "early_st_patience": 20},
                         "session": {"gpu": gpu, "in_md5": False}}, lstm2val_dts, w2v_mdl)

if stage == 0:
    lstm2val_mdl.train(dev=True, save_model=True)

elif stage == 1:
    lstm2val_mdl.train(dev=False, save_model=True)

'''

# LSTM&FBOW2RST&VAL ##################################################################################################

lstmbow2rstval_dts = LSTMBOW2RSTVALdataset({"city": city, "seed": seed, "data_path": base_path, "save_path": base_path + "Datasets/",
                                            "remove_plurals": remove_plurals, "stemming": stemming, "remove_accents": remove_accents, "remove_numbers": remove_numbers,
                                            "min_reviews_rst": min_reviews_rst, "min_reviews_usr": min_reviews_usr,
                                            "min_df": 5, "num_palabras": bow_n_words, "presencia": False, "text_column": "text",
                                            "n_max_words": 0, "truncate_padding": True,
                                            "test_dev_split": .1})

fixed_bow = True

if fixed_bow:
    lstmbow2rstval_mdl = LSTMFBOW2RSTVAL({"model": {"learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": 500, "batch_size": b_size, "seed": seed,
                                          "early_st_first_epoch": 0, "early_st_monitor": "val_loss", "early_st_monitor_mode": "min", "early_st_patience": 20},
                                          "session": {"gpu": gpu, "in_md5": False}}, lstmbow2rstval_dts, w2v_mdl, bow2rst_mdl)

    lstmbow2rstval_mdl.train(dev=True, save_model=True)

else:

    lstmbow2rstval_mdl = LSTMBOW2RSTVAL({"model": {"learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": 500, "batch_size": b_size, "seed": seed,
                                         "early_st_first_epoch": 0, "early_st_monitor": "val_loss", "early_st_monitor_mode": "min", "early_st_patience": 20},
                                         "session": {"gpu": gpu, "in_md5": False}}, lstmbow2rstval_dts, w2v_mdl)

    lstmbow2rstval_mdl.train(dev=True, save_model=True)


# lstmbow2rstval_mdl.evaluate(verbose=1)
# lstmbow2rstval_mdl.eval_custom_text("Quiero comer un cachopo de cecina como una casa de grande")
lstmbow2rstval_mdl.eval_custom_text("Quiero comer grande, barato y abundante")

# ToDo: Obtener ese número prediciendo los 5 rst más probables, los 5 siguientes etc...
# ToDo: Baseline para valoración
