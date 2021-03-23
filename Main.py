# -*- coding: utf-8 -*-

from src.Common import parse_cmd_args

from src.datasets.text_datasets.W2Vdataset import W2Vdataset
from src.datasets.text_datasets.BOW2RSTdataset import BOW2RSTdataset
from src.datasets.text_datasets.LSTM2VALdataset import LSTM2VALdataset

from src.models.text_models.W2V import W2V
from src.models.text_models.BOW2RST import BOW2RST
from src.models.text_models.LSTM2VAL import LSTM2VAL

import nvgpu
import numpy as np

########################################################################################################################

args = parse_cmd_args()

city = "barcelona".lower().replace(" ", "") if args.ct is None else args.ct
stage = 0 if args.stg is None else args.stg

gpu = int(np.argmin(list(map(lambda x: x["mem_used_percent"], nvgpu.gpu_info()))))
seed = 100 if args.sd is None else args.sd
l_rate = 1e-3 if args.lr is None else args.lr
n_epochs = 5000 if args.ep is None else args.ep
b_size = 4096 if args.bs is None else args.bs

base_path = "/media/nas/pperez/data/TripAdvisor/"

# W2V ##################################################################################################################
w2v_dts = W2Vdataset({"cities": ["gijon", "barcelona", "madrid"], "city": "multi", "seed": seed, "data_path": base_path, "save_path": base_path + "Datasets/",
                      "remove_accents": True, "remove_numbers": True})

w2v_mdl = W2V({"model": {"train_set": "ALL_TEXTS", "min_count": 100, "window": 5, "n_dimensions": 300, "seed": seed},
               "session": {"gpu": gpu, "in_md5": False}}, w2v_dts)

# w2v_mdl.train()

# BOW2RST ##############################################################################################################
'''
bow2rst_dts = BOW2RSTdataset({"city": city, "seed": seed, "data_path": base_path, "save_path": base_path + "Datasets/",
                              "remove_accents": True, "remove_numbers": True,
                              "min_reviews_rst": 100, "min_reviews_usr": 1,
                              "min_df": 5, "num_palabras": 500, "presencia": False, "text_column": "text",
                              "test_dev_split": .1})

bow2rst_mdl = BOW2RST({"model": {"learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": n_epochs, "batch_size": b_size, "seed": seed,
                                 "early_st_first_epoch": 0, "early_st_monitor": "val_loss", "early_st_monitor_mode": "min", "early_st_patience": 20},
                       "session": {"gpu": gpu, "in_md5": False}}, bow2rst_dts)

bow2rst_mdl.train(dev=True, save_model=False)

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
'''
# LSTM2VAL ###########################################################################################################

lstm2val_dts = LSTM2VALdataset({"cities": [city], "city": city, "seed": seed, "data_path": base_path, "save_path": base_path + "Datasets/",
                                "remove_accents": True, "remove_numbers": True,
                                "n_max_words": 0, "test_dev_split": .1, "truncate_padding": False})

lstm2val_mdl = LSTM2VAL({"model": {"learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": 500, "batch_size": b_size, "seed": seed,
                                   "early_st_first_epoch": 0, "early_st_monitor": "val_loss", "early_st_monitor_mode": "min", "early_st_patience": 20},
                         "session": {"gpu": gpu, "in_md5": False}}, lstm2val_dts, w2v_mdl)

# lstm2val_mdl.train(dev=True, save_model=False)

if stage == 0:
    lstm2val_mdl.train(dev=True, save_model=True)

elif stage == 1:
    lstm2val_mdl.train(dev=False, save_model=True)
