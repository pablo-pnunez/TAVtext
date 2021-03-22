# -*- coding: utf-8 -*-

from src.Common import parse_cmd_args

from src.datasets.text_datasets.W2Vdataset import W2Vdataset
from src.datasets.text_datasets.BOW2RSTdataset import BOW2RSTdataset

from src.models.text_models.W2V import W2V
from src.models.text_models.BOW2RST import BOW2RST

import nvgpu
import numpy as np

########################################################################################################################

args = parse_cmd_args()

city = "gijon".lower().replace(" ", "") if args.ct is None else args.ct
stage = 0 if args.stg is None else args.stg

gpu = int(np.argmin(list(map(lambda x: x["mem_used_percent"], nvgpu.gpu_info()))))
seed = 100 if args.sd is None else args.sd
l_rate = 5e-4 if args.lr is None else args.lr
n_epochs = 1000 if args.ep is None else args.ep
b_size = 4096 if args.bs is None else args.bs

base_path = "/media/nas/pperez/data/TripAdvisor/"

# W2V ##################################################################################################################
'''
w2v_dts = W2Vdataset({"cities": ["gijon", "barcelona", "madrid"], "city": "multi", "remove_accents": True, "remove_numbers": True, "seed": seed,
                      "data_path": base_path, "save_path": base_path + "Datasets/"})

w2v_mdl = W2V({"model": {"train_set": "ALL_TEXTS", "min_count": 100, "window": 5, "n_dimensions": 300, "seed": seed},
               "session": {"gpu": gpu, "in_md5": False}}, w2v_dts)

w2v_mdl.train()

w2v_mdl.test("cachopo")
'''
# BOW2RST ##############################################################################################################

bow2rst_dts = BOW2RSTdataset({"city": city, "seed": seed, "data_path": base_path, "save_path": base_path + "Datasets/",
                              "remove_accents": True, "remove_numbers": True,
                              "min_reviews_rst": 100, "min_reviews_usr": 1,
                              "min_df": 5, "num_palabras": 200, "presencia": False, "text_column": "text",
                              "test_dev_split": .1})

bow2rst_mdl = BOW2RST({"model": {"learning_rate": l_rate, "final_learning_rate": l_rate//2, "epochs": n_epochs, "batch_size": b_size, "seed": seed,
                                 "early_st_first_epoch": 0, "early_st_monitor": "val_loss", "early_st_monitor_mode": "min", "early_st_patience": 20},
                       "session": {"gpu": gpu, "in_md5": False}}, bow2rst_dts)

if stage == 0:
    bow2rst_mdl.train(dev=True, save_model=True)
    bow2rst_mdl.baseline(test=False)

elif stage == 1:
    bow2rst_mdl.train(dev=False, save_model=True)
    bow2rst_mdl.baseline(test=True)

# LSTM2STARS ###########################################################################################################
