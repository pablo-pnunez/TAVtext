# -*- coding: utf-8 -*-

from src.Common import parse_cmd_args
from src.datasets.text_datasets.W2VDataset import W2VDatasetClass
from src.models.text_models.W2V import W2V

import nvgpu
import numpy as np

########################################################################################################################

args = parse_cmd_args()

city = "gijon".lower().replace(" ", "") if args.ct is None else args.ct
stage = 0 if args.stg is None else args.stg

gpu = int(np.argmin(list(map(lambda x: x["mem_used_percent"], nvgpu.gpu_info()))))
seed = 100 if args.sd is None else args.sd
l_rate = 5e-4 if args.lr is None else args.lr
n_epochs = 4000 if args.ep is None else args.ep
b_size = 1024 if args.bs is None else args.bs

base_path = "/media/nas/pperez/data/TripAdvisor/"

# W2V ##################################################################################################################

w2v_dts = W2VDatasetClass({"cities": ["gijon", "barcelona", "madrid"], "city": "multi", "remove_accents": True, "remove_numbers": True, "seed": seed,
                           "data_path": base_path, "save_path": base_path+"Datasets/"})

w2v_mdl = W2V({"model": {"train_set": "ALL_TEXTS", "min_count": 100, "window": 5, "n_dimensions": 300, "seed": seed},
               "session": {"gpu": gpu, "in_md5": False}}, w2v_dts)

w2v_mdl.train()

w2v_mdl.test("cachopo")

# W2V ##################################################################################################################
