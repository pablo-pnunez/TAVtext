from src.datasets.text_datasets.RestaurantDataset import RestaurantDataset
from src.datasets.text_datasets.AmazonDataset import AmazonDataset
from src.datasets.text_datasets.POIDataset import POIDataset

from src.models.text_models.ATT2ITM import ATT2ITM
from src.models.text_models.BOW2ITM import BOW2ITM
from src.models.text_models.USEM2ITM import USEM2ITM

import json
import nvgpu
import argparse
import numpy as np
import pandas as pd
from src.Common import print_b

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, help="Dataset", required=True)
parser.add_argument('-subset', type=str, help="Subset", required=True)
parser.add_argument('-model', type=str, help="Model name", required=True)
args = parser.parse_args()

gpu = int(np.argmin(list(map(lambda x: x["mem_used_percent"], nvgpu.gpu_info()))))

experiment_data = pd.read_csv("models/best_models.csv")
experiment_data = experiment_data.loc[(experiment_data.dataset == args.dataset) & (experiment_data.subset == args.subset) & (experiment_data.model == args.model)]
MD5 = experiment_data["model_md5"].values[0]
print_b(f"Loading best model: {MD5}")


# Se carga el dataset del modelo
model_path = f"models/{args.model}/{args.dataset}/{args.subset}/{MD5}"
with open(f'{model_path}/cfg.json') as f: model_config = json.load(f)
dts_cfg = model_config["dataset_config"]

if args.dataset == "restaurants": text_dataset = RestaurantDataset(dts_cfg)
elif args.dataset == "pois": text_dataset = POIDataset(dts_cfg)
elif args.dataset == "amazon": text_dataset = AmazonDataset(dts_cfg)
else: raise ValueError


# Cargar la configuración para los mejores modelos
with open(f'{model_path}/cfg.json') as f: model_config = json.load(f)
mdl_cfg = {"model": model_config["model"], "session": {"gpu": gpu, "mixed_precision": True, "in_md5": False}}

# Cargar el modelo pertinente
if "ATT2ITM" == args.model: model_class = ATT2ITM(mdl_cfg, text_dataset)
elif "BOW2ITM" == args.model: model_class = BOW2ITM(mdl_cfg, text_dataset)
elif "USEM2ITM" == args.model: model_class = USEM2ITM(mdl_cfg, text_dataset)
else: raise NotImplementedError

# Entrenar el modelo final
model_class.train(dev=False, save_model=True)
