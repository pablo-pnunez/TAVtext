# -*- coding: utf-8 -*-
import numpy as np
import argparse
import nvgpu
import os

gpu = np.argmin([g["mem_used_percent"] for g in nvgpu.gpu_info()])
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

from src.experiments.Common import load_best_model

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, help="Dataset", required=True)
parser.add_argument('-subset', type=str, help="Subset", required=True)
parser.add_argument('-model', type=str, help="Model name", required=True)
args = parser.parse_args()

# Cargar mejor modelo
model_class = load_best_model(model=args.model, dataset=args.dataset, subset=args.subset, gpu=gpu)

# Entrenar el modelo final
model_class.train(dev=False, save_model=True)

# Evaluar el modelo final
model_class.evaluate(test=True, user_info=True)