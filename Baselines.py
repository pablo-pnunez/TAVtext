from src.Common import print_b, print_e
from src.datasets.text_datasets.RestaurantDataset import RestaurantDataset
from src.datasets.text_datasets.AmazonDataset import AmazonDataset
from src.datasets.text_datasets.POIDataset import POIDataset

from src.models.text_models.USEM2ITM import USEM2ITM
from src.models.text_models.ATT2ITM import ATT2ITM
from src.models.text_models.BOW2ITM import BOW2ITM

from cornac.metrics import Recall, Precision, FMeasure
from cornac.hyperopt import GridSearch, Discrete
from cornac.eval_methods import BaseMethod
from cornac.data.text import BaseTokenizer
from cornac.experiment import Experiment
from cornac.data import ReviewModality
import cornac

import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import nvgpu
import json


def load_set(dataset, subset):
    gpu = int(np.argmin(list(map(lambda x: x["mem_used_percent"], nvgpu.gpu_info())))) 

    models = ["USEM2ITM","ATT2ITM", "BOW2ITM"]
    eval_data = {}

    for model in models:
        best_model = pd.read_csv("models/best_models.csv")
        best_model = best_model.loc[(best_model.dataset == dataset) & (best_model.subset == subset) & (best_model.model == model)]["model_md5"].values[0]
        model_path = f"models/{model}/{dataset}/{subset}/{best_model}"
        with open(f'{model_path}/cfg.json') as f: model_config = json.load(f)
        dts_cfg = model_config["dataset_config"]
        with open(f'{model_path}/cfg.json') as f: model_config = json.load(f)
        mdl_cfg = {"model": model_config["model"], "session": {"gpu": gpu, "mixed_precision": False, "in_md5": False}}

        print_b(f"Loading best model: {best_model}")

        if dataset == "restaurants":
            # text_dataset = RestaurantDataset(dts_cfg, load=["TRAIN_DEV", "TEXT_TOKENIZER", "TEXT_SEQUENCES", "WORD_INDEX", "VOCAB_SIZE", "MAX_LEN_PADDING", "N_ITEMS", "FEATURES_NAME", "BOW_SEQUENCES"])
            text_dataset = RestaurantDataset(dts_cfg)
        elif dataset == "pois":
            text_dataset = POIDataset(dts_cfg)
        elif dataset == "amazon":
            text_dataset = AmazonDataset(dts_cfg)
        else:
            raise ValueError

        # Primero se eliminan los usuarios que no están en train de los conjuntos de val y test
        # Esto es solo para evaluar nuestro modelo en igualdad de condiciones
        train_users = text_dataset.DATA["TRAIN_DEV"][text_dataset.DATA["TRAIN_DEV"]["dev"] == 0].userId.unique()
        text_dataset.DATA["TRAIN_DEV"] = text_dataset.DATA["TRAIN_DEV"][text_dataset.DATA["TRAIN_DEV"]["userId"].isin(train_users)]
        text_dataset.DATA["TEST"] = text_dataset.DATA["TEST"][text_dataset.DATA["TEST"]["userId"].isin(train_users)]
        text_dataset.DATA["TEST"] = text_dataset.DATA["TEST"].drop_duplicates(subset=["userId", "id_item"], keep='last', inplace=False)

        # Evaluar nuestro modelo
        if "ATT2ITM" == model: model_class = ATT2ITM(mdl_cfg, text_dataset)
        elif "BOW2ITM" == model: model_class = BOW2ITM(mdl_cfg, text_dataset)
        elif "USEM2ITM" == model: model_class = USEM2ITM(mdl_cfg, text_dataset)
        else: raise NotImplementedError    
        # Cargar el modelo y evaluar
        model_class.train(dev=False, save_model=True)
        eval_data[model] = model_class.evaluate(test=True)

    # Luego se cargan los datos y se adaptan a Cornac
    all_data = pd.read_pickle(f"{text_dataset.DATASET_PATH}ALL_DATA")
    all_data["rating"] /= 10
    all_data = all_data[["userId", "id_item", "rating", "dev", "test", "text"]]

    # Eliminar usuarios desconocidos y dividir en 3 subconjuntos
    train_data = all_data[(all_data["dev"] == 0) & (all_data["test"] == 0)]
    train_users = train_data["userId"].unique()
    id_user, userId = pd.factorize(train_data["userId"])
    user_map = pd.DataFrame(zip(userId, id_user), columns=["userId", "id_user"])
    val_data = all_data[(all_data["dev"] == 1) & (all_data["userId"].isin(train_users))]
    test_data = all_data[(all_data["test"] == 1) & (all_data["userId"].isin(train_users))]

    train_data = train_data.merge(user_map)[["id_user", "id_item", "rating"]]
    val_data = val_data.merge(user_map)[["id_user", "id_item", "rating"]].drop_duplicates(subset=["id_user", "id_item"], keep='last', inplace=False)
    test_data = test_data.merge(user_map)[["id_user", "id_item", "rating"]].drop_duplicates(subset=["id_user", "id_item"], keep='last', inplace=False)

    # Instantiate a Base evaluation method using the provided train and test sets
    eval_method = BaseMethod.from_splits(train_data=train_data.to_records(index=False), val_data=val_data.to_records(index=False), test_data=test_data.to_records(index=False),  verbose=False, rating_threshold=3)
    # Ojo, lo anterior elimina las repeticiones de USUARIO, ITEM

    # max_vocab = 3000
    # max_doc_freq = 0.5
    # tokenizer = BaseTokenizer()
    # reviews = all_data.drop_duplicates(subset=["userId", "id_item"], keep='last', inplace=False).merge(user_map)[["id_user", "id_item", "text"]].to_records(index=False).tolist()
    # eval_method = BaseMethod.from_splits(train_data=train_data.to_records(index=False), review_text=rm, val_data=val_data.to_records(index=False), test_data=test_data.to_records(index=False),  verbose=True, rating_threshold=3)

    return eval_data, eval_method


seed = 2048

parser = argparse.ArgumentParser()
parser.add_argument('-dst', type=str, help="Dataset")
parser.add_argument('-sst', type=str, help="Subset")
args = parser.parse_args()

# datasets = {"pois": ["barcelona", "gijon"]}
dataset = "restaurants" if args.dst is None else args.dst
subset = "gijon" if args.sst is None else args.sst

# Cargar los datos
eval_data, eval_method = load_set(dataset, subset)

metrics = [
    FMeasure(k=1), FMeasure(k=5), FMeasure(k=10),
    Recall(k=1), Recall(k=5), Recall(k=10),
    Precision(k=1), Precision(k=5), Precision(k=10)]

md_bpr = cornac.models.BPR(seed=seed, verbose=True)
md_ease = cornac.models.EASE(seed=seed, verbose=True)

models = [
    cornac.models.MostPop(),
    GridSearch(
        model=md_bpr, space=[
            Discrete("k", [25, 50]),
            Discrete("max_iter", [50, 100]),
            Discrete("learning_rate", [1e-4, 5e-4, 1e-3]),
        ], metric=FMeasure(k=1), eval_method=eval_method),
    GridSearch(
        model=md_ease, space=[
            Discrete("posB", [True, False]),
        ], metric=FMeasure(k=1), eval_method=eval_method),
    # cornac.models.MF(seed=seed),  # Best parameter settings: {'k': 30, 'learning_rate': 5e-06, 'max_iter': 10}
    # cornac.models.MMMF(seed=seed),  # Best parameter settings: {'k': 5, 'learning_rate': 0.001, 'max_iter': 50}
    # cornac.models.NeuMF(seed=seed),
    # cornac.models.WBPR(seed=seed),
    # cornac.models.FM(seed=seed),
    # cornac.models.HPF(seed=seed),
    # cornac.models.NMF(seed=seed),
    # cornac.models.PMF(seed=seed),
    # cornac.models.SKMeans(seed=seed),
    # cornac.models.SVD(seed=seed),
    # cornac.models.WMF(seed=seed),
]

experiment = Experiment(
    eval_method=eval_method,
    show_validation=False,
    models=models,
    metrics=metrics,
    save_dir=f"models/ColdStart/{dataset}/{subset}",
    verbose=True
)

experiment.run()

metric_names = ['F1@1', 'F1@5', 'F1@10', 'Precision@1', 'Precision@5', 'Precision@10', 'Recall@1', 'Recall@5', 'Recall@10']
model_names = []
final_res = []

for model_name, model_data in eval_data.items():
    model_names.append(model_name)
    final_res.append(model_data[metric_names].values.tolist()[0])

for result in experiment.result:
    model_names.append(result.model_name)
    final_res.append([result.metric_avg_results[mtr] for mtr in metric_names])

final_res = pd.DataFrame(final_res, columns=metric_names)
final_res.insert(0, "Model", model_names)
final_res = final_res.sort_values("Model")
final_res.to_csv(f"models/ColdStart/{dataset}/{subset}/results.csv", index=False)

print(final_res.to_string())
