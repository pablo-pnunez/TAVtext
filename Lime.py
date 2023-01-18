
from lime import lime_text
from src.Common import parse_cmd_args
from lime.lime_text import LimeTextExplainer

from src.datasets.text_datasets.RestaurantDataset import RestaurantDataset
from src.models.text_models.ATT2ITM import ATT2ITM

import re
import nvgpu
import numpy as np
import pandas as pd
import tensorflow as tf

args = parse_cmd_args()

model = "ATT2ITM" if args.mn is None else args.mn
dataset = "restaurants".lower().replace(" ", "") if args.dst is None else args.dst
subset = "gijon".lower().replace(" ", "") if args.sst is None else args.sst

stage = -2 if args.stg is None else args.stg
model_v = "0" if args.mv is None else args.mv
neg_rate = 10

gpu = int(np.argmin(list(map(lambda x: x["mem_used_percent"], nvgpu.gpu_info())))) if args.gpu is None else args.gpu
seed = 100 if args.sd is None else args.sd
l_rate = 1e-4 if args.lr is None else args.lr
n_epochs = 1000 if args.ep is None else args.eps
b_size = 4096 if args.bs is None else args.bs

# if city=="london": b_size = 512

min_reviews_rst = 100
min_reviews_usr = 1
bow_pct_words = 10 if args.bownws is None else args.bownws
w2v_dimen = 512  # 300

remove_stopwords = 2  # 0, 1 o 2 (No quitar, quitar manual, quitar automático)
lemmatization = True
remove_accents = True
remove_numbers = True
truncate_padding = True

language = "es" if subset in ["gijon", "madrid", "barcelona"] else "fr" if subset in ["paris"] else "en"

if dataset == "restaurants":
    base_path = "/media/nas/datasets/tripadvisor/restaurants/"
elif dataset == "pois":
    base_path = "/media/nas/datasets/tripadvisor/pois/"
    language = "es"  # Están todas en español
elif dataset == "amazon":
    base_path = "/media/nas/datasets/amazon/"

dts_cfg = {"dataset": dataset, "subset": subset, "language": language, "seed": seed, "data_path": base_path, "save_path": "data/",  # base_path + "Datasets/",
           "remove_stopwords": remove_stopwords, "remove_accents": remove_accents, "remove_numbers": remove_numbers,
           "lemmatization": lemmatization,
           "min_reviews_rst": min_reviews_rst, "min_reviews_usr": min_reviews_usr,
           "min_df": 5, "bow_pct_words": bow_pct_words, "presencia": False, "text_column": "text",  # BOW
           "n_max_words": -50, "test_dev_split": .1, "truncate_padding": truncate_padding}

att2itm_mdl_cfg = {"model": {"model_version": model_v, "learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": n_epochs, "batch_size": b_size, "seed": seed,
                             "early_st_first_epoch": 0, "early_st_monitor": "val_loss", "early_st_monitor_mode": "min", "early_st_patience": 50},
                   "session": {"gpu": gpu, "in_md5": False}}

text_dataset = RestaurantDataset(dts_cfg)

att2itm_mdl = ATT2ITM(att2itm_mdl_cfg, text_dataset)
att2itm_mdl.train(dev=True, save_model=True)

classes = text_dataset.DATA["TRAIN_DEV"][["name", "id_item"]].drop_duplicates().sort_values("id_item").reset_index(drop=True)
class_names = classes["name"].values

explainer = LimeTextExplainer(class_names=class_names)

# For the multiclass case, we have to determine for which labels we will get explanations, via the 'labels' parameter.
# Below, we generate explanations for labels 0 and 17


def classifier_fn(d):
    # takes a list of d strings and outputs a (d, k) numpy array with prediction probabilities, where k is the number of classes. For ScikitClassifiers , this is classifier.predict_proba.
    ret = []
    for text in d:
        text = re.sub(r"\s+", " ", text, 0, re.MULTILINE)
        if len(text.strip()) > 0:
            text_prepro = att2itm_mdl.DATASET.prerpocess_text(text)
            lstm_text = att2itm_mdl.DATASET.DATA["TEXT_TOKENIZER"].texts_to_sequences([text_prepro])
            # lstm_text = [list(map(lambda x: att2itm_mdl.DATASET.DATA["TEXT_TOKENIZER"].word_index[x], text_prepro.split(" ")))]
            lstm_text_pad = tf.keras.preprocessing.sequence.pad_sequences(lstm_text, maxlen=att2itm_mdl.DATASET.DATA["MAX_LEN_PADDING"])
            preds_rst = att2itm_mdl.MODEL.predict([lstm_text_pad, np.arange(att2itm_mdl.DATASET.DATA["N_ITEMS"])[None, :]], verbose=0)
            ret.append(preds_rst.flatten())
        else:
            ret.append(np.zeros(len(classes)))

    return np.row_stack(ret)


np.random.seed(None)
test_real_sample = pd.read_pickle(text_dataset.DATASET_PATH+"ALL_DATA")
test_real_sample = test_real_sample[test_real_sample.test == 1]
test_real_sample = test_real_sample.iloc[np.random.randint(0, len(test_real_sample))]
text_instance = test_real_sample.text_source
text_instance = re.sub(r"\s+", " ", text_instance, 0, re.MULTILINE)

print("[RESTAURANTE REAL]: ", test_real_sample["name"])

att2itm_mdl.evaluate_text(text_instance)

items = classifier_fn([text_instance]).flatten().argsort()[-1:-5:-1]  # Items recomendados
exp = explainer.explain_instance(text_instance, classifier_fn, num_features=10, labels=items)

for itm in items:
    print('Explanation for class \033[1m %s \033[0m \n' % class_names[itm])
    print('\n'.join(map(str, exp.as_list(label=itm))))
    print("-"*50)
