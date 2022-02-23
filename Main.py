# -*- coding: utf-8 -*-
import json
import nvgpu
import numpy as np

from src.Common import parse_cmd_args

from src.datasets.text_datasets.W2Vdataset import W2Vdataset
from src.datasets.text_datasets.RSTVALdataset import RSTVALdataset

from src.models.text_models.W2V import W2V
from src.models.text_models.BOW2VAL import BOW2VAL
from src.models.text_models.LSTM2VAL import LSTM2VAL

from src.models.text_models.LSTM2RST import LSTM2RST
from src.models.text_models.BOW2RST import BOW2RST

from src.models.text_models.LSTMBOW2RSTVAL import LSTMFBOW2RSTVAL
from src.models.text_models.LSTMBOW2RSTVAL import LSTMBOW2RSTVAL


# #######################################################################################################################

args = parse_cmd_args()

model = "BOW2VAL" if args.mn is None else args.mn
city = "gijon".lower().replace(" ", "") if args.ct is None else args.ct

stage = 1 if args.stg is None else args.stg
model_v = "3" if args.mv is None else args.mv

gpu = int(np.argmin(list(map(lambda x: x["mem_used_percent"], nvgpu.gpu_info())))) if args.gpu is None else args.gpu
seed = 100 if args.sd is None else args.sd
l_rate = 5e-4 if args.lr is None else args.lr
n_epochs = 1000 if args.ep is None else args.ep
b_size = 128 if args.bs is None else args.bs

min_reviews_rst = 100
min_reviews_usr = 1
bow_pct_words = 10 if args.bownws is None else args.bownws
w2v_dimen = 300

remove_stopwords = 2  # 0, 1 o 2 (No quitar, quitar manual, quitar automático)
lemmatization = True

stemming = False
remove_plurals = False
remove_accents = True
remove_numbers = True

base_path = "/media/nas/pperez/data/TripAdvisor/"

# ToDo: Que pasa con vegana (no aparece en el vocabulario)?
# ToDo: Retornar frases de las reviews como explicación?
# ToDo: Se puede obtener explicaciones también con w2v a la entrada? (creo que ya lo probé, poniendo una sola palabra en la lstm y viendo las probabilidades de la salida para cada restaurante)

# DATASET CONFIG #######################################################################################################

dts_cfg = {"city": city, "seed": seed, "data_path": base_path, "save_path": "data/",  # base_path + "Datasets/",
           "remove_plurals": remove_plurals, "remove_stopwords": remove_stopwords, "remove_accents": remove_accents, "remove_numbers": remove_numbers,
           "stemming": stemming, "lemmatization": lemmatization,
           "min_reviews_rst": min_reviews_rst, "min_reviews_usr": min_reviews_usr,
           "min_df": 5, "bow_pct_words": bow_pct_words, "presencia": False, "text_column": "text",  # BOW
           "n_max_words": 0, "test_dev_split": .1, "truncate_padding": True}

if stage == 0:
    rstval = RSTVALdataset(dts_cfg, load=["TRAIN_DEV", "WORD_INDEX", "VOCAB_SIZE", "FEATURES_NAME", "MAX_LEN_PADDING", "N_RST"])
else:
    rstval = RSTVALdataset(dts_cfg)

if "LSTM" in model:
    # W2V ----------------------------------------------------------------------------------------------------------------
    cities = ["gijon", "barcelona", "madrid"] if city in ["gijon", "barcelona", "madrid"] else []
    cities = ["newyorkcity", "london"] if city in ["newyorkcity", "london"] else cities
    cities = ["paris"] if city in ["paris"] else cities

    w2v_dts = W2Vdataset({"cities": cities, "city": "multi", "seed": seed, "data_path": base_path, "save_path": "data/",  # base_path + "Datasets/",
                          "remove_plurals": remove_plurals, "stemming": stemming, "lemmatization": lemmatization,
                          "remove_accents": remove_accents, "remove_numbers": remove_numbers,
                          }, load=[])

    w2v_mdl = W2V({"model": {"train_set": "ALL_TEXTS", "min_count": 100, "window": 5, "n_dimensions": w2v_dimen, "seed": seed},
                   "session": {"gpu": gpu, "in_md5": False}}, w2v_dts)

    w2v_mdl.train()

    if "LSTM2VAL" == model:
        # MODELO 1: LSTM2VAL ###################################################################################################
        lstm2val_mdl_cfg = {"model": {"model_version": model_v, "learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": n_epochs, "batch_size": b_size, "seed": seed,
                                    "early_st_first_epoch": 0, "early_st_monitor": "val_mean_absolute_error", "early_st_monitor_mode": "min", "early_st_patience": 20},
                            "session": {"gpu": gpu, "in_md5": False}}

        if stage == 0:
            lstm2val_mdl = LSTM2VAL(lstm2val_mdl_cfg, rstval, w2v_mdl)
            lstm2val_mdl.train(dev=True, save_model=True)
            # lstm2val_mdl.baseline()
            # lstm2val_mdl.evaluate(test=False)ez

        if stage == 1:
            bst_cfg = {"gijon": "32ba236b8eccf04c8e3236c259c59956", "barcelona": "e976e1661a848cdbb87efef2288cf762", "madrid": "faaa56ac7ce23b17881f3be8bca31e34"}
            # Sobreescribir la configuración por la mejor conocida:
            with open('models/LSTM2VAL/%s/%s/cfg.json' % (city, bst_cfg[city])) as f: best_cfg_data = json.load(f)
            dts_cfg = best_cfg_data["dataset_config"]
            rstval = RSTVALdataset(dts_cfg)
            lstm2val_mdl_cfg["model"] = best_cfg_data["model"]
            lstm2val_mdl = LSTM2VAL(lstm2val_mdl_cfg, rstval, w2v_mdl)

            lstm2val_mdl.train(dev=False, save_model=True)
            lstm2val_mdl.baseline(test=True)
            lstm2val_mdl.evaluate(test=True)

    if "LSTM2RST" == model:
        # MODELO 3: LSTM2RST ###################################################################################################
        lstm2rst_mdl_cfg = {"model": {"model_version": model_v, "learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": n_epochs, "batch_size": b_size, "seed": seed,
                                    "early_st_first_epoch": 0, "early_st_monitor": "val_accuracy", "early_st_monitor_mode": "max", "early_st_patience": 20},
                            "session": {"gpu": gpu, "in_md5": False}}

        if stage == 0:
            lstm2rst_mdl = LSTM2RST(lstm2rst_mdl_cfg, rstval, w2v_mdl)
            lstm2rst_mdl.train(dev=True, save_model=True)
            # lstm2rst_mdl.baseline()
            # lstm2rst_mdl.evaluate(test=False)

        if stage == 1:
            bst_cfg = {"gijon": "1c072eb640c097bf3c3f1a791f40c62b", "barcelona": "698a03dc1c00adaae7600e0144a8119a", "madrid": "8633acfd23aa82b5aca6c3e810cb3710"}
            # Sobreescribir la configuración por la mejor conocida:
            with open('models/LSTM2RST/%s/%s/cfg.json' % (city, bst_cfg[city])) as f: best_cfg_data = json.load(f) 
            dts_cfg = best_cfg_data["dataset_config"]
            rstval = RSTVALdataset(dts_cfg)
            lstm2rst_mdl_cfg["model"] = best_cfg_data["model"]
            lstm2rst_mdl = LSTM2RST(lstm2rst_mdl_cfg, rstval, w2v_mdl)

            lstm2rst_mdl.train(dev=False, save_model=True)
            lstm2rst_mdl.baseline(test=True)
            lstm2rst_mdl.evaluate(test=True)
            lstm2rst_mdl.evaluate_text("Busco un restaurante barato")

        '''
        # Obtener, para cada palabra, los restaurantes más afines
        for wrd_idx, wrd in enumerate(rstval.DATA["FEATURES_NAME"]):
            bow_word = np.zeros(bow_n_words)
            bow_word[wrd_idx] = 1
            pred = bow2rst_mdl.MODEL.predict(np.expand_dims(bow_word, 0))
            rst_ids = np.argsort(-pred)[0][:3]
            rst_names = rstval.DATA["TRAIN_DEV"].loc[rstval.DATA["TRAIN_DEV"].id_restaurant.isin(rst_ids)].name.unique()

            print(wrd, " => ", ", ".join(rst_names))
        '''

    if "LSTMBOW2RSTVAL" == model:
        # MODELO 5: LSTMBOW2RSTVAL ###########################################################################################
        lstmbow2rstval_mdl_cfg = {"model": {"model_version": model_v, "learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": n_epochs, "batch_size": b_size, "seed": seed,
                                            "early_st_first_epoch": 0, "early_st_monitor": "val_loss", "early_st_monitor_mode": "min", "early_st_patience": 20},
                                  "session": {"gpu": gpu, "in_md5": False}}

        if stage == 0:
            lstmbow2rstval_mdl = LSTMBOW2RSTVAL(lstmbow2rstval_mdl_cfg, rstval, w2v_mdl)
            lstmbow2rstval_mdl.train(dev=True, save_model=True)
            # lstmbow2rstval_mdl.baseline()
            # lstmbow2rstval_mdl.evaluate(test=False)

        if stage == 1:
            bst_cfg = {"gijon": "fcec46055a28f7430cb7119ca19f9ec9", "barcelona": "3ad69708e0be5c12ce48c97e1cf96791", "madrid": "d50f37f1de4e4e1aff5e07af2865364c", "paris": "81197eefea1c3c3344bafa16c3e1e237", "newyorkcity": "b428f1f775205a1322bc8f371e420e74"}
            # Sobreescribir la configuración por la mejor conocida:
            with open('models/LSTMBOW2RSTVAL/%s/%s/cfg.json' % (city, bst_cfg[city])) as f: best_cfg_data = json.load(f)

            dts_cfg = best_cfg_data["dataset_config"]
            rstval = RSTVALdataset(dts_cfg)
            lstmbow2rstval_mdl_cfg["model"] = best_cfg_data["model"]
            lstmbow2rstval_mdl = LSTMBOW2RSTVAL(lstmbow2rstval_mdl_cfg, rstval, w2v_mdl)

            lstmbow2rstval_mdl.train(dev=False, save_model=True)
            lstmbow2rstval_mdl.baseline(test=True)
            lstmbow2rstval_mdl.evaluate(test=True)

            # Ejemplos de recomendación
            if city == "gijon":
                lstmbow2rstval_mdl.eval_custom_text("Quiero comer un arroz con bogavante y con buenas vistas")
                lstmbow2rstval_mdl.eval_custom_text("Quiero comer un buen cachopo y beber sidra")
                lstmbow2rstval_mdl.eval_custom_text("Quiero probar la peor y más cara comida de la ciudad")
            if city == "paris":
                lstmbow2rstval_mdl.eval_custom_text("Restaurant avec la meilleure Steak Tartare de Paris")
            if city == "newyorkcity":
                lstmbow2rstval_mdl.eval_custom_text("Where can I eat the typical pastrami sandwich?")
                lstmbow2rstval_mdl.eval_custom_text("Where can I breakfast some cheesecake and coffee?")

else:

    if "BOW2VAL" == model:
        # MODELO 2: BOW2VAL  #################################################################################################

        bow2val_mdl_cfg = {"model": {"model_version": model_v, "learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": n_epochs, "batch_size": b_size, "seed": seed,
                                     "early_st_first_epoch": 0, "early_st_monitor": "val_mean_absolute_error", "early_st_monitor_mode": "min", "early_st_patience": 20},
                           "session": {"gpu": gpu, "in_md5": False}}

        if stage == 0:
            bow2val_mdl = BOW2VAL(bow2val_mdl_cfg, rstval)
            bow2val_mdl.train(dev=True, save_model=True)
            # bow2val_mdl.baseline()
            # bow2val_mdl.evaluate(test=False)

        if stage == 1:
            bst_cfg = {"gijon": "a91cdeda9af8ccb79214b435f14c0f40", "barcelona": "1ebdba6928f8d29ae3c25d27b6970396", "madrid": "bb11e8d8a4e96f0a4697991b0a63b02a"}
            # Sobreescribir la configuración por la mejor conocida:
            with open('models/BOW2VAL/%s/%s/cfg.json' % (city, bst_cfg[city])) as f: best_cfg_data = json.load(f) # 300
            # with open('models/BOW2VAL/gijon/15489c29fa15711844cf2300107a246d/cfg.json') as f: best_cfg_data = json.load(f) # 400
            dts_cfg = best_cfg_data["dataset_config"]
            rstval = RSTVALdataset(dts_cfg)
            bow2val_mdl_cfg["model"] = best_cfg_data["model"]
            bow2val_mdl = BOW2VAL(bow2val_mdl_cfg, rstval)

            bow2val_mdl.train(dev=False, save_model=True)
            bow2val_mdl.baseline(test=True)
            bow2val_mdl.evaluate(test=True)

    if "BOW2RST" == model:
        # MODELO 4: BOW2RST  ###################################################################################################
        bow2rst_mdl_cfg = {"model": {"model_version": model_v, "learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": n_epochs, "batch_size": b_size, "seed": seed,
                                     "early_st_first_epoch": 20, "early_st_monitor": "val_accuracy", "early_st_monitor_mode": "max", "early_st_patience": 20},
                           "session": {"gpu": gpu, "in_md5": False}}

        if stage == 0:
            bow2rst_mdl = BOW2RST(bow2rst_mdl_cfg, rstval)
            bow2rst_mdl.train(dev=True, save_model=True)
            # bow2rst_mdl.baseline()
            # bow2rst_mdl.evaluate(test=False)

        if stage == 1:
            bst_cfg = {"gijon": "c1f6541e4fac0312424cec3d8dfde6c3", "barcelona": "bc0ee46301cabbac60c6882752a58370", "madrid": "95058ad056b72a7ccd6f767da5429d40", "newyorkcity": "c3b019e94f4304e66dca2f0be0bb9fee", "paris": "8071cd9b1d81852e5bfece40a4b9d44a"}
            # Sobreescribir la configuración por la mejor conocida:
            with open('models/BOW2RST/%s/%s/cfg.json' % (city, bst_cfg[city])) as f: best_cfg_data = json.load(f)  # 300
            # with open('models/BOW2RST/gijon/c81670f3048bc05122aace9a0c996d37/cfg.json') as f: best_cfg_data = json.load(f)  # 400
            dts_cfg = best_cfg_data["dataset_config"]
            rstval = RSTVALdataset(dts_cfg)
            bow2rst_mdl_cfg["model"] = best_cfg_data["model"]
            bow2rst_mdl = BOW2RST(bow2rst_mdl_cfg, rstval)

            bow2rst_mdl.train(dev=False, save_model=True)
            bow2rst_mdl.baseline(test=True)
            bow2rst_mdl.evaluate(test=True)

            bow2rst_mdl.eval_custom_text("Quiero comer un arroz con bogavante y con buenas vistas")
            bow2rst_mdl.eval_custom_text("Donde puedo comer comida vegana")
            bow2rst_mdl.eval_custom_text("I want to eat some vegan food")
            bow2rst_mdl.eval_custom_text("The cheapest pizza in town")
            bow2rst_mdl.eval_custom_text("Spanish paella and sangria")
            bow2rst_mdl.eval_custom_text("Je veux manger des steaks pas chers")
