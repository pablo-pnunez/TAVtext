# -*- coding: utf-8 -*-
from src.Common import parse_cmd_args, print_w
import pandas as pd
import numpy as np
import nvgpu
import os

args = parse_cmd_args()
gpu = np.argmin([g["mem_used_percent"] for g in nvgpu.gpu_info()]) if args.gpu is None else args.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

from src.experiments.Common import load_best_model
from src.Callbacks import EpochTime
import tensorflow as tf

model = "ATT2ITM" if args.mn is None else args.mn
dataset = "restaurants".lower().replace(" ", "") if args.dst is None else args.dst
subset = "gijon".lower().replace(" ", "") if args.sst is None else args.sst

model_class = load_best_model(model=model, dataset=dataset, subset=subset)

save_path = model_class.MODEL_PATH+"model_stats.csv"

if not os.path.exists(save_path):
    log = EpochTime()

    train_sequence, dev_sequence =  model_class.get_train_dev_sequences(dev=True)
    hist = model_class.MODEL.fit(train_sequence.batch(model_class.CONFIG["model"]['batch_size']).cache().prefetch(tf.data.AUTOTUNE),
                            epochs=1,
                            verbose=0,
                            callbacks=[log],
                            validation_data=dev_sequence.cache().batch(model_class.CONFIG["model"]['batch_size']).prefetch(tf.data.AUTOTUNE),
                            max_queue_size=20)


    dev_log_path = model_class.MODEL_PATH+"dev/log.csv"
    dev_log_data = pd.read_csv(dev_log_path)

    if model_class.CONFIG["model"]["early_st_monitor_mode"] == "min": final_epoch_number = dev_log_data[model_class.CONFIG["model"]["early_st_monitor"]].argmin()+1 
    else: final_epoch_number = dev_log_data[model_class.CONFIG["model"]["early_st_monitor"]].argmax()+1

    model_epochs = final_epoch_number
    model_eptime = hist.history["e_time"][-1]
    model_params = model_class.MODEL.count_params()

    model_res = (model, dataset, subset, model_epochs, model_eptime, model_params, model_class.MD5)
    pd.DataFrame([model_res], columns=["model", "dataset", "subset", "model_epochs", "model_eptime", "model_params", "model_md5"]).to_csv(save_path, index=False)

else:
    print_w("Stats already computed")


