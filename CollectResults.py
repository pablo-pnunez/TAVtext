import os
import json 
import pandas as pd
import numpy as np


city="gijon"
model="BOW2RST"
dev=True

if "BOW" in model:
    columns=["val_accuracy", "val_top_5", "val_top_10"]
    method=(np.max,np.argmax)
else:
    columns=["val_mean_absolute_error"]
    method=(np.min,np.argmin)

path="models/%s/%s/" % (model, city)

ret = []
for f in os.listdir(path):
    config_file=path+f+"/cfg.json"
    log_file=path+f+("/dev/" if dev else "")+"log.csv"
    log_data = pd.read_csv(log_file)

    with open(config_file) as json_file: config_data = json.load(json_file)

    res = {**config_data["model"],**config_data["dataset_config"]}

    for column in columns:
        res[column]=method[0](log_data[column])
        res["best_"+column+"_epoch"]=method[1](log_data[column])+1
    res["model_md5"]=f
    
    ret.append(list(res.values()))

ret = pd.DataFrame(ret, columns=list(res.keys()))
ret = ret.loc[:,ret.apply(pd.Series.nunique) != 1] # Eliminar columnas que no varían.
ret.to_csv(model+"_GS.csv")
print(ret)