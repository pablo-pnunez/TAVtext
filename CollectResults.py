import os
import json 
import pandas as pd
import numpy as np


city="gijon"
model="LSTM2RST"
dev=True

if "RSTVAL" in model:
    columns = {"val_loss":"min", "val_valor_model_mean_absolute_error":"min",  "val_output_rst_top_5": "max", "val_output_rst_top_10":"max"}
elif "RST" in model:
    columns = {"val_accuracy":"max", "val_top_5": "max", "val_top_10":"max"}
elif "VAL" in model:
    columns = {"val_mean_absolute_error":"min"}
   
path="models/%s/%s/" % (model, city)

ret = []
for f in os.listdir(path):
    config_file=path+f+"/cfg.json"
    log_file=path+f+("/dev/" if dev else "")+"log.csv"
    log_data = pd.read_csv(log_file)

    with open(config_file) as json_file: config_data = json.load(json_file)

    res = {**config_data["model"],**config_data["dataset_config"]}

    for column in columns:
        method = (np.min, np.argmin) if columns[column] == "min" else (np.max, np.argmax)
        res[columns[column]+"_"+column]=method[0](log_data[column])
        res[columns[column]+"_"+column+"_epoch"]=method[1](log_data[column])+1
    res["model_md5"]=f
    
    ret.append(list(res.values()))

ret = pd.DataFrame(ret, columns=list(res.keys()))
ret = ret.loc[:,ret.apply(pd.Series.nunique) != 1] # Eliminar columnas que no varían.
ret.to_csv(model+"_GS.csv")
print(ret)