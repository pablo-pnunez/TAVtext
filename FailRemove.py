import os
import shutil
import pandas as pd

path = "models/LSTM2VAL/newyorkcity/"

for folder in os.listdir(path):
    total_path = path+folder+"/"
    
    if not os.path.exists(total_path+"dev/") or not os.path.exists(total_path+"dev/history.jpg"):
        shutil.rmtree(total_path)
        print(total_path)
    else:
        try:
            log = pd.read_csv(total_path+"dev/log.csv")
        except Exception:
            shutil.rmtree(total_path)
            print(total_path)
