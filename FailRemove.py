import os
import shutil
import pandas as pd

sets = {"restaurants": ["gijon", "barcelona", "madrid", "paris", "newyorkcity", "london"],
        "pois": ["barcelona", "madrid", "paris", "newyorkcity", "london"],
        "amazon": ["fashion", "digital_music"]}

models = ["ATT2ITM", "BOW2ITM", "USEM2ITM"]
models = ["BERT2ITM"]


# Para cada modelo/conjunto/ciudad, mirar si se acabó la ejecución
for model in models:
    for dataset in sets.keys():
        for city in sets[dataset]:
            path = f"models/{model}/{dataset}/{city}/"

            if os.path.exists(path):
                for folder in os.listdir(path):
                    total_path = path + folder + "/"
                    
                    if not os.path.exists(total_path + "dev/") or not os.path.exists(total_path + "dev/history.jpg"):
                        shutil.rmtree(total_path)
                        print(total_path)
                    else:
                        try:
                            log = pd.read_csv(total_path + "dev/log.csv")
                        except Exception:
                            shutil.rmtree(total_path)
                            print(total_path)
            else:
                print(f"Not executed yet: {path}")

