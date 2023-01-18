import os
import json
from tkinter.messagebox import NO
import pandas as pd
import numpy as np


def format_range(wks, fr, fc, lr, lc, fmt):

    for row in range(fr, lr+1):
        for col in range(fc, lc+1):

            try:
                value = wks.table[row][col].number
            except Exception:
                value = None

            wks.write(row, col, value, fmt)      


# model = "ATT2ITM"
models = ["ATT2ITM", "BOW2ITM", "USEM2ITM"]

sets = {"restaurants": ["gijon", "barcelona", "madrid", "paris", "newyorkcity", "london"],
        "pois": ["barcelona", "madrid", "paris", "newyorkcity", "london"],
        "amazon": ["fashion", "digital_music"]}

dev = True

# Para cada modelo/conjunto/ciudad, mirar si se acabó la ejecución
for model in models:
    for dataset in sets.keys():

        writer = pd.ExcelWriter(f"{model}_{dataset}_GS.xlsx", engine="xlsxwriter")

        for subset in sets[dataset]:

            if "RSTVAL" in model:
                columns = {"val_loss": "min", "val_valor_model_mean_absolute_error": "min",  "val_output_rst_top_5": "max", "val_output_rst_top_10": "max"}  # adaptar al nuevo modo
                raise NotImplemented
            elif "RST" in model:
                columns = {"val_accuracy": "max", "val_top_5": "max", "val_top_10": "max"}  # adaptar al nuevo modo
                raise NotImplemented
            elif "VAL" in model:
                columns = {"val_mean_absolute_error": "min"}  # adaptar al nuevo modo
                raise NotImplemented
            elif "ITM" in model:
                columns = {"val_r1": {"best": "max", "others": ["val_r5", "val_r10", "epoch", "model_md5"]}}

            path = f"models/{model}/{dataset}/{subset}/"

            if not os.path.exists(path): continue

            # Obtener nombres de columnas y como obtener el mejor valors
            column_name = list(columns.keys())[0]
            column_best = columns[column_name]["best"]
            column_others = columns[column_name]["others"]
            column_best_name = column_best + "_" + column_name

            ret = []
            # Para cada fichero de resultados, obtener el mejor valor utilizando los parámetros establecidos en "columns"
            for f in os.listdir(path):
                config_file = path+f+"/cfg.json"
                log_file = path+f+("/dev/" if dev else "")+"log.csv"

                try: 
                    log_data = pd.read_csv(log_file)
                    log_data["epoch"] = log_data["epoch"]+1  # Sumar siempre 1 a las epochs
                except Exception:
                    continue

                with open(config_file) as json_file:
                    config_data = json.load(json_file)

                res = {**config_data["model"], **config_data["dataset_config"]}

                method = (np.min, np.argmin) if column_best == "min" else (np.max, np.argmax)
                best_epoch_data = log_data.iloc[method[1](log_data[column_name])]
                res[column_best_name] = best_epoch_data[column_name]

                for othc in column_others: 
                    if othc == "model_md5": res["model_md5"] = f
                    else: res[othc] = best_epoch_data[othc]

                ret.append(list(res.values()))

            ret = pd.DataFrame(ret, columns=list(res.keys()))
            ret = ret.loc[:, ret.apply(pd.Series.nunique) != 1]  # Eliminar columnas que no varían.

            # Una vez tenemos todos los mejores resultados guardadoes en "ret", podemos generar informes
            # Si hay varios modelos, separar los resultados
            index_pivot=["learning_rate"]

            if "model_version" in ret.columns:
                index_pivot = ["model_version"]+index_pivot
                ret["model_version"] = ret["model_version"].astype(int)

            # res_table = ret.pivot_table(index=index_pivot, columns=["batch_size"], values=column_best_name)
            res_table = ret.pivot_table(index=index_pivot, columns=["batch_size"], values=[column_best_name]+column_others, aggfunc=lambda x: x)[[column_best_name]+column_others]

            # epoch_table = ret.pivot_table(index=index_pivot, columns=["batch_size"], values=column_best_name_epoch)
            # md5_table = ret.pivot_table(index=index_pivot, columns=["batch_size"], values="model_md5", aggfunc=lambda x: x)

            n_cols = len(res_table.reset_index().columns)
            n_cols_data = len(res_table.columns)
            n_cols_dif = n_cols - n_cols_data
            n_rows = len(res_table)

            mrg = 1  # margin

            # Cuál es el mejor de todos?
            if column_best == "min": best_result = ret.sort_values(column_best_name).iloc[0]
            else: best_result = ret.sort_values(column_best_name, ascending=False).iloc[0]

            if "model_version" not in best_result.keys(): best_result["model_version"] = 0
            line_best_result = [dataset, subset, model, best_result["model_version"], best_result["learning_rate"], best_result["batch_size"], best_result[column_best_name]]
            for othc in column_others: line_best_result.append(best_result[othc])
            print("\t".join(map(str, line_best_result)))

            res_table.to_excel(writer, sheet_name=subset, startrow=1+mrg, startcol=mrg)
            # epoch_table.to_excel(writer, sheet_name=subset, startrow=2*mrg+3+n_rows, startcol=mrg)
            # md5_table.to_excel(writer, sheet_name=subset, startrow=1+mrg, startcol=2*mrg + n_cols)

            # FORMATEAR EL FICHERO EXCEL ###################################################################################################
            perc_format = writer.book.add_format({'num_format': '0.00%', 'align': 'center', 'valign': 'vcenter'})
            center_format = writer.book.add_format({'align': 'center', 'valign': 'vcenter'})

            worksheet = writer.sheets[subset]

            # Títulos
            # title_format = writer.book.add_format({ 'bold': 1, 'border': 1, 'align': 'center', 'valign': 'vcenter', 'fg_color': 'gray'})
            # worksheet.merge_range(mrg, mrg, mrg, mrg+n_cols-1, column_best_name.title(), title_format)
            # worksheet.merge_range(2*mrg+2+n_rows, mrg, 2*mrg+2+n_rows, mrg+n_cols-1, column_best_name_epoch.title(), title_format)
            # worksheet.merge_range(mrg, 2*mrg+n_cols, mrg, 2*mrg+2*n_cols-1, 'Model MD5', title_format)

            # worksheet.set_column(mrg, mrg+n_cols_dif-1, 15, center_format)
            # worksheet.set_column(2*mrg+n_cols, 2*mrg+n_cols+n_cols_dif-1, 15, center_format)
            # worksheet.set_column(2*mrg + n_cols + n_cols_dif, 2*mrg + 2*n_cols-1, 33, center_format)

            # worksheet.conditional_format(2+mrg, mrg+n_cols_dif, 2+mrg+n_rows-1, mrg+n_cols-1, {'type': '3_color_scale'})
            # worksheet.conditional_format(2*mrg+4+n_rows, mrg+n_cols_dif, 2*mrg+4+n_rows+n_rows-1, mrg+n_cols-1, {'type': '3_color_scale'})

            # format_range(worksheet, 2+mrg, mrg+n_cols_dif, 2+mrg+n_rows-1, mrg+n_cols-1, perc_format)
            # format_range(worksheet, 2*mrg+4+n_rows, mrg+n_cols_dif, 2*mrg+4+n_rows+n_rows-1, mrg+n_cols-1, center_format)

        writer.save()
