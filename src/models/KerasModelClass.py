# -*- coding: utf-8 -*-
from src.models.ModelClass import *
from src.Common import print_e, print_g
from src.Callbacks import linear_decay, CustomStopper

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pylab as plt
import seaborn as sns


class KerasModelClass(ModelClass):

    def __init__(self, config, dataset):
        ModelClass.__init__(self, config=config, dataset=dataset)

    def __config_session__(self):
        # Selecciona una de las gpu dispobiles
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.CONFIG["session"]["gpu"])

        gpus = tf.config.experimental.list_physical_devices("GPU")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)

    def __fix_seeds__(self):
        # Fijar las semillas de numpy y TF
        np.random.seed(self.CONFIG["model"]["seed"])
        random.seed(self.CONFIG["model"]["seed"])
        tf.random.set_seed(self.CONFIG["model"]["seed"])

    def train(self, dev=False, save_model=False):

        train_seq = self.Sequence_TRAIN()

        if dev:
            dev_seq = self.Sequence_DEV()
            self.__train_model__(train_sequence=train_seq, dev_sequence=dev_seq, save_model=save_model)
        else:
            self.__train_model__(train_sequence=train_seq, dev_sequence=None, save_model=save_model)

    def __train_model__(self, train_sequence=None, dev_sequence=None, save_model=False, stop_monitor="val_loss", stop_mode="min"):

        is_dev = dev_sequence is not None

        train_cfg = {"verbose": 2, "workers": 1, "max_queue_size": 40}

        callbacks = []

        # Learning rate decay (lineal o cosine)
        lrs = tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: linear_decay(current_lr=lr,
                                                                                      initial_rl=self.CONFIG["model"]["learning_rate"],
                                                                                      final_lr=self.CONFIG["model"]["learning_rate"]/2,
                                                                                      epochs=self.CONFIG["model"]['epochs']))
        callbacks.append(lrs)

        # La carpeta cambia si es train_dev o train final
        final_folder = "dev/" if is_dev else ""

        # El número de epochs es el que se hizo en dev o el que se obtenga utilizando un early stopping
        if not is_dev:
            dev_log_path = self.MODEL_PATH+"dev/log.csv"
            if os.path.exists(dev_log_path):
                dev_log_data = pd.read_csv(dev_log_path)
                final_epoch_number = dev_log_data.val_f1.argmax()
            else:
                print_e("Unknown DEV epoch number...")
                exit()

        # Si es DEV, se hace añade el EarlyStopping (Esta versión custom permite no mirar las x primeras epochs)
        else:
            est = CustomStopper(monitor=stop_monitor, start_epoch=500, patience=100, verbose=1, mode=stop_mode)
            callbacks.append(est)

        # Si se quiere almacenar la salida del modelo (pesos/csv)
        if save_model:
            if os.path.exists(self.MODEL_PATH + final_folder + "checkpoint"):
                overwrite = input("Model already exists. Do you want to overwrite it? (y/n)")
                if overwrite == "y":
                    sure = input("Are you sure? (y/n)")
                    if sure != "y":
                        return
                else:
                    return

            # Crear la carpeta
            os.makedirs(self.MODEL_PATH + final_folder, exist_ok=True)

            # CSV logger
            log = tf.keras.callbacks.CSVLogger(self.MODEL_PATH + final_folder + "log.csv", separator=',', append=False)
            callbacks.append(log)

            # Solo guardar mirando "stop_monitor" en val cuando hay DEV
            if is_dev:
                mc = tf.keras.callbacks.ModelCheckpoint(self.MODEL_PATH + final_folder + "weights", save_weights_only=True, save_best_only=True, monitor=stop_monitor, mode=stop_mode)
                callbacks.append(mc)

        else:
            print_g("Not saving the model...")

        # Si es el entrenamiento final, no hay dev
        if not is_dev:
            hist = self.MODEL.fit(train_sequence,
                                  steps_per_epoch=train_sequence.__len__(),
                                  epochs=final_epoch_number,
                                  verbose=train_cfg["verbose"],
                                  workers=train_cfg["workers"],
                                  callbacks=callbacks,
                                  max_queue_size=train_cfg["max_queue_size"])

            self.MODEL.save_weights(self.MODEL_PATH + "weights")

        # Si es para gridsearch se añade el dev
        else:
            hist = self.MODEL.fit(train_sequence,
                                  steps_per_epoch=train_sequence.__len__(),
                                  epochs=self.CONFIG["model"]['epochs'],
                                  verbose=train_cfg["verbose"],
                                  validation_data=dev_sequence,
                                  validation_steps=dev_sequence.__len__(),
                                  workers=train_cfg["workers"],
                                  callbacks=callbacks,
                                  max_queue_size=train_cfg["max_queue_size"])

        # Almacenar gráfico con el entrenamiento
        if save_model:
            done_epochs = len(hist.history["loss"])
            plt.figure(figsize=(int((done_epochs*8)/500), 8))  # HAY QUE MEJORAR ESTO
            hplt = sns.lineplot(range(done_epochs), hist.history["f1"], label="f1")
            if is_dev:
                hplt = sns.lineplot(range(done_epochs), hist.history["val_f1"], label="val_f1")
            hplt.set_yticks(np.asarray(range(0, 110, 10)) / 100)
            hplt.set_xticks(range(0, done_epochs, 20))
            hplt.set_xticklabels(range(0, done_epochs, 20), rotation=45)
            hplt.set_title("Train history")
            hplt.set_xlabel("Epochs")
            hplt.set_ylabel("F1")
            hplt.grid(True)
            if not is_dev:
                plt.savefig(self.MODEL_PATH + final_folder + "history.jpg")
            else:
                plt.savefig(self.MODEL_PATH + final_folder + "history.jpg")
            plt.clf()
