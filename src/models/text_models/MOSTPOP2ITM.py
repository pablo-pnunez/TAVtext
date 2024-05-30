# -*- coding: utf-8 -*-
from src.models.text_models.RSTModel import RSTModel
import tensorflow as tf
import numpy as np

class MOSTPOP2ITM(RSTModel):
    """ Baseline """
    def __init__(self, config, dataset):
        RSTModel.__init__(self, config=config, dataset=dataset)

    def get_model(self):
        
        # Número de reviews por user 
        item_pop = self.DATASET.DATA["TRAIN_DEV"]["id_item"].value_counts().reset_index().sort_values("index").reset_index(drop=True)["id_item"].values
        # Normalizar entre 0 y 1 (innecesario, pero bueno)
        most_pop = np.round((item_pop-min(item_pop))/(max(item_pop)-min(item_pop)),2).tolist()
        # Definir la entrada del modelo
        input_layer = tf.keras.layers.Input(shape=(None,))
        # Crear una capa Lambda que retorna una lista fija de valores
        output_layer = tf.keras.layers.Lambda(lambda x: tf.tile(tf.constant([most_pop]), [tf.shape(x)[0], 1]))(input_layer)
        # Definir el modelo
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        # Esto no vale para nada por que no se entrena, pero se necesita para que funcione la evaluación
        model.compile(loss="mse", optimizer="adam")
        
        return model
        
    def __create_tfdata__(self, dataframe):
        data_x = tf.data.Dataset.from_tensor_slices(dataframe.seq.values)              
        data_y = tf.data.Dataset.from_tensor_slices(dataframe.id_item.values)
        data_y = data_y.map(lambda x: tf.one_hot(x, self.DATASET.DATA["N_ITEMS"]), num_parallel_calls=tf.data.AUTOTUNE)
        return tf.data.Dataset.zip((data_x, data_y))
    
    def train(self, dev=False, save_model=True):
        pass