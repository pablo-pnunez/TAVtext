# -*- coding: utf-8 -*-
from src.Common import print_g
from src.models.text_models.RSTModel import RSTModel
import matplotlib.pyplot as plt

import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm


class XATT2LIKE(RSTModel):
    """ Predecir, utilizando attention, a partir de una review y un restaurante, si la review es del restaurante """

    def __init__(self, config, dataset, w2v_model):
        self.W2V_MODEL = w2v_model
        RSTModel.__init__(self, config=config, dataset=dataset)

    def train(self, dev=False, save_model=False, train_cfg={}):

        train_cfg = {"verbose": 2, "workers": 6, "class_weight": {0: 1, 1: self.CONFIG["model"]["negatve_rate"]}, "max_queue_size": 20, "multiprocessing": True}

        if dev:
            train_seq, dev_seq = self.get_train_dev_sequences(dev=dev)
            self.__train_model__(train_sequence=train_seq, dev_sequence=dev_seq, save_model=save_model, train_cfg=train_cfg)
        else:
            train_dev_seq = self.get_train_dev_sequences(dev=dev)
            self.__train_model__(train_sequence=train_dev_seq, dev_sequence=None, save_model=save_model, train_cfg=train_cfg)

    def get_model(self):
        print_g("Loading w2v...")
        word_vectors = self.W2V_MODEL.MODEL.wv
        w2v_emb_size = word_vectors.vectors.shape[1]
        embedding_matrix = np.zeros((self.DATASET.DATA["VOCAB_SIZE"], w2v_emb_size))
        for word, i in self.DATASET.DATA["WORD_INDEX"].items():
            try:
                embedding_vector = word_vectors[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                continue

        # Borrar modelo para ahorrar memoria
        del word_vectors

        model = self.get_sub_model(w2v_emb_size, embedding_matrix)

        return model

    def get_sub_model(self, w2v_emb_size, embedding_matrix):

        mv = self.CONFIG["model"]["model_version"]

        query_in = tf.keras.Input(shape=(self.DATASET.DATA["MAX_LEN_PADDING"]), dtype='int32', name="in_text")
        restaurant_in = tf.keras.Input(shape=(1,), dtype='int32', name="in_rest")

        # XAttention
        if mv == "0":
            rst_emb_size = 256
            key_emb_size = 64
            value_emb_size = 1

            query_emb = tf.keras.layers.Embedding(self.DATASET.DATA["VOCAB_SIZE"], w2v_emb_size, name="emb_text", weights=[embedding_matrix], trainable=False, mask_zero=True)(query_in)
            # query_emb = tf.keras.layers.Dropout(.3)(query_emb)
            rst_emb = tf.keras.layers.Embedding(self.DATASET.DATA["N_RST"], rst_emb_size, name="emb_rest")(restaurant_in)
            # rst_emb = tf.keras.layers.Dropout(.3)(rst_emb)
            model, attention = tf.keras.layers.MultiHeadAttention(num_heads=1, use_bias=False, output_shape=value_emb_size, key_dim=key_emb_size, value_dim=value_emb_size, name="att", dropout=0.0)(query=rst_emb, key=query_emb, value=query_emb, return_attention_scores=True)
            # model = tf.keras.layers.Dense(128, activation='relu')(model)
            # model = tf.keras.layers.Dense(64, activation='relu')(model)
            # model_out = tf.keras.layers.Dense(1, activation='sigmoid', name="out")(model)
            model_out = tf.keras.layers.Activation("sigmoid", name="out")(model)

        if mv == "1":  # Utilizando bias
            rst_emb_size = 256
            key_emb_size = 64
            value_emb_size = self.DATASET.DATA["N_RST"]

            query_emb = tf.keras.layers.Embedding(self.DATASET.DATA["VOCAB_SIZE"], w2v_emb_size, name="emb_text", weights=[embedding_matrix], trainable=False, mask_zero=True)(query_in)
            rst_emb = tf.keras.layers.Embedding(self.DATASET.DATA["N_RST"], rst_emb_size, name="emb_rest")(restaurant_in)
            model, attention = tf.keras.layers.MultiHeadAttention(num_heads=1, use_bias=True, output_shape=value_emb_size, key_dim=key_emb_size, value_dim=value_emb_size, name="att", dropout=0.0)(query=rst_emb, key=query_emb, value=query_emb, return_attention_scores=True)
            # model = tf.keras.layers.Dense(1)(model)
            model_out = tf.keras.layers.Activation("softmax", name="out")(model)

            # model_out = tf.keras.layers.Dense(self.DATASET.DATA["N_RST"], activation='softmax')(model)


        model = tf.keras.models.Model(inputs=[query_in, restaurant_in], outputs=[model_out])
        metrics = ['accuracy', tf.keras.metrics.Precision(name="pr"), tf.keras.metrics.Recall(name="rc")]
        model.compile(optimizer=tf.keras.optimizers.Adam(self.CONFIG["model"]["learning_rate"]), loss="binary_crossentropy", metrics=metrics)

        print(model.summary())

        return model

    def __create_dataset(self, dataframe, dset="train"):
        
        dataframe = dataframe[["seq", "id_restaurant"]].copy()
        # dataframe["prediction"] = 1
        # dataframe["aug"] = False
        dataframe = dataframe.sample(frac=1)

        data_x1 = tf.data.Dataset.from_tensor_slices(np.row_stack(dataframe.seq.to_list()))  # Textos
        data_x2 = tf.data.Dataset.from_tensor_slices(dataframe.id_restaurant)  # Restauarantes
        
        data_y = tf.data.Dataset.from_tensor_slices([1]).repeat(len(dataframe))  # Predicción

        if dset == "train":
            # 10, 20, 30
            times = self.CONFIG["model"]["negatve_rate"]
            shuffle_each_epoch = True
            seed = None
        else:
            times = 1
            shuffle_each_epoch = False
            seed = self.CONFIG["model"]["seed"]

        neg_len = len(dataframe) * times

        data_x1_a = data_x1.repeat(times)
        data_x2_a = data_x2.repeat(times).shuffle(neg_len, seed=seed, reshuffle_each_iteration=shuffle_each_epoch)
        data_y_a = tf.data.Dataset.from_tensor_slices([0]).repeat(neg_len)

        data_x1 = data_x1.concatenate(data_x1_a)
        data_x2 = data_x2.concatenate(data_x2_a)
        data_y = data_y.concatenate(data_y_a)
        
        '''
        n_items = len(dataframe)
        dataframe_aug = dataframe.copy().sample(n_items)
        dataframe_aug["aug"] = True
        dataframe_aug["prediction"] = 0
        dataframe_aug["id_restaurant"] = dataframe_aug["id_restaurant"].apply(lambda x: int(np.random.choice(np.concatenate([range(0, x), range(x+1, self.DATASET.DATA["N_RST"])]))))
        dataframe = pd.concat([dataframe, dataframe_aug], axis=0).reset_index(drop=True)
        '''

        data_x = tf.data.Dataset.zip((data_x1, data_x2))

        return tf.data.Dataset.zip((data_x, data_y)).shuffle(len(data_x))

    def get_train_dev_sequences(self, dev):

        all_data = self.DATASET.DATA["TRAIN_DEV"]

        if dev:
            train_data = all_data[all_data["dev"] == 0]
            dev_data = all_data[all_data["dev"] == 1]
            train_gn = self.__create_dataset(train_data, dset="train")
            dev_gn = self.__create_dataset(dev_data, dset="test")
            return train_gn, dev_gn
        else:
            train_dev_gn = self.__create_dataset(all_data)
            return train_dev_gn

    def evaluate(self, test=False):

        if test:
            test_data = self.DATASET.DATA["TEST"]
        else:
            test_data = self.DATASET.DATA["TRAIN_DEV"][self.DATASET.DATA["TRAIN_DEV"]["dev"] == 1]

        test_gn = self.__create_dataset(test_data)

        # Evaluar la recomendación de restaurantes


        # Evaluar la tarea de clasificación
        ret = self.MODEL.evaluate(test_gn.cache().batch(self.CONFIG["model"]['batch_size']).prefetch(tf.data.AUTOTUNE), verbose=0)
        print_g(dict(zip(self.MODEL.metrics_names, ret)))
       
    def plot_weights(self, text, att_weights, file_name="att_mtx"):
        data = att_weights
        data = np.round(data, 3)
        text = text.split(" ")
        data = np.expand_dims(data, 0) if len(data.shape) < 3 else data

        sz = len(text)
        plots = data.shape[0]

        fig = plt.figure(figsize=(sz*plots, sz))

        for i in range(plots):
            att = data[i, -sz:, -sz:]
            ax = fig.add_subplot(1, plots, i+1)
            cax = ax.matshow(att, extent=[0, sz, sz, 0])  # ,  cmap=matplotlib.cm.Spectral_r)
            # fig.colorbar(cax)
           
            for r in range(sz):
                for c in range(sz):
                    ax.text(c+.5, r+.5, f'{att[r,c]:0.2f}', ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

            ax.set_xticks(range(sz))
            ax.set_yticks(range(sz))
            ax.grid()

            ax.set_xticklabels('')
            ax.set_xticks(np.arange(sz)+.5, minor=True)
            ax.set_xticklabels(text, minor=True, rotation = 90)

            ax.set_yticklabels('')
            ax.set_yticks(np.arange(sz)+.5, minor=True)
            ax.set_yticklabels(text, minor=True)


        plt.savefig(f'{self.MODEL_PATH}{file_name}.png', bbox_inches='tight')

    def evaluate_text(self, text):

        print("\n")
        print_g("\'%s\'" % text)

        n_rsts = 3
        lstm_text = self.DATASET.DATA["TEXT_TOKENIZER"].texts_to_sequences([self.DATASET.prerpocess_text(text)])
        lstm_text_pad = tf.keras.preprocessing.sequence.pad_sequences(lstm_text, maxlen=self.DATASET.DATA["MAX_LEN_PADDING"])

        att_md = tf.keras.models.Model(inputs=[self.MODEL.input], outputs=[self.MODEL.get_layer("att").output[1]])

        # Calculo de salida de forma manual (SIN BIAS)
        '''
        rst = 0
        text_w2v = self.MODEL.get_layer("emb_text").weights[0].numpy().squeeze()[lstm_text_pad]
        text_values = text_w2v @ self.MODEL.get_layer("att").weights[2].numpy().squeeze()
        att_rst = att_md.predict([lstm_text_pad, np.array([rst])]).squeeze()
        weighted_values = (att_rst * text_values).sum()
        prediction = tf.nn.sigmoid(weighted_values * self.MODEL.get_layer("att").weights[3].numpy()[0][0][0]).numpy()
        '''
        
        all_att = att_md.predict([lstm_text_pad, np.array([range(self.DATASET.DATA["N_RST"])])]).squeeze()
        if len(all_att.shape) >2: all_att = all_att.max(0)
        all_att = all_att[:, -len(lstm_text[0]):]

        all_probs = self.MODEL.predict([lstm_text_pad, np.array([range(self.DATASET.DATA["N_RST"])])]).squeeze()

        # Obtener predicción y pesos para todos los restaurantes
        column_names=["name", "id_restaurant", "pred"]+text.split(" ")
        all_rst_att = [tuple([self.DATASET.DATA["TRAIN_DEV"][self.DATASET.DATA["TRAIN_DEV"].id_restaurant == i]["name"].values[0], i]+[all_probs[i]]+f) for i, f in enumerate(all_att.tolist())]
        all_rst_att = pd.DataFrame(all_rst_att, columns=column_names)
        all_rst_att = all_rst_att.sort_values(["pred"], ascending=False).reset_index(drop=True)
        all_rst_att.to_excel("all_rsts_att.xlsx")
        print(all_rst_att.head(10))

        # Para el mejor restaurante, evaluar sus reviews
        best_rst = all_rst_att.iloc[0]["id_restaurant"]
        rst_data = self.DATASET.DATA["TRAIN_DEV"][self.DATASET.DATA["TRAIN_DEV"].id_restaurant == best_rst]

        rev_preds = att_md.predict([np.row_stack(rst_data["seq"].to_list()), np.array([best_rst]*len(rst_data))]).squeeze()

        rst_wd_att = []

        for idx, (_, rev) in tqdm(enumerate(rst_data.iterrows()), total=len(rst_data)):
            txt_wds = rev.text.split(" ")
            txt_att = rev_preds[idx] 
            if len(txt_att.shape) > 1: txt_att = txt_att.max(0)  # Sumar cabezas
            txt_att = txt_att[-rev.n_words_text:]
            rst_wd_att.extend(list(zip(txt_wds, txt_att)))
        
        rst_wd_att = pd.DataFrame(rst_wd_att, columns=["word", "att"])
        rst_wd_att = rst_wd_att.groupby("word").apply(lambda x: pd.Series({"word":x.word.values[0], "att_sum": x.att.sum(), "freq": int(len(x))})).reset_index(drop=True)
        rst_wd_att = rst_wd_att.sort_values(["att_sum", "freq"], ascending=False)
        rst_wd_att.to_excel(f"rst{best_rst:03d}_wds.xlsx")
        print(rst_wd_att.head(10))

