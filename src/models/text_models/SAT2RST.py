# -*- coding: utf-8 -*-
from shutil import which
from turtle import color, width

from regex import P
from src.Common import print_g
from src.Metrics import precision, recall, f1
from src.models.Common import create_weighted_binary_crossentropy
from src.models.text_models.RSTModel import RSTModel
from src.sequences.BaseSequence import BaseSequence
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer


class SAT2RST(RSTModel):
    """ Predecir, a partir de una review codificada mendiante LSTM, el restaurante de la review """

    def __init__(self, config, dataset, w2v_model):
        self.W2V_MODEL = w2v_model
        RSTModel.__init__(self, config=config, dataset=dataset)

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
        self.MODEL_VERSION = mv

        rst_no = self.DATASET.DATA["N_RST"]
        vocab_size = self.DATASET.DATA["VOCAB_SIZE"]

        model_in = tf.keras.Input(shape=(self.DATASET.DATA["MAX_LEN_PADDING"]), dtype='int32')
        rest_in = tf.keras.Input(shape=(rst_no), dtype='int32')

        if mv == "10":

            model = tf.keras.layers.Embedding(vocab_size, rst_no, name="emb_text", mask_zero=True)(model_in)
            model = tf.keras.layers.Dropout(.5)(model)
            # model = tf.keras.layers.BatchNormalization()(model)
            model = tf.keras.layers.Activation("tanh")(model)
            # model = tf.keras.layers.Dense(256, activation="tanh")(model)
            # model = tf.keras.layers.Dropout(.1)(model)
            # model = tf.keras.layers.Dense(out_shape, activation="tanh")(model)
            model = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1))(model)
            model_out = tf.keras.layers.Activation("softmax", name="out")(model)

        if mv == "oscar_11":

            emb_size = w2v_emb_size
            head_emb_size = 64
            heads = 2

            head_outs = []
      
            query_emb = tf.keras.layers.Embedding(self.DATASET.DATA["VOCAB_SIZE"], w2v_emb_size, name="w2v_text", weights=[embedding_matrix], trainable=False, mask_zero=True)
            # query_emb = tf.keras.layers.Embedding(self.DATASET.DATA["VOCAB_SIZE"], emb_size, name="w2v_text", mask_zero=True)
            mask_query = query_emb.compute_mask(model_in)
            mask_query = tf.expand_dims(tf.cast(mask_query, dtype=tf.float32), -1)
            mask_query = tf.tile(mask_query, [1, 1, rst_no])
            query_emb = query_emb(model_in)

            rests_emb = tf.keras.layers.Embedding(rst_no, emb_size, name=f"in_rsts")
            rests_emb = rests_emb(rest_in)

            for h in range(heads):

                ht_emb = tf.keras.layers.Dense(head_emb_size, name=f"h{h}_text")(query_emb)
                hr_emb = tf.keras.layers.Dense(head_emb_size, name=f"h{h}_rest")(rests_emb)

                ht_emb = tf.keras.layers.Dropout(.0)(ht_emb)
                hr_emb = tf.keras.layers.Dropout(.2)(hr_emb)

                model = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True), name=f"h{h}_dotprod")([ht_emb, hr_emb])
                # model = tf.keras.layers.BatchNormalization()(model)
                model = tf.keras.layers.Lambda(lambda x: x[0]*x[1])([model, mask_query])

                head_outs.append(model)

            # model = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 0), name="dotprod")(head_outs)
            model = tf.keras.layers.Lambda(lambda x: tf.math.reduce_max(x, 0))(head_outs)
            model = tf.keras.layers.Activation("tanh", name="dotprod")(model)
            model = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1),  name="sum")(model)
            model_out = tf.keras.layers.Activation("sigmoid", name="out")(model)

        if mv == "oscar_11_con":

            emb_size = w2v_emb_size
            head_emb_size = 32
            heads = 1

            q_embs = []
            r_embs = []
      
            query_emb = tf.keras.layers.Embedding(self.DATASET.DATA["VOCAB_SIZE"], w2v_emb_size, name="w2v_text", weights=[embedding_matrix], trainable=False, mask_zero=True)
            # query_emb = tf.keras.layers.Embedding(self.DATASET.DATA["VOCAB_SIZE"], emb_size, name="w2v_text", mask_zero=True)
            mask_query = query_emb.compute_mask(model_in)
            mask_query = tf.expand_dims(tf.cast(mask_query, dtype=tf.float32), -1)
            mask_query = tf.tile(mask_query, [1, 1, rst_no])
            query_emb = query_emb(model_in)

            rests_emb = tf.keras.layers.Embedding(rst_no, emb_size, name=f"in_rsts")
            rests_emb = rests_emb(rest_in)

            for h in range(heads):

                ht_emb = tf.keras.layers.Dense(head_emb_size, name=f"h{h}_text")(query_emb)
                hr_emb = tf.keras.layers.Dense(head_emb_size, name=f"h{h}_rest")(rests_emb)
                q_embs.append(ht_emb)
                r_embs.append(hr_emb)

            ht_emb = tf.keras.layers.Concatenate()(q_embs)
            hr_emb = tf.keras.layers.Concatenate()(r_embs)
            hr_emb = tf.keras.layers.Dropout(.1)(hr_emb)

            model = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True), name=f"h{h}_dotprod")([ht_emb, hr_emb])
            # model = tf.keras.layers.BatchNormalization()(model)
            model = tf.keras.layers.Lambda(lambda x: x[0]*x[1])([model, mask_query])
            model = tf.keras.layers.Activation("tanh", name="dotprod")(model)
            model = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1),  name="sum")(model)
            model_out = tf.keras.layers.Activation("sigmoid", name="out")(model)

        if mv == "oscar_11_basic":

            emb_size = 600
            h_size = 300
            # query_emb = tf.keras.layers.Embedding(self.DATASET.DATA["VOCAB_SIZE"], w2v_emb_size, name="w2v_text", weights=[embedding_matrix], trainable=False, mask_zero=True)
            query_emb = tf.keras.layers.Embedding(self.DATASET.DATA["VOCAB_SIZE"], emb_size, name="w2v_text", mask_zero=True)
            mask_query = query_emb.compute_mask(model_in)
            mask_query = tf.expand_dims(tf.cast(mask_query, dtype=tf.float32), -1)
            mask_query = tf.tile(mask_query, [1, 1, rst_no])
            ht_emb = query_emb(model_in)
            ht_emb = tf.keras.layers.Activation("tanh")(ht_emb)
            ht_emb = tf.keras.layers.Dense(h_size, name="word_emb")(ht_emb)

            rests_emb = tf.keras.layers.Embedding(rst_no, emb_size, name=f"in_rsts")
            hr_emb = rests_emb(rest_in)
            hr_emb = tf.keras.layers.Activation("tanh")(hr_emb)
            hr_emb = tf.keras.layers.Dense(h_size, name="rest_emb")(hr_emb)

            model = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([ht_emb, hr_emb])
            # model = tf.keras.layers.BatchNormalization()(model)
            # model = tf.keras.layers.GaussianNoise(.5)(model)

            model = tf.keras.layers.Lambda(lambda x: x[0]*x[1])([model, mask_query])
            model = tf.keras.layers.Activation("tanh", name="dotprod")(model)

            model = tf.keras.layers.Dropout(.2)(model)

            model = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1),  name="sum")(model)
            model_out = tf.keras.layers.Activation("sigmoid", name="out")(model)

        if mv == "oscar_simple_mask":

            def compute_attention_mask(query_emb_mask, key_emb_mask=None):
                if key_emb_mask is None: key_emb_mask = query_emb_mask
                att_mask = tf.logical_and(tf.expand_dims(query_emb_mask, 2), tf.expand_dims(key_emb_mask, 1))
                return att_mask

            key_emb_size = 128
            value_emb_size = key_emb_size

            rest_in = rest_in+1

            rst_in_emb = np.eye(rst_no+1)
            rst_in_emb[:,0] = 0

            query_emb = tf.keras.layers.Embedding(self.DATASET.DATA["VOCAB_SIZE"], w2v_emb_size, name="emb_text", weights=[embedding_matrix], trainable=False, mask_zero=True)
            rests_emb = tf.keras.layers.Embedding(rst_no, rst_no, name="emb_rests", embeddings_initializer=tf.constant_initializer(rst_in_emb), trainable=False, mask_zero=True)
            
            attention_mask = compute_attention_mask(query_emb.compute_mask(model_in), rests_emb.compute_mask(rest_in))

            query_emb = query_emb(model_in)
            rests_emb = rests_emb(rest_in)

            att_ly = tf.keras.layers.MultiHeadAttention(num_heads=1, use_bias=False, kernel_constraint=tf.keras.constraints.NonNeg(), output_shape=rst_no, key_dim=key_emb_size, value_dim=value_emb_size, name="att", dropout=0.0)
            model, attention = att_ly(query=query_emb, value=rests_emb, return_attention_scores=True, attention_mask=attention_mask)
            model = tf.keras.layers.GlobalAveragePooling1D()(model, attention_mask[:, 1])
            model_out = tf.keras.layers.Activation("softmax", name="out")(model)          

        if mv == "oscar":

            def compute_attention_mask(query_emb_mask, key_emb_mask=None):
                if key_emb_mask is None: key_emb_mask = query_emb_mask
                att_mask = tf.logical_and(tf.expand_dims(query_emb_mask, 2), tf.expand_dims(key_emb_mask, 1))
                return att_mask

            key_emb_size = 128
            value_emb_size = key_emb_size

            query_emb = tf.keras.layers.Embedding(self.DATASET.DATA["VOCAB_SIZE"], w2v_emb_size, name="emb_text", weights=[embedding_matrix], trainable=False, mask_zero=True)
            rests_emb = tf.keras.layers.Embedding(rst_no, rst_no, name="emb_rests", embeddings_initializer=tf.constant_initializer(np.eye(rst_no)), trainable=False, mask_zero=True)
            
            attention_mask = compute_attention_mask(query_emb.compute_mask(model_in), rests_emb.compute_mask(rest_in))

            query_emb = query_emb(model_in)
            rests_emb = rests_emb(rest_in)

            att_ly = tf.keras.layers.MultiHeadAttention(num_heads=6, output_shape=rst_no, key_dim=key_emb_size, value_dim=value_emb_size, name="att", dropout=0.0)
            model, attention = att_ly(query=query_emb, value=rests_emb, return_attention_scores=True)  #, attention_mask=attention_mask)
            model = tf.keras.layers.GlobalAveragePooling1D()(model)
            model_out = tf.keras.layers.Activation("softmax", name="out")(model)          

        if mv == "oscar_mask":

            def compute_attention_mask(query_emb_mask, key_emb_mask=None):
                if key_emb_mask is None: key_emb_mask = query_emb_mask
                att_mask = tf.logical_and(tf.expand_dims(query_emb_mask, 2), tf.expand_dims(key_emb_mask, 1))
                return att_mask

            key_emb_size = 128
            value_emb_size = key_emb_size

            query_emb = tf.keras.layers.Embedding(self.DATASET.DATA["VOCAB_SIZE"], w2v_emb_size, name="emb_text", weights=[embedding_matrix], trainable=False, mask_zero=True)
            rests_emb = tf.keras.layers.Embedding(rst_no, rst_no, name="emb_rests", embeddings_initializer=tf.constant_initializer(np.eye(rst_no)), trainable=False, mask_zero=True)
            
            attention_mask = compute_attention_mask(query_emb.compute_mask(model_in), rests_emb.compute_mask(rest_in))

            query_emb = query_emb(model_in)
            rests_emb = rests_emb(rest_in)

            att_ly = tf.keras.layers.MultiHeadAttention(num_heads=6, output_shape=rst_no, key_dim=key_emb_size, value_dim=value_emb_size, name="att", dropout=0.0)
            model, attention = att_ly(query=query_emb, value=rests_emb, return_attention_scores=True , attention_mask=attention_mask)
            model = tf.keras.layers.GlobalAveragePooling1D()(model)
            model_out = tf.keras.layers.Activation("softmax", name="out")(model)       

        if mv == "oscar_inv":

            def compute_attention_mask(query_emb_mask, key_emb_mask=None):
                if key_emb_mask is None: key_emb_mask = query_emb_mask
                att_mask = tf.logical_and(tf.expand_dims(query_emb_mask, 2), tf.expand_dims(key_emb_mask, 1))
                return att_mask

            key_emb_size = 128
            value_emb_size = key_emb_size

            query_emb = tf.keras.layers.Embedding(self.DATASET.DATA["VOCAB_SIZE"], w2v_emb_size, name="emb_text", weights=[embedding_matrix], trainable=False, mask_zero=True)
            rests_emb = tf.keras.layers.Embedding(rst_no, rst_no, name="emb_rests", embeddings_initializer=tf.constant_initializer(np.eye(rst_no)), trainable=False, mask_zero=True)
            
            attention_mask = compute_attention_mask(query_emb.compute_mask(model_in), rests_emb.compute_mask(rest_in))

            query_emb = query_emb(model_in)
            rests_emb = rests_emb(rest_in)

            att_ly = tf.keras.layers.MultiHeadAttention(num_heads=4, output_shape=rst_no, key_dim=key_emb_size, value_dim=value_emb_size, name="att", dropout=0.5)
            model, attention = att_ly(query=rests_emb, value=query_emb, return_attention_scores=True)  #, attention_mask=attention_mask)
            model = tf.keras.layers.GlobalAveragePooling1D()(model)
            model_out = tf.keras.layers.Activation("softmax", name="out")(model)    

        if mv == "oscar_inv_simple":

            key_emb_size = 128
            value_emb_size = 1

            query_emb = tf.keras.layers.Embedding(self.DATASET.DATA["VOCAB_SIZE"], w2v_emb_size, name="emb_text", weights=[embedding_matrix], trainable=False, mask_zero=True)
            rests_emb = tf.keras.layers.Embedding(rst_no, rst_no, name="emb_rests", embeddings_initializer=tf.constant_initializer(np.eye(rst_no)), trainable=False, mask_zero=True)
            
            query_emb = query_emb(model_in)
            rests_emb = rests_emb(rest_in)

            att_ly = tf.keras.layers.MultiHeadAttention(num_heads=6, use_bias=False, output_shape=1, key_dim=key_emb_size, value_dim=value_emb_size, name="att", dropout=0.0)
            model, attention = att_ly(query=rests_emb, value=query_emb, return_attention_scores=True)
            model = tf.keras.layers.Flatten()(model)    
            model_out = tf.keras.layers.Activation("softmax", name="out")(model)

        if "oscar" in self.MODEL_VERSION:
            model = tf.keras.models.Model(inputs=[model_in, rest_in], outputs=[model_out], name=f"{self.MODEL_NAME}_{self.MODEL_VERSION}")
        else:
            model = tf.keras.models.Model(inputs=[model_in], outputs=[model_out], name=f"{self.MODEL_NAME}_{self.MODEL_VERSION}")
        
        activation_function = model.layers[-1].activation.__name__
        optimizer = tf.keras.optimizers.Adam(self.CONFIG["model"]["learning_rate"])

        if activation_function == "sigmoid":
            metrics = ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='t5'), tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='t10')]
            model.compile(loss=create_weighted_binary_crossentropy(1, 1), metrics=metrics, optimizer=optimizer)
        elif activation_function == "softmax":
            metrics = ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='t5'), tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='t10')]
            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=metrics, optimizer=optimizer)
        
        print(model.summary())
        return model

    def __create_dataset(self, dataframe):
        data_x = tf.data.Dataset.from_tensor_slices(np.row_stack(dataframe.seq.to_list()))
        data_y = tf.data.Dataset.from_tensor_slices(dataframe.id_restaurant.values)
        data_y = data_y.map(lambda x: tf.one_hot(x, self.DATASET.DATA["N_RST"]), num_parallel_calls=tf.data.AUTOTUNE)
        
        if len(self.MODEL.inputs) == 2:
            data_x_rsts = tf.data.Dataset.from_tensor_slices([range(self.DATASET.DATA["N_RST"])]).repeat(len(dataframe))
            data_x = tf.data.Dataset.zip((data_x, data_x_rsts))
        
        return tf.data.Dataset.zip((data_x, data_y))

    def get_train_dev_sequences(self, dev):
 
        if self.DATASET.CONFIG["city"] != "gijon":
            all_data = self.DATASET.DATA["TRAIN_DEV"].sample(frac=1)
        else: 
            all_data = self.DATASET.DATA["TRAIN_DEV"]

        if dev:
            train_data = all_data[all_data["dev"] == 0]
            dev_data = all_data[all_data["dev"] == 1]
            train_gn = self.__create_dataset(train_data)
            dev_gn = self.__create_dataset(dev_data)
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

        if isinstance(test_gn, tf.data.Dataset):
            ret = self.MODEL.evaluate(test_gn.cache().batch(self.CONFIG["model"]['batch_size']).prefetch(tf.data.AUTOTUNE), verbose=0)
        else:
            ret = self.MODEL.evaluate(test_gn, verbose=0)

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

    def emb_tsne(self):

        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from bokeh.plotting import ColumnDataSource, figure, output_file, save
        from bokeh.layouts import column, row, gridplot
        from scipy.spatial.distance import cdist
        from bokeh.models.widgets import Dropdown

        wrd_embs = tf.keras.models.Model(inputs=[self.MODEL.input[0]], outputs=[self.MODEL.get_layer("word_emb").output])
        rst_embs = tf.keras.models.Model(inputs=[self.MODEL.input[1]], outputs=[self.MODEL.get_layer("rest_emb").output])

        rst_embs = rst_embs.predict([list(range(self.DATASET.DATA["N_RST"]))]).squeeze()

        rest_names = self.DATASET.DATA["TRAIN_DEV"][["id_restaurant", "name"]].sort_values("id_restaurant").drop_duplicates().name.values.tolist()
        word_names = np.array(["UNK"]+list(self.DATASET.DATA["WORD_INDEX"].keys()))
        # word_freq = list(zip(self.DATASET.DATA["TEXT_TOKENIZER"].word_counts.keys(),self.DATASET.DATA["TEXT_TOKENIZER"].word_counts.values()))
        # word_freq = pd.DataFrame(word_freq, columns=["word", "freq"]).sort_values("freq", ascending=False).reset_index()
        # most_freq_w = word_freq# [word_freq.freq>20]

        # wrd_embs = wrd_embs.predict(most_freq_w["index"].values).squeeze()
        # word_names = np.array(most_freq_w["word"].values)

        wrd_embs = wrd_embs.predict(list(range(self.DATASET.DATA["VOCAB_SIZE"]))).squeeze()

        att = cdist(rst_embs, wrd_embs, metric=np.dot)
        att = np.tanh(att)

        for id_r, r in enumerate(rest_names):
            farthest = list(word_names[np.argsort(att[id_r,:])[:5]])
            closest = list(word_names[np.argsort(att[id_r,:])[-5:]])

            print(rest_names[id_r], closest, farthest)

        if wrd_embs.shape[1]>2:
            tsne = TSNE(n_components=2, init="pca", learning_rate="auto", metric=lambda x, y: -np.dot(x, y))        
            all_tsne = tsne.fit_transform(np.concatenate([rst_embs, wrd_embs]))
            rst_pca = all_tsne[:self.DATASET.DATA["N_RST"], :]
            wrd_pca = all_tsne[self.DATASET.DATA["N_RST"]:, :]
        else:
            rst_pca = rst_embs
            wrd_pca = wrd_embs

        source_w = ColumnDataSource(data=dict(x=wrd_pca[:, 0], y=wrd_pca[:, 1], desc=word_names))
        source_r = ColumnDataSource(data=dict(x=rst_pca[:, 0], y=rst_pca[:, 1], desc=rest_names))

        TOOLTIPS = [("index", "$index"), ("(x,y)", "($x, $y)"), ("desc", "@desc")]
        p = figure(width=800, height=600, tooltips=TOOLTIPS)
        p.scatter('x', 'y', size=5, source=source_w, color="red")
        p.scatter('x', 'y', size=10, source=source_r)
        output_file(filename="tsne.html", title="Static HTML file")
        
        menu = list(zip(rest_names,list(map(str,range(self.DATASET.DATA["N_RST"])))))
        dropdown = Dropdown(label="Restaurante", menu=menu, width=200)
        
        save(p)

    def evaluate_text(self, text):

        print("\n")
        print(f"\033[92m[QUERY] '{text}'\033[0m")

        imput_number = len(self.MODEL.inputs)

        n_rsts = 5
        text_prepro = self.DATASET.prerpocess_text(text)
        lstm_text = self.DATASET.DATA["TEXT_TOKENIZER"].texts_to_sequences([text_prepro])
        lstm_text_pad = tf.keras.preprocessing.sequence.pad_sequences(lstm_text, maxlen=self.DATASET.DATA["MAX_LEN_PADDING"])
        print(f"[PREPR] '{text_prepro}'")
        print(f"[TXT2ID] '{lstm_text}'")
   
        if "att" in [ly.name for ly in self.MODEL.layers]:
            att_md = tf.keras.models.Model(inputs=[self.MODEL.input], outputs=[self.MODEL.get_layer("att").outputs[-1]])
            
            if imput_number == 2:
                att = att_md.predict([lstm_text_pad, np.arange(self.DATASET.DATA["N_RST"])[None,:]]).squeeze(axis=0)
                att = att.mean(axis=0)

                att_text = pd.DataFrame(att[:,-len(lstm_text[0]):].T, columns=self.DATASET.DATA["TRAIN_DEV"].sort_values("id_restaurant")["name"].unique())
                att_text.insert(0, "text", text_prepro.split(" "))
                att_text.transpose().to_excel("att_text.xlsx")

                # np.savetxt("att.csv", att[:,-len(lstm_text[0]):])
                for widx, word in enumerate(self.DATASET.prerpocess_text(text).split(" ")[::-1]):
                    if "oscar_inv" in self.MODEL_VERSION:  rst = np.argsort(-att[:, -widx-1])[:3]
                    else:  rst = np.argsort(-att[-widx-1])[:3]
                    rst_list = []
                    for rst_sel in rst: rst_list.append(self.DATASET.DATA["TRAIN_DEV"].loc[self.DATASET.DATA["TRAIN_DEV"].id_restaurant == rst_sel]["name"].values[0])
                    print(f' · {word} -> {rst_list}')
                
            else:
                att = att_md.predict(lstm_text_pad).squeeze(axis=0)
            
                self.plot_weights(text=text, att_weights=att)
                self.plot_weights(text=text, att_weights=att.max(axis=0), file_name="att_mtx_max")

        elif "emb_text" in [ly.name for ly in self.MODEL.layers]:
            emb_mdl = tf.keras.models.Model(inputs=[self.MODEL.input], outputs=[self.MODEL.get_layer("emb_text").output])
            emb_text = emb_mdl.predict(lstm_text_pad).squeeze()[-len(lstm_text[0]):, :]
            emb_text = pd.DataFrame(emb_text, columns=self.DATASET.DATA["TRAIN_DEV"].sort_values("id_restaurant")["name"].unique())
            emb_text.insert(0, "text", text_prepro.split(" "))
            emb_text.to_excel("word_embs.xlsx")

        elif "dotprod" in [ly.name for ly in self.MODEL.layers]:

            att_md = tf.keras.models.Model(inputs=[self.MODEL.input], outputs=[self.MODEL.get_layer("dotprod").output])
            
            if imput_number == 2:
                att = att_md.predict([lstm_text_pad, np.arange(self.DATASET.DATA["N_RST"])[None,:]]).squeeze(axis=0)

                print(att.T[0, 0])

                att_text = pd.DataFrame(att[-len(lstm_text[0]):, :], columns=self.DATASET.DATA["TRAIN_DEV"].sort_values("id_restaurant")["name"].unique())
                att_text.insert(0, "text", text_prepro.split(" "))
                att_text.transpose().to_excel("att_text.xlsx")

        print("-"*30)
        
        # Obtener para la review al completo, el top 3 de restaurantes predichos por el modelo
        if imput_number == 2:
            preds_rst = self.MODEL.predict([lstm_text_pad, np.arange(self.DATASET.DATA["N_RST"])[None,:]])
        else:
            preds_rst = self.MODEL.predict(lstm_text_pad)

        preds_rst_arg = np.argsort(-preds_rst.flatten())[:n_rsts]
        preds_rst_vls = np.sort(-preds_rst.flatten())[:n_rsts]
        for i, rst in enumerate(preds_rst_arg):
            rst_name = self.DATASET.DATA["TRAIN_DEV"].loc[self.DATASET.DATA["TRAIN_DEV"].id_restaurant == rst]["name"].values[0]
            print(f"\t[{-preds_rst_vls[i]:0.2f}] {rst_name}")
      
        '''
        print(f"\tEMB {'-'*50}")

        embs = self.MODEL.get_layer("embedding").get_weights()[0]

        # Para cada una de las palabras de la review, obtener el restaurante que más importancia le da
        for w in lstm_text[0]:
            rsts_ids = np.argsort((-embs[w,:]))[:3]
            rsts_val = -np.sort((-embs[w,:]))[:3]
            restr_wrd = self.DATASET.DATA["TRAIN_DEV"].loc[self.DATASET.DATA["TRAIN_DEV"].id_restaurant.isin(rsts_ids)]["name"].unique().tolist()
            print("\t ·%s => %s " % (list(self.DATASET.DATA["WORD_INDEX"].keys())[w-1], ", ".join(restr_wrd)))
        '''

        '''
        print(f"\tLSTM {'-'*50}")

        for w in lstm_text[0]:

            preds_wrd = self.MODEL.predict(tf.keras.preprocessing.sequence.pad_sequences([[w]], maxlen=self.DATASET.DATA["MAX_LEN_PADDING"])).flatten()
            rsts_ids = np.argsort(-preds_wrd.flatten())[:3]
            restr_wrd = self.DATASET.DATA["TRAIN_DEV"].loc[self.DATASET.DATA["TRAIN_DEV"].id_restaurant.isin(rsts_ids)]["name"].unique().tolist()
            print("\t ·%s => %s " % (list(self.DATASET.DATA["WORD_INDEX"].keys())[w-1], ", ".join(restr_wrd)))
        '''