# -*- coding: utf-8 -*-
import tensorflow_ranking as tfr
import tensorflow as tf

from src.models.text_models.att.w2v.W2VATT import W2VATT
from src.Metrics import f1 as f1_score


class W2VATT2ITM(W2VATT):
    """ Predecir, a partir de los embeddings W2V de una review y los de los items, el restaurante de la review """

    def __init__(self, config, dataset):
        W2VATT.__init__(self, config=config, dataset=dataset)

    def get_sub_model(self):

        mv = self.CONFIG["model"]["model_version"]
        self.MODEL_VERSION = mv

        rst_no = self.DATASET.DATA["N_ITEMS"]
        pad_len = self.DATASET.DATA["MAX_LEN_PADDING"]
        vocab_size = self.DATASET.DATA["VOCAB_SIZE"]

        emb_size = 128
        w2v_size = emb_size*3

        # Entrenar un W2V y obtener embeddings
        w2v_embeddings = self.get_w2v_embeddings(w2v_size)

        text_in = tf.keras.Input(shape=(pad_len), dtype='int32')
        rest_in = tf.keras.Input(shape=(rst_no), dtype='int32')
        
        model = None

        if mv == "0": # Se añade la estandarización a la salida y se simplifica el modelo
           
            query_emb = tf.keras.layers.Embedding(w2v_embeddings.shape[0], w2v_size, weights=[w2v_embeddings],  
                                                  mask_zero=True, name="all_words", trainable=False)
            mask_query = tf.cast(tf.math.not_equal(text_in, 0), tf.float32) # Se obtiene la máscara del texto para un solo item
            mask_query = tf.tile(tf.expand_dims(mask_query, axis=-1),[1,1,rst_no]) # Se repite para todos los items

            ht_emb = query_emb(text_in)
            ht_emb = tf.keras.layers.Dense(emb_size * 2)(ht_emb)
            ht_emb = tf.keras.layers.Dense(emb_size)(ht_emb)

            ht_emb = tf.keras.layers.Lambda(lambda x: x, name="word_emb")(ht_emb)

            rests_emb = tf.keras.layers.Embedding(rst_no, emb_size * 3, name=f"in_rsts")
            hr_emb = rests_emb(rest_in)
            hr_emb = tf.keras.layers.Dense(emb_size * 2)(hr_emb)
            hr_emb = tf.keras.layers.Dense(emb_size)(hr_emb)
            hr_emb = tf.keras.layers.Lambda(lambda x: x, name="rest_emb")(hr_emb)
            model = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([ht_emb, hr_emb])           
            model = tf.keras.layers.Lambda(lambda x: x[0]*x[1])([model, mask_query])
            model = tf.keras.layers.Activation("tanh", name="dotprod")(model)

            model = tf.keras.layers.Dropout(.4)(model)

            model = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1), name="sum")(model)
            model_out = tf.keras.layers.Activation("sigmoid", name="out", dtype='float32')(model)

            model = tf.keras.models.Model(inputs=[text_in, rest_in], outputs=[model_out], name=f"{self.MODEL_NAME}_{self.MODEL_VERSION}")

            optimizer = tf.keras.optimizers.legacy.Adam(self.CONFIG["model"]["learning_rate"])

            metrics = [tfr.keras.metrics.NDCGMetric(topn=10, name="NDCG@10"), tfr.keras.metrics.RecallMetric(topn=1, name='r1'), tfr.keras.metrics.RecallMetric(topn=5, name='r5'), tfr.keras.metrics.RecallMetric(topn=10, name='r10'),
                       tfr.keras.metrics.PrecisionMetric(topn=5, name='p5'), tfr.keras.metrics.PrecisionMetric(topn=10, name='p10'), f1_score]

            if mv == "0":
                loss = tf.keras.losses.CategoricalCrossentropy()  # Prueba
            elif mv == "1":
                loss = tf.keras.losses.BinaryFocalCrossentropy()

        model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

        print(model.summary())

        return model

    def __create_tfdata__(self, dataframe):

        seq_data = self.DATASET.DATA["TEXT_SEQUENCES"][dataframe.seq.values]
        rst_data = dataframe.id_item.values
        
        data_y = tf.data.Dataset.from_tensor_slices(rst_data)
        data_y = data_y.map(lambda x: tf.one_hot(x, self.DATASET.DATA["N_ITEMS"]), num_parallel_calls=tf.data.AUTOTUNE)
        
        data_x_text = tf.data.Dataset.from_tensor_slices(seq_data)
        data_x_item = tf.data.Dataset.from_tensor_slices([range(self.DATASET.DATA["N_ITEMS"])]).repeat(len(dataframe))
        data_x = tf.data.Dataset.zip((data_x_text, data_x_item))

        return tf.data.Dataset.zip((data_x, data_y))
