# -*- coding: utf-8 -*-
import tensorflow_ranking as tfr
import tensorflow as tf

from src.models.text_models.att.w2v.W2VATT import W2VATT


class W2VATT2VAL(W2VATT):
    """ Predecir, a partir de los embeddings W2V de una review y los de los items, el restaurante de la review """

    def __init__(self, config, dataset):
        W2VATT.__init__(self, config=config, dataset=dataset)

    def get_sub_model(self):

        mv = self.CONFIG["model"]["model_version"]
        self.MODEL_VERSION = mv

        rst_no = self.DATASET.DATA["N_ITEMS"]
        pad_len = self.DATASET.DATA["MAX_LEN_PADDING"]
        vocab_size = self.DATASET.DATA["VOCAB_SIZE"]

        emb_size = 256  # 128

        # Entrenar un W2V y obtener embeddings
        w2v_embeddings = self.get_w2v_embeddings(emb_size)

        text_in = tf.keras.Input(shape=(pad_len), dtype='int32')
        rest_in = tf.keras.Input(shape=(rst_no), dtype='int32')
        
        model = None

        if mv == "0": # Se añade la estandarización a la salida y se simplifica el modelo
            dropout = .1

            # TODO: DIFENRENCIA ENTRE VOCABSIZE y VOCABULARIO DEL TOKENIZADOR
            
            query_emb = tf.keras.layers.Embedding(w2v_embeddings.shape[0], emb_size, weights=[w2v_embeddings],  
                                                  mask_zero=True, name="all_words", trainable=False)
            mask_query = tf.cast(tf.math.not_equal(text_in, 0), tf.float32) # Se obtiene la máscara del texto para un solo item
            mask_query = tf.tile(tf.expand_dims(mask_query, axis=-1),[1,1,rst_no]) # Se repite para todos los items

            ht_emb = query_emb(text_in)
            # ht_emb = tf.keras.layers.Activation("tanh")(ht_emb)
            # ht_emb = tf.keras.layers.Dense(emb_size)(ht_emb)

            ht_emb = tf.keras.layers.Lambda(lambda x: x, name="word_emb")(ht_emb)
            ht_emb = tf.keras.layers.Dropout(dropout)(ht_emb)

            items_emb = tf.keras.layers.Embedding(rst_no, emb_size, name="all_items")
            hr_emb = items_emb(rest_in)

            hr_emb = tf.keras.layers.Lambda(lambda x: x, name="rest_emb")(hr_emb)
            hr_emb = tf.keras.layers.Dropout(dropout)(hr_emb)

            model = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True), name="dot_mul")([ht_emb, hr_emb])
            model = tf.keras.layers.Lambda(lambda x: x[0] * x[1] , name="dot_mask")([model, mask_query])
            model = tf.keras.layers.Activation("tanh", name="dotprod")(model)
            model = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, 1), name="sum")(model)

            model = tf.keras.layers.Activation(self.custom_activation_smoothstep)(model) # Sigmoide entre 1 y 5                                 
            # model = tf.keras.layers.Activation(self.custom_activation_sigmoid)(model) # Sigmoide entre 1 y 5   
            # model = tf.keras.layers.ReLU(max_value=5)(model)
                              
            model_out = tf.keras.layers.Activation("linear", name="out", dtype='float32')(model)
                        
            model = tf.keras.models.Model(inputs=[text_in, rest_in], outputs=[model_out], name=f"{self.MODEL_NAME}_{self.MODEL_VERSION}")
            optimizer = tf.keras.optimizers.legacy.Adam(self.CONFIG["model"]["learning_rate"])

            metrics = [tfr.keras.metrics.NDCGMetric(topn=10, name="NDCG@10"), tf.keras.metrics.MeanAbsoluteError(name="MAE"),
                       tfr.keras.metrics.RecallMetric(topn=5, name='RC@5'), tfr.keras.metrics.RecallMetric(topn=10, name='RC@10')]
                     
        model.compile(loss=self.custom_loss, metrics=metrics, optimizer=optimizer)

        print(model.summary())

        return model

   