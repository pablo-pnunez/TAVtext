# -*- coding: utf-8 -*-
import tensorflow_ranking as tfr
import tensorflow as tf

from src.Metrics import f1 as f1_score
from src.models.text_models.att.ATT2ITM import ATT2ITM

class ATT2ITM_2(ATT2ITM):
    """ Con este modelo pretendemos hacer un "ablation test" del modelo propuesto, eliminando algo (tanh?) """

    def __init__(self, config, dataset):
        ATT2ITM.__init__(self, config=config, dataset=dataset)

    def get_model(self):
        model = self.get_sub_model()
        return model

    def get_sub_model(self):

        mv = self.CONFIG["model"]["model_version"]
        self.MODEL_VERSION = mv

        rst_no = self.DATASET.DATA["N_ITEMS"]
        pad_len = self.DATASET.DATA["MAX_LEN_PADDING"]
        vocab_size = self.DATASET.DATA["VOCAB_SIZE"]

        text_in = tf.keras.Input(shape=(pad_len), dtype='int32')
        rest_in = tf.keras.Input(shape=(rst_no), dtype='int32')
        
        model_out = None
        model_out2 = None

        if mv == "0" or mv == "1":  # Modelo básico sin heads, solo una capa oculta

            emb_size = 128

            # init = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
            use_bias = True
            
            # word_importance = tf.keras.layers.Embedding(vocab_size, 1, name="word_importance", embeddings_initializer="ones", mask_zero=True)(text_in)

            query_emb = tf.keras.layers.Embedding(vocab_size, emb_size * 3, mask_zero=True)
            mask_query = query_emb.compute_mask(text_in)
            mask_query = tf.expand_dims(tf.cast(mask_query, dtype=tf.float32), -1)
            mask_query = tf.tile(mask_query, [1, 1, rst_no])
            ht_emb = query_emb(text_in)
            # ht_emb = tf.keras.layers.Activation("tanh")(ht_emb)
            ht_emb = tf.keras.layers.Dense(emb_size * 2, use_bias=use_bias)(ht_emb)
            # ht_emb = tf.keras.layers.Activation("tanh")(ht_emb)
            ht_emb = tf.keras.layers.Dense(emb_size, use_bias=use_bias)(ht_emb)
            # ht_emb = tf.keras.layers.Activation("tanh")(ht_emb)
           
            '''
            wr_att_embs = tf.keras.layers.Embedding(vocab_size, 32, mask_zero=True, name="wr_att_embs")
            wr_att_embs = wr_att_embs(text_in)
            wr_att_embs = tf.keras.layers.Dropout(.8)(wr_att_embs)     
            wr_attention_scores = tf.matmul(wr_att_embs, wr_att_embs, transpose_b=True)
            ht_emb = tf.einsum("abc,acd->abd", wr_attention_scores, ht_emb)
            '''            

            ht_emb = tf.keras.layers.Lambda(lambda x: x, name="word_emb")(ht_emb)

            '''
            position_embeddings = keras_nlp.layers.SinePositionEncoding()(ht_emb)
            ht_emb = ht_emb + position_embeddings
            '''
            
            rests_emb = tf.keras.layers.Embedding(rst_no, emb_size * 3, name=f"in_rsts")
            hr_emb = rests_emb(rest_in)
            hr_emb = tf.keras.layers.Dense(emb_size * 2, use_bias=use_bias)(hr_emb)
            hr_emb = tf.keras.layers.Dense(emb_size, use_bias=use_bias)(hr_emb)
            hr_emb = tf.keras.layers.Lambda(lambda x: x, name="rest_emb")(hr_emb)

            model = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([ht_emb, hr_emb])
            
            model = tf.keras.layers.Lambda(lambda x: x[0]*x[1])([model, mask_query])
            model = tf.keras.layers.Activation("linear", name="dotprod")(model)

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
        
        return model
