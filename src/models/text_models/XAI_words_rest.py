import pandas as pd
import tensorflow as tf
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import seaborn as sns

import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.preprocessing import normalize
from gensim.models import KeyedVectors

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


def plot_roc(name, labels, predictions, **kwargs):
  fp, tp, _ = roc_curve(labels, predictions)

  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  #plt.xlim([-0.5, 20])
  #plt.ylim([80, 100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')
  # plt.show()


def pinta_ROC(train_labels, train_predictions, test_labels, test_predictions, colors):
    plot_roc("Train", train_labels, train_predictions, color=colors[0])
    plot_roc("Test", test_labels, test_predictions, color=colors[0], linestyle='--')
    plt.legend(loc='lower right')
    plt.show()


def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5, 5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  plt.show()

  # print('dislike como dislike (True Negatives): ', cm[0][0])
  # print('dislike como like (fallo) (False Positives): ', cm[0][1])
  # print('like como dislike (fallo) (False Negatives): ', cm[1][0])
  # print('like como like (True Positives): ', cm[1][1])
  # print('Total dislike: ', np.sum(cm[0]))
  # print('Total like: ', np.sum(cm[1]))


def plot_metric(history, metric, valida):
    train_metrics = history.history[metric]
    if valida:
        val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    if valida:
        plt.plot(epochs, val_metrics)
        plt.title('Training and validation ' + metric)
    else:
        plt.title('Training ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    if valida:
        plt.legend(["train_"+metric, 'val_' + metric])
    plt.show()


def reparte_val_train_test(valoraciones, ratioTest):
    # lo calculo otra vez para que solo aparezcan los que pasaron el filtro
    rev_por_usr = pd.to_numeric(valoraciones['rating']).groupby(valoraciones['userId']).count().sort_values()
    para_test = np.ceil(rev_por_usr * ratioTest).astype('int')
    train = []
    test = []
    for i in range(para_test.shape[0]):
        iduser = para_test.index[i]
        user_rev = valoraciones[valoraciones['userId'] == iduser]
        # en para_test[i] se indica el número de valoraciones que deben ir al test
        test.extend(user_rev.iloc[0:para_test[i]].values.tolist())
        train.extend(user_rev.iloc[para_test[i]:].values.tolist())
    val_train = pd.DataFrame(train, columns=valoraciones.columns)
    val_test = pd.DataFrame(test, columns=valoraciones.columns)
    return val_train, val_test


def crear_modelo(LR, num_rest, dim_rest):
    rest_input = tf.keras.layers.Input(shape=(dim_rest,), name="input_rest")
    output = tf.keras.layers.Dense(num_rest, activation='softmax', name="output_layer")(rest_input)

    model = tf.keras.Model(
        inputs=rest_input,
        outputs=output,  # , genre_pred],
    )
    model.summary()

    metrics = [
        # tf.keras.metrics.Accuracy(name='accuracy'),
        'accuracy',
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10'),
    ]  # si utilizo este 'metrics' con tf.keras.metrics.Accuracy(name='accuracy'), no me calcula bien la accuracy ¿?

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=metrics,
        # metrics=['accuracy', 'top_k_categorical_accuracy'],  # metrics,
    )

    return model  # , model_emb


def entrenar(VALIDATION, epochs, batch_size, learning_rate, num_users, num_rest, dim_input_users, dim_input_rest,
             X_train, X_train_dev, X_test, y_train, y_train_dev, y_test):

    model = crear_modelo(learning_rate, num_rest, dim_input_rest)  # , X)

    # ENTRENAMOS EL MODELO Y EVALUAMOS EN EL CONJUNTO DE TEST
    if VALIDATION:
        # DECLARAMOS UN OBJETO EARLY STOPPING PARA DECIDIR LA MEJOR EPOCH
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        #early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=20)

        # checkpoint = ModelCheckpoint('model/model_T_{epoch:06d}.h5', save_best_only=True)
        history = model.fit(x=X_train[nombre_feat], y=y_train, validation_data=(X_dev[nombre_feat], y_dev),
                            epochs=epochs, batch_size=batch_size, callbacks=[early_stop])  # , checkpoint])
    else:
        history = model.fit(x=X_train_dev[nombre_feat], y=y_train_dev, epochs=epochs, batch_size=batch_size)


    losses = model.evaluate(X_test[nombre_feat], y_test)

    print('*-* Test CategoricalCrossentropy:', losses[0])
    print('*-* Test Accuracy:', losses[1])
    print('*-* Test Top5CategoricalAccuracy:', losses[2])
    print('*-* Test Top10CategoricalAccuracy:', losses[3])


    if VALIDATION:
        if early_stop.stopped_epoch == 0:
            print('Para por número de iteraciones:', max_epochs)
            total_epochs = max_epochs
        else:
            print('Mejor epoch:', early_stop.stopped_epoch)
            total_epochs = early_stop.stopped_epoch

    prediction = model.predict(X_test[nombre_feat])

    # se calcula la popularidad para que actúe como baseline
    # y se convierte a probabilidad
    if VALIDATION:
        popularidad = y_train.sum(axis=0) / sum(y_train.sum(axis=0))
    else:
        popularidad = y_train_dev.sum(axis=0) / sum(y_train_dev.sum(axis=0))
    pred_popularidad = np.matlib.repmat(popularidad, y_test.shape[0], 1)

    # kk = np.array((y_test.values.argmax(axis=1), pred_popularidad.argmax(axis=1))).transpose()
    # pd.DataFrame(kk).to_csv('KK.csv')
    # kk2 = np.array((y_test.values.argmax(axis=1), prediction.argmax(axis=1))).transpose()

    m = tf.keras.metrics.Accuracy()
    m.update_state(y_true=y_test.values.argmax(axis=1), y_pred=pred_popularidad.argmax(axis=1))
    print("ACCURACY por popularidad: %.4f" % (m.result().numpy()))
    m.reset_states()
    m.update_state(y_true=y_test.values.argmax(axis=1), y_pred=prediction.argmax(axis=1))
    print("ACCURACY nuestro sistema: %.4f" % (m.result().numpy()))

    m = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
    m.update_state(y_true=y_test, y_pred=pred_popularidad)
    print("TOP5ACC por popularidad:  %.4f" % (m.result().numpy()))
    m.reset_states()
    m.update_state(y_true=y_test, y_pred=prediction)
    print("TOP5ACC nuestro sistema:  %.4F" % (m.result().numpy()))

    m = tf.keras.metrics.TopKCategoricalAccuracy(k=10)
    m.update_state(y_true=y_test, y_pred=pred_popularidad)
    print("TOP10ACC por popularidad: %.4f" % (m.result().numpy()))
    m.reset_states()
    m.update_state(y_true=y_test, y_pred=prediction)
    print("TOP10ACC nuestro sistema: %.4f" % (m.result().numpy()))


    # GRÁFICOS
    plot_metric(history, 'loss', valida=VALIDATION)
    plot_metric(history, 'accuracy', valida=VALIDATION)
    plot_metric(history, 'top_5', valida=VALIDATION)
    plot_metric(history, 'top_10', valida=VALIDATION)

    if VALIDATION:
        return model, total_epochs
    else:
        return model, prediction


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
FILTRO = 1  # los usuarios deben tener al menos este número de reviews
# ---------------------------------------------------------------------------------------------------------------------
PESOS = False  # por si se quiere dar más peso a la clase 0 (dislike)
# ---------------------------------------------------------------------------------------------------------------------
ONLY_FROM_TEXT = False  # falso => (rest, user)->nota      true => texto->nota
# ---------------------------------------------------------------------------------------------------------------------
# ONEHOT_REST = False  # [ONLY_FROM_TEXT==False (rest, user)->nota]  true=>one-hot    false=>vector de 200 palabras
# ---------------------------------------------------------------------------------------------------------------------
USING_LSTM = False  # [ONLY_FROM_TEXT==True texto->nota] true => LSTM   false => BOW
# ---------------------------------------------------------------------------------------------------------------------

valoraciones = pd.read_pickle('../../Reviews/rest_textos_ratings/gijon_con_usrid.pkl')
print('Número de reseñas inicial:', valoraciones.shape[0])

# me quedo con las valoraciones sobre los restaurantes con más de 100 reseñas
# se cargan los vectores que representan a los restaurantes
rest_features = pd.read_pickle('../rest_freq_100.pkl')
ids_rest = rest_features.index
valoraciones = valoraciones[valoraciones['restaurantId'].isin(ids_rest)]
print('Nos quedamos con las reseñas de los restaurantes que tienen más de 100 reseñas')
print('Número de restaurantes:', ids_rest.shape[0])
print('Número de reseñas:', valoraciones.shape[0])

# para comprobar que tienen más de 100 reseñas
# print(valoraciones.groupby('restaurantId').count().reset_index().sort_values('rating'))
# print(pd.to_numeric(valoraciones['rating']).groupby(valoraciones['restaurantId']).count().sort_values())

# para ver cuántas reviews tienen los usuarios
rev_por_usr = pd.to_numeric(valoraciones['rating']).groupby(valoraciones['userId']).count().sort_values()
idx_usr = rev_por_usr[rev_por_usr >= FILTRO].index
idx_usr = idx_usr[idx_usr != '']  # elimino índices vacíos y quedan 383 usuarios
print('Nos quedamos con los usuarios que tienen al menos', FILTRO, 'reseñas')
print('Número de usuarios:', idx_usr.shape[0])

valoraciones = valoraciones[valoraciones['userId'].isin(idx_usr)]
print('Número de reseñas:', valoraciones.shape[0])


XTEXT = 'text'  # 'title' para el título de la reseña y 'text' para la reseña
if ONLY_FROM_TEXT:

    # quitamos nulos si los hubiese
    valoraciones = valoraciones.loc[valoraciones[XTEXT].notnull(), :]

    if USING_LSTM:
        # Para entrenar con 'text' hay que poner TRUNCAR_PARA_PADDING=True
        # Para entrenar con 'title' hay que poner TRUNCAR_PARA_PADDING=False
        TRUNCAR_PARA_PADDING = True
        learning_rate = 1e-3
        BATCH_SIZE = 1024
        INITIAL_EPOCH = 0

        # NOS QUEDAMOS SOLO CON LAS REVIEWS CON TEXTOS o TITLES CON NO MÁS DE 2000 CARACTERES (se elimina en torno al 1%)
        valoraciones['num_car'] = valoraciones[XTEXT].map(len)
        # n = 2000
        # print(">", n, "=", len(valoraciones[valoraciones.num_car > n]), "-",
        #       len(valoraciones[valoraciones.num_car > n]) * 100 / len(valoraciones), "%")
        valoraciones = valoraciones.loc[valoraciones.num_car <= 2000, ['title', 'text', 'rating', 'restaurantId', 'userId']]
        print("Quedándonos con reviews de hasta 2000 car, resultan", valoraciones.shape, "ejemplos")

        # quitamos lo que no son caracteres, dígitos o espacios en blanco
        print('Quitando lo que no son caracteres, dígitos o espacios en blanco...', end=" ")
        valoraciones[XTEXT] = valoraciones[XTEXT].apply(lambda x: re.sub('[^\w\s]', '', x).strip())
        print('done!')

        # X / y
        X = valoraciones[XTEXT].copy()
        y = valoraciones['rating'].copy()

        # TOKENIZAMOS LAS PALABRAS (ASOCIA CADA PALABRA A UN ÍNDICE (WORD_INDEX))
        tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True)
        tokenizer.fit_on_texts(X)

        # GUARDAMOS EL MAPEO ENTRE PALABRA E ÍNDICE (ejemplo 'de': 2)
        word_index = tokenizer.word_index
        print("*** Intervienen", len(word_index), "palabras y utilizamos todas ***")

        # TRAIN/DEV/TEST SPLIT (sin estratificar para prevenir errores)
        TEST = 0.10
        DEV = 0.10
        Xt_train_dev, Xt_test, y_train_dev, y_test = train_test_split(X, y, stratify=None, test_size=TEST,
                                                                      random_state=2032)
        Xt_train, Xt_dev, y_train, y_dev = train_test_split(Xt_train_dev, y_train_dev, stratify=None,
                                                            test_size=DEV, random_state=2032)

        # TRANSFORMAMOS LAS FRASES A SECUENCIAS DE NÚMEROS SEGÚN EL MAPEO DE WORD_INDEX
        sequences_train = tokenizer.texts_to_sequences(Xt_train)
        sequences_dev = tokenizer.texts_to_sequences(Xt_dev)
        sequences_test = tokenizer.texts_to_sequences(Xt_test)

        MULT = 3
        if TRUNCAR_PARA_PADDING:
            # CALCULAMOS LA MÁXIMA LONGITUD EN PADDING (MEDIA+STD) SE PIERDEN PALABRAS EN ALGUNAS REVIEWS
            long_texts = []
            for l in sequences_train:
                long_texts.append(len(l))
            max_len_padding = int(np.mean(long_texts) + MULT * np.std(long_texts))
            print("media:", np.mean(long_texts), "\nsdt:", np.std(long_texts))
            print("Padding limitado a:", max_len_padding, "palabras")
        else:
            max_len_padding = None

        # PADDING CON 0S A LA IZQUIERDA. ENTRADAS DEL MODELO
        X_train = tf.keras.preprocessing.sequence.pad_sequences(sequences_train, maxlen=max_len_padding)
        X_dev = tf.keras.preprocessing.sequence.pad_sequences(sequences_dev, maxlen=max_len_padding)
        X_test = tf.keras.preprocessing.sequence.pad_sequences(sequences_test, maxlen=max_len_padding)

        print("% test =", TEST, "\n% dev =", DEV)
        print("       (num_ejemplos, num_palabras)")
        print("train:", X_train.shape)
        print("dev  :", X_dev.shape)
        print("test :", X_test.shape)

        # DIMENSIONES A TENER EN CUENTA
        max_length = max(X_train.shape[1], X_dev.shape[1],
                         X_test.shape[1])  # Cogemos la dimension máxima (la frase más larga)
        vocab_size = len(word_index) + 1  # El +1 se utiliza para las palabras que no estén en el Word2Vec (saco roto)
        EMBEDDING_DIM = 300  # Output de la capa Embedding (el dado por el modelo descargado)
        output_n = 1  # problema de regresión (predecir el rating)

        # CARGAMOS EL MODELO WORD2VEC PREENTRENADO
        EMB_PATH_PROPIO_CON_TILDES = '../../word2vec/Word2Vec_entrenado/word2vec-300-min20-w5.txt.bz2'
        word_vectors = KeyedVectors.load_word2vec_format(EMB_PATH_PROPIO_CON_TILDES)  # Word2Vec propio con tildes

        # INICIALIZAMOS LA MATRIZ EMBEDDING CON 0s
        embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
        print("embedding_matrix shape =", embedding_matrix.shape)

        # RELLENO LA EMBEDDING_MATRIX CON LOS VECTORES CORRESPONDIENTES
        for word, i in word_index.items():
            try:
                embedding_vector = word_vectors[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                continue

        # BORRAMOS EL MODELO PARA AHORRAR MEMORIA
        del word_vectors

        # ARQUITECTURA DE MODELO
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM,
                                            weights=[embedding_matrix], trainable=False, mask_zero=True))
        model.add(tf.keras.layers.LSTM(128))  # 256
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(output_n))
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                      metrics=['mean_absolute_error'])
        model.summary()
        print(" ** Learnig rate:", model.optimizer._hyper['learning_rate'], "**")

        # DECLARAMOS UN OBJETO EARLY STOPPING PARA DECIDIR LA MEJOR EPOCH
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        # ENTRENAMOS EL MODELO Y EVALUAMOS EN EL CONJUNTO DE TEST
        # checkpoint = ModelCheckpoint(SAVE_PATH + 'model_' + CIUDAD + '_{epoch:06d}.h5', save_best_only=True)
        history = model.fit(X_train, tf.convert_to_tensor(y_train, np.int32), initial_epoch=INITIAL_EPOCH,
                            validation_data=(X_dev, tf.convert_to_tensor(y_dev, np.int32)),
                            batch_size=BATCH_SIZE, epochs=100, callbacks=[early_stop])  #, checkpoint])
        loss, acc = model.evaluate(X_test, tf.convert_to_tensor(y_test, np.int32))
        print('Test mean_squared_error:', loss)
        print('Test mean_absolute_error:', acc)
        print('Test mean_absolute_error prediciendo la MEDIA:', sum(abs(y_train.mean() - y_test)) / y_test.shape[0])
        print('Mejor epoch:', early_stop.stopped_epoch)

        # todo: se debería entrenar [train+dev] -> [test]

        prediction = model.predict(X_test)
        print('y_train     (media y desviación: %.2f+-%.2f' % (y_train.mean(), y_train.std()))
        print('y_test      (media y desviación: %.2f+-%.2f' % (y_test.mean(), y_test.std()))
        print('prediction  (media y desviación: %.2f+-%.2f' % (np.mean(prediction), np.std(prediction)))

        # GRÁFICOS
        plot_metric(history, 'loss', valida=True)
        plot_metric(history, 'mean_absolute_error', valida=True)
    else:
        print("mediante BOW")

        NUM_PALABRAS = 200
        learning_rate = 1e-3
        BATCH_SIZE = 256

        # GENERAMOS BOW
        print('Vectorizando...', end=" ")
        spanish_stopwords = stopwords.words('spanish')
        spanish_stopwords += ['además', 'allí', 'aquí', 'asturias', 'así', 'aunque', 'años', 'cada', 'casa', 'casi',
                              'comido', 'comimos', 'cosas', 'creo', 'decir', 'después', 'dos', 'día', 'fin', 'gijon',
                              'gijón', 'hace', 'hacer', 'hora', 'ido', 'igual', 'ir', 'lado', 'luego', 'mas', 'merece',
                              'mismo', 'momento', 'mucha', 'muchas', 'parece', 'parte', 'pedimos', 'pedir', 'probar',
                              'puede', 'puedes', 'pues', 'punto', 'relación', 'reservar', 'seguro', 'semana', 'ser',
                              'si',
                              'sido', 'siempre', 'sitio', 'sitios', 'solo', 'sí', 'tan', 'tener', 'toda', 'tomar',
                              'tres',
                              'unas', 'varias', 'veces', 'ver', 'verdad', 'vez', 'visita', 'bastante', 'duda', 'gran',
                              'menos', 'no', 'nunca', 'opinión', 'primera', 'primero', 'segundo', '10', 'mejor',
                              'mejores']
        spanish_stopwords += ['100', '15', '20', '30', 'alguna', 'asturiana', 'caso', 'centro', 'cierto', 'comentario',
                              'cosa',
                              'cualquier', 'cuanto', 'cuenta', 'da', 'decidimos', 'demasiado', 'dentro', 'destacar',
                              'detalle',
                              'dia', 'días', 'esperamos', 'esperar', 'general', 'gracias', 'haber', 'hacen', 'hecho',
                              'lleno',
                              'media', 'minutos', 'noche', 'nota', 'poder', 'ponen', 'probado', 'puedo', 'reserva',
                              'resto',
                              'sabor', 'sólo', 'tiempo', 'todas', 'tomamos', 'totalmente', 'vamos', 'varios', 'vida',
                              'único']
        spanish_stopwords += ['50', 'ahora', 'aún', 'cerca', 'ciudad', 'cuatro', 'elegir', 'encima', 'falta', 'final',
                              'ganas',
                              'hoy', 'llegamos', 'medio', 'mundo', 'nuevo', 'ocasiones', 'opción', 'pareció', 'pasar',
                              'pedido',
                              'pesar', 'poner', 'probamos', 'pronto', 'realmente', 'salimos', 'sirven', 'situado',
                              'tampoco',
                              'tarde', 'tipo', 'va', 'vas', 'voy']
        spanish_stopwords += ['12', 'come', 'demás', 'ello', 'etc', 'incluso', 'llegar', 'pasado', 'primer', 'pusieron',
                              'quedamos', 'quieres', 'saludo', 'tambien', 'trabajo', 'tras', 'verano']
        spanish_stopwords += ['algún', 'cenamos', 'comentarios', 'comiendo', 'dan', 'dice', 'domingo', 'ofrecen',
                              'razonable',
                              'tamaño']
        spanish_stopwords += ['nadie', 'ningún', 'opiniones', 'quizás', 'san', 'sino']
        spanish_stopwords += ['atendió', 'pega', 'sábado']
        spanish_stopwords += ['dicho', 'par', 'total']
        vectorizer = CountVectorizer(stop_words=spanish_stopwords, min_df=5, max_features=NUM_PALABRAS,
                                     binary=False)  # Frecuencia
        bow = vectorizer.fit_transform(valoraciones[XTEXT])
        print('done!')

        # X / y
        X = bow  # valoraciones[XTEXT].copy()
        y = valoraciones['rating'].copy()

        # TRAIN/DEV/TEST SPLIT (sin estratificar para prevenir errores)
        TEST = 0.10
        DEV = 0.10
        X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, stratify=None, test_size=TEST,
                                                                      random_state=2032)
        X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, stratify=None,
                                                            test_size=DEV, random_state=2032)

        # ARQUITECTURA DE MODELO
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(64, input_dim=NUM_PALABRAS, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                      metrics=['mean_absolute_error'])
        model.summary()
        print(" ** Learnig rate:", model.optimizer._hyper['learning_rate'], "**")

        # DECLARAMOS UN OBJETO EARLY STOPPING PARA DECIDIR LA MEJOR EPOCH
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        # ENTRENAMOS EL MODELO Y EVALUAMOS EN EL CONJUNTO DE TEST
        # checkpoint = ModelCheckpoint(SAVE_PATH + 'model_' + CIUDAD + '_{epoch:06d}.h5', save_best_only=True)
        history = model.fit(X_train.todense(), tf.convert_to_tensor(y_train, np.int32), initial_epoch=0,
                            validation_data=(X_dev.todense(), tf.convert_to_tensor(y_dev, np.int32)),
                            batch_size=BATCH_SIZE, epochs=100, callbacks=[early_stop])  # , checkpoint])
        loss, acc = model.evaluate(X_test.todense(), tf.convert_to_tensor(y_test, np.int32))
        print('Test mean_squared_error:', loss)
        print('Test mean_absolute_error:', acc)
        print('Test mean_absolute_error prediciendo la MEDIA:', sum(abs(y_train.mean() - y_test)) / y_test.shape[0])
        print('Mejor epoch:', early_stop.stopped_epoch)

        # todo: se debería entrenar [train+dev] -> [test]

        prediction = model.predict(X_test.todense())
        print('y_train     (media y desviación: %.2f+-%.2f' % (y_train.mean(), y_train.std()))
        print('y_test      (media y desviación: %.2f+-%.2f' % (y_test.mean(), y_test.std()))
        print('prediction  (media y desviación: %.2f+-%.2f' % (np.mean(prediction), np.std(prediction)))

        # GRÁFICOS
        plot_metric(history, 'loss', valida=True)
        plot_metric(history, 'mean_absolute_error', valida=True)

else:
    print("Se calcula el BOW de cada review")

    NUM_PALABRAS = 200

    # GENERAMOS BOW
    print('Vectorizando...', end=" ")
    spanish_stopwords = stopwords.words('spanish')
    spanish_stopwords += ['además', 'allí', 'aquí', 'asturias', 'así', 'aunque', 'años', 'cada', 'casa', 'casi',
                          'comido', 'comimos', 'cosas', 'creo', 'decir', 'después', 'dos', 'día', 'fin', 'gijon',
                          'gijón', 'hace', 'hacer', 'hora', 'ido', 'igual', 'ir', 'lado', 'luego', 'mas', 'merece',
                          'mismo', 'momento', 'mucha', 'muchas', 'parece', 'parte', 'pedimos', 'pedir', 'probar',
                          'puede', 'puedes', 'pues', 'punto', 'relación', 'reservar', 'seguro', 'semana', 'ser',
                          'si',
                          'sido', 'siempre', 'sitio', 'sitios', 'solo', 'sí', 'tan', 'tener', 'toda', 'tomar',
                          'tres',
                          'unas', 'varias', 'veces', 'ver', 'verdad', 'vez', 'visita', 'bastante', 'duda', 'gran',
                          'menos', 'no', 'nunca', 'opinión', 'primera', 'primero', 'segundo', '10', 'mejor',
                          'mejores']
    spanish_stopwords += ['100', '15', '20', '30', 'alguna', 'asturiana', 'caso', 'centro', 'cierto', 'comentario',
                          'cosa',
                          'cualquier', 'cuanto', 'cuenta', 'da', 'decidimos', 'demasiado', 'dentro', 'destacar',
                          'detalle',
                          'dia', 'días', 'esperamos', 'esperar', 'general', 'gracias', 'haber', 'hacen', 'hecho',
                          'lleno',
                          'media', 'minutos', 'noche', 'nota', 'poder', 'ponen', 'probado', 'puedo', 'reserva',
                          'resto',
                          'sabor', 'sólo', 'tiempo', 'todas', 'tomamos', 'totalmente', 'vamos', 'varios', 'vida',
                          'único']
    spanish_stopwords += ['50', 'ahora', 'aún', 'cerca', 'ciudad', 'cuatro', 'elegir', 'encima', 'falta', 'final',
                          'ganas',
                          'hoy', 'llegamos', 'medio', 'mundo', 'nuevo', 'ocasiones', 'opción', 'pareció', 'pasar',
                          'pedido',
                          'pesar', 'poner', 'probamos', 'pronto', 'realmente', 'salimos', 'sirven', 'situado',
                          'tampoco',
                          'tarde', 'tipo', 'va', 'vas', 'voy']
    spanish_stopwords += ['12', 'come', 'demás', 'ello', 'etc', 'incluso', 'llegar', 'pasado', 'primer', 'pusieron',
                          'quedamos', 'quieres', 'saludo', 'tambien', 'trabajo', 'tras', 'verano']
    spanish_stopwords += ['algún', 'cenamos', 'comentarios', 'comiendo', 'dan', 'dice', 'domingo', 'ofrecen',
                          'razonable',
                          'tamaño']
    spanish_stopwords += ['nadie', 'ningún', 'opiniones', 'quizás', 'san', 'sino']
    spanish_stopwords += ['atendió', 'pega', 'sábado']
    spanish_stopwords += ['dicho', 'par', 'total']
    vectorizer = CountVectorizer(stop_words=spanish_stopwords, min_df=5, max_features=NUM_PALABRAS,
                                 binary=False)  # Frecuencia

    bow = vectorizer.fit_transform(valoraciones[XTEXT])
    nombre_feat = sorted(vectorizer.vocabulary_)  # se ordenan según aparecen en la matriz bow

    # SE NORMALIZA EL VECTOR DE CADA review
    values = bow.todense()
    normed_bow = normalize(values, axis=1, norm='l1')
    # SE INCORPORAN LAS CODIFICACIONES BOW DE LOS TEXTOS EN valoraciones
    valoraciones[nombre_feat] = normed_bow

    print('done!')
    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    # AÑADIENDO ONE-HOT Y FEATURES DE LOS RESTAURANTES
    # ---------------------------------------------------------------------------------------------------------------------
    # asigno un id numérico a los usuarios empezando por el 0 (para el one-hot)
    print('Asignando ids numéricos a los usuarios...')
    valoraciones['userIdint'] = -np.ones((valoraciones.shape[0], 1), np.int)
    for i in range(idx_usr.shape[0]):
        valoraciones.loc[valoraciones['userId'] == idx_usr[i], 'userIdint'] = i
        if i % 300 == 0:
            print('user', i)

    # añado al dataframe la descrippción de cada restaurante para cada valoración y
    # asigno un id numérico a los restaurantes empezando por el 0 (para el one-hot si quiero utilizarlo)
    rest_features = rest_features.reset_index()
    # nombre_feat = rest_features.columns.drop(['restaurantId', 'nombre'])
    # valoraciones[nombre_feat] = np.zeros((valoraciones.shape[0], nombre_feat.shape[0]))
    valoraciones['restaurantIdint'] = -np.ones((valoraciones.shape[0], 1), np.int)
    id_rest_int = 0
    for id in rest_features['restaurantId']:
        # print('rest:', id)
        # feat = rest_features.loc[rest_features['restaurantId'] == id, nombre_feat]
        mask = valoraciones['restaurantId'] == id
        # valoraciones.loc[mask, nombre_feat] = np.matlib.repmat(feat, np.sum(mask), 1)
        valoraciones.loc[mask, 'restaurantIdint'] = id_rest_int
        if id_rest_int % 30 == 0:
            print('restaurantes procesados:', id_rest_int)
        id_rest_int += 1

    # AÑADIENDO RESTAURANTE ONE-HOT
    rest_oh = tf.keras.utils.to_categorical(valoraciones['restaurantIdint'])
    nombre_clases = []
    for i in range(valoraciones['restaurantIdint'].unique().shape[0]):
        nombre_clases.append('r'+str(i))
    valoraciones[nombre_clases] = rest_oh

    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    # SEPARANDO TRAIN-DEV-TEST
    # ---------------------------------------------------------------------------------------------------------------------

    # X / y
    X = valoraciones[np.concatenate((nombre_feat, ['restaurantIdint', 'userIdint']))].copy()
    y = valoraciones[nombre_clases].copy()
    # TRAIN/DEV/TEST SPLIT (sin estratificar para prevenir errores)
    TEST = 0.10
    DEV = 0.10
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, stratify=None, test_size=TEST,
                                                                  random_state=2032)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, stratify=None,
                                                        test_size=DEV, random_state=2032)

    '''# se barajan las valoraciones
    valoraciones = valoraciones.sample(frac=1, axis=0, random_state=2032)
    TEST = 0.10
    DEV = 0.10
    print('Separando [train, dev] - [test]...')
    val_train_dev, val_test = reparte_val_train_test(valoraciones, TEST)
    print('done!\nSeparando [train] - [dev]...')
    val_train, val_dev = reparte_val_train_test(val_train_dev, DEV)
    print('done!\n')

    X_train_dev = val_train_dev[np.concatenate((nombre_feat, ['restaurantIdint', 'userIdint']))]
    y_train_dev = val_train_dev[nombre_clases]

    X_train = val_train[np.concatenate((nombre_feat, ['restaurantIdint', 'userIdint']))]
    y_train = val_train[nombre_clases]

    X_dev = val_dev[np.concatenate((nombre_feat, ['restaurantIdint', 'userIdint']))]
    y_dev = val_dev[nombre_clases]

    X_test = val_test[np.concatenate((nombre_feat, ['restaurantIdint', 'userIdint']))]
    y_test = val_test[nombre_clases]'''


    print('num ejemplos en TRAIN:', y_train.shape[0])
    print('num ejemplos en DEV  :', y_dev.shape[0])
    print('num ejemplos en TEST :', y_test.shape[0])


    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    # ENTRENAMIENTO
    # ---------------------------------------------------------------------------------------------------------------------
    learning_rate = 1e-3
    dim_input_users = 1  # el entero a partir del cual se calcula el embedding
    tam_batch = 256
    max_epochs = 10000
    dim_input_rest = NUM_PALABRAS  # porque ya le damos el vector de NUM_PALABRAS palabras

    VALIDATION = True
    model, n_epochs = entrenar(VALIDATION, max_epochs, tam_batch, learning_rate,
                               idx_usr.shape[0], ids_rest.shape[0], dim_input_users, dim_input_rest,
                               X_train, X_train_dev, X_test, y_train, y_train_dev, y_test)

    # hacer que entrene con [train+dev] -> [test]
    VALIDATION = False
    max_epochs = n_epochs  # las epochs que devolvió el modelo anterior
    model, n_epochs = entrenar(VALIDATION, max_epochs, tam_batch, learning_rate,
                               idx_usr.shape[0], ids_rest.shape[0], dim_input_users, dim_input_rest,
                               X_train, X_train_dev, X_test, y_train, y_train_dev, y_test)



print('fin!')
