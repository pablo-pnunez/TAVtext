import tensorflow as tf
import numpy as np
import time
import ray
import os


@ray.remote(num_gpus=1)
class TrainingActor(object):
    def __init__(self, dataset, config):
        (x_train, y_train) = dataset
        self.config = config

        # self.__config_session__()

        # Crear un dataset de ejemplo
        self.dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.dataset = self.dataset.batch(512)

        # Crear el modelo
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(2,), activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        # Compilar el modelo
        self.model.compile(optimizer='adam', loss='mse')

    def train(self):
        print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        # Entrenar el modelo
        self.model.fit(self.dataset, epochs=1000, verbose=1)
        return 0

    def __config_session__(self):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)


if __name__ == "__main__":
                                      
    context = ray.init(num_gpus=2)

    # Crear datos de ejemplo
    x_train = np.random.rand(10000, 2)
    y_train = np.random.rand(10000, 1)

    # Read the mnist dataset and put it into shared memory once so that workers don't create their own copies.
    dataset_id = ray.put((x_train, y_train))
    training_actors = [TrainingActor.remote(dataset_id, seed) for seed in range(4)]

    # Make them all train in parallel.
    accuracy_ids = [actor.train.remote() for actor in training_actors]
    print(ray.get(accuracy_ids))
