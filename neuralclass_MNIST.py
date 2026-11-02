
import numpy as np
from neural_class import Layer, neural_net, dataset

import random as rd
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow.keras.datasets import fashion_mnist # type: ignore

# Load the separate train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Concatenate them into a single dataset (70,000 images total)
x_all = np.concatenate([x_train, x_test], axis=0)
y_all = np.concatenate([y_train, y_test], axis=0)

batch_size = 64
architecture = [[512, "arctan_sin"],[1024, "arctan_sin"],[128, "arctan_sin"], [32,"arctan_sin"],[10, "softmax"]]
Data = dataset(x_all, y_all)
Data.prepare(10, 0.9, batch_size)  # 10 classes, 80% training data
neural_network = neural_net(Data.x.shape[1], architecture)
neural_network.fit(Data, learning_rate=0.1, epochs=50)