
import numpy as np
from neural_class import Layer, neural_net, dataset

import random as rd
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow.keras.datasets import mnist # type: ignore

# Load the separate train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Concatenate them into a single dataset (70,000 images total)
x_all = np.concatenate([x_train, x_test], axis=0)
y_all = np.concatenate([y_train, y_test], axis=0)
activattion = "t_sin"  # Choose activation function: "relu", "sigmoid", "tanh", "arctan_sin", "relu_sin", "tanh_sin"
batch_size = 128
architecture = [[2500,activattion ], [2000, activattion], [1500, activattion],[1000, activattion],[500, activattion],[10, "softmax"]]
Data = dataset(x_all, y_all)
test_data = dataset(x_test, y_test)
test_data.prepare(10, 0.1, batch_size)  # 10 classes, 80% training data
Data.prepare(10, 0.9, batch_size)  # 10 classes, 80% training data
neural_network = neural_net(Data.x.shape[1], architecture)
neural_network.fit(Data, learning_rate=0.001, epochs=10, x_test=test_data.x_test, y_test=test_data.y_test)  # Train the model