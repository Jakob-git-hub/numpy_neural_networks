from optimised_forward_cnn import neural_network
import numpy as np

import random as rd
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow.keras.datasets import fashion_mnist # type: ignore

# Loads directly as 3D arrays
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize to 0.0 - 1.0
X_train = X_train / 255.0
X_test = X_test / 255.0

train_test_split = 0.8
X_test =  X_test[0:int(X_test.shape[0]*0.1)]
y_test = y_test[0:int(y_test.shape[0]*0.1)]

print(X_train.shape)  
print(y_train.shape)
output_dim = 10

model = neural_network(256, 128, X_train.shape[1]*X_train.shape[2], output_dim)
loss = model.fit(X_train, y_train, X_test, y_test, 0.01 ,32 ,50)

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Finaler Test-Loss: {test_loss:.4f}")
print(f"Finale Test-Accuracy: {test_accuracy:.2%}")