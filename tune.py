from optimised_forward_cnn import neural_network
import numpy as np
from sklearn.datasets import load_digits
import random as rd
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow.keras.datasets import mnist # type: ignore

# Loads directly as 3D arrays
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize to 0.0 - 1.0
X_train = X_train / 255.0
X_test = X_test / 255.0

train_test_split = 0.8
X_test =  X_test[0:int(X_test.shape[0]*0.1)]
y_test = y_test[0:int(y_test.shape[0]*0.1)]

print(X_train.shape)  
print(X_test.shape)
#input_train_data= input_data[0:int(input_data.shape[0]*train_test_split)]
#output_train_data = output_data[0:int(output_data.shape[0]*train_test_split)]

#input_test_data = input_data[int(input_data.shape[0]*train_test_split):]
#output_test_data = output_data[int(output_data.shape[0]*train_test_split):]

neuronen_exp = 3
starting_neurons = 32
average_loss = []
print("Starting Grid Search for Neurons and Learning Rate")
for i in range(1,neuronen_exp+1):
  model = neural_network(neurons1=int(starting_neurons*(2**(i-1))), neurons2=int(starting_neurons*(2**(i-3))), input_dim=784, output_dim=10)
  for j in range(1,10):
    model.reset_weights()
    model.fit(X_train, y_train, learning_rate=0.3*(0.8**j), batch_size=200, epochs=25)
    evaluation = model.evaluate(X_train, y_train)
    average_loss.append(evaluation)
    print(f"Neurons: {starting_neurons*(2**(i-1)), starting_neurons*(2**(i-3))}, Learning Rate: {0.3*(0.8**j)}, Average Loss: {average_loss[-1]}, Accuracy: {1-average_loss[-1]}")

print(average_loss)
print(min(average_loss))
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# 1. Die 1D-Liste in eine 2D-Matrix umwandeln (6 Zeilen für Neuronen, 9 Spalten für Learning Rate)
loss_matrix = np.array(average_loss).reshape(neuronen_exp, 9)

# 2. Beschriftungen aus deinen Schleifenbereich definieren (range(1,9) und range(1,10))
neuron_counts = [2**(i+5) for i in range(1, neuronen_exp+1)]  # [2, 4, 8, 16, 32, 64, 128, 256]
learning_rates = [
    0.1 * j for j in range(1, 10)
]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 3. Heatmap zeichnen
plt.figure(figsize=(10, 6))
sns.heatmap(
    loss_matrix,
    annot=True,
    fmt=".4f",
    xticklabels=learning_rates,
    yticklabels=neuron_counts,
    cmap="viridis_r",
    cbar_kws={"label": "Average Loss"},
)

plt.title("Grid Search: Average Loss", fontsize=14)
plt.xlabel("Learning Rate", fontsize=12)
plt.ylabel("Anzahl Neuronen (Hidden Layer)", fontsize=12)
plt.tight_layout()
plt.show()








