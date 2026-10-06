from optimised_forward_cnn import neural_network
import numpy as np
average_loss = []
neuronen_exp = 9
input_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
output_data = np.array([[0],[1],[1],[0]], dtype=np.float32)
input_dim = input_data.shape[1]
output_dim = output_data.shape[1]


for i in range(1,neuronen_exp+1):
  model = neural_network(neurons=2**i, input_dim=input_dim, output_dim=output_dim)
  for j in range(1,10):
    model.reset_weights()
    model.fit(input_data, output_data, learning_rate=0.1*j, epochs=2000)
    evaluation = model.evaluate(100, input_data, output_data)
    average_loss.append(evaluation)
    print(f"Neurons: {2**i}, Learning Rate: {0.1*j}, Average Loss: {average_loss[-1]}")

print(average_loss)
  
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# 1. Die 1D-Liste in eine 2D-Matrix umwandeln (6 Zeilen für Neuronen, 9 Spalten für Learning Rate)
loss_matrix = np.array(average_loss).reshape(neuronen_exp, 9)

# 2. Beschriftungen aus deinen Schleifenbereich definieren (range(1,9) und range(1,10))
neuron_counts = [2**i for i in range(1, neuronen_exp+1)]  # [2, 4, 8, 16, 32, 64, 128, 256]
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