#Hier wurde der forward pass komplett selber geschrieben ohne dass numpy dot oder matmul verwendet wurde.
#Das neuronale Netzwerk war allerdings schon komplett anpassbar über neuronen parameter, d.h man konnte die Anzahl der Neuronen in Input layer, einem Hidde layer und dem output layer bestimmen, dies wurde selber implementiert
#Die Backpropagation wurde allerdings noch von ki gemini übernommen, da ich sie noch nicht vollständig verstanden habe.
import numpy as np

def main(neurons, learning_rate):
  def get_weight_vector(neurons, input):
    weight_vector = np.random.uniform(low=-1.0, high=1.0, size=(neurons, input))
    return weight_vector
  def get_bias_vector(neurons):
    bias_vector = np.random.uniform(low=-1.0, high=1.0, size=(neurons,))
    return bias_vector

  def relu(x):
      return np.maximum(0, x)
  def relu_derivative(x):
      return np.where(x > 0, 1.0, 0.0)
  def sigmoid(x):
      return 1 / (1 + np.exp(-x))


  def create_network(layer1_neurons, layer2_neurons, input_neurons):
    layer1_weight = get_weight_vector(layer1_neurons, input_neurons)
    layer1_bias = get_bias_vector(layer1_neurons)

    layer2_weight = get_weight_vector(layer2_neurons, layer1_weight.shape[0])
    layer2_bias = get_bias_vector(layer2_neurons)

    return layer1_weight, layer1_bias, layer2_weight, layer2_bias



  def forward_pass(xor_values, layer1_weight, layer1_bias, layer2_weight, layer2_bias):
    
    layer1_linear = np.zeros((layer1_weight.shape[0]), dtype=np.float32)
    for i in range(len(layer1_bias)):
        for j in range(len(xor_values)):
            layer1_linear[i] += (xor_values[j] * layer1_weight[i][j])
        layer1_linear[i] += layer1_bias[i]

    layer1_output = relu(layer1_linear)
    layer2_output = np.zeros( layer2_weight.shape[0], dtype=np.float32)

    for i in range(len(layer2_bias)):
        for j in range(len(layer1_output)):
            layer2_output[i] += (layer1_output[j] * layer2_weight[i][j])
        layer2_output[i] += layer2_bias[i]

    final_output = sigmoid(layer2_output)

    return layer1_linear, layer1_output, final_output

  layer1_neurons = neurons
  layer2_neurons = 1
  epochs = 2000
  learning_rate = learning_rate


  xor_input = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
  xor_output = np.array([[0],[1],[1],[0]], dtype=np.float32)
  input_neurons = xor_input.shape[1]

  layer1_weight, layer1_bias, layer2_weight, layer2_bias = create_network(layer1_neurons, layer2_neurons, input_neurons)

#Einfacher Forwardpass für alle Inputs der XOR Funktion
  for i in range(1):
    output = np.zeros((xor_input.shape[0], layer2_neurons), dtype=np.float32)
    loss = np.zeros((xor_input.shape[0], layer2_neurons), dtype=np.float32)
    for j in range(xor_input.shape[0]):
      l1_linear, l1_out, output[j] = forward_pass(xor_input[j], layer1_weight, layer1_bias, layer2_weight, layer2_bias)
      loss[j] = (output[j] - xor_output[j])**2

    
    #print(output)
    #print(loss)

  for epoch in range(epochs):
    for j in range(xor_input.shape[0]):
      xor_values = xor_input[j]
      y_true = xor_output[j]
      
      # 1. Forward Pass
      l1_linear, l1_out, output = forward_pass(xor_values, layer1_weight, layer1_bias, layer2_weight, layer2_bias)
      
      # 2. Backpropagation Layer 2 (Output-Schicht)
      error_output = (output - y_true) * output * (1 - output) # Ableitung Sigmoid
      layer2_weight_copy = layer2_weight.copy()  # Kopie der Gewichte für die Berechnung der Fehler in Layer 1
      for i in range(layer2_weight.shape[0]):
          for k in range(layer2_weight.shape[1]):
              grad_w2 = error_output[i] * l1_out[k]
              layer2_weight[i][k] -= learning_rate * grad_w2
          layer2_bias[i] -= learning_rate * error_output[i]

      # 3. Backpropagation Layer 1 (Hidden-Schicht)
      error_hidden = np.zeros_like(l1_linear)
      for k in range(len(l1_linear)):
          sum_error_weights = 0
          for i in range(layer2_weight.shape[0]):
              sum_error_weights += error_output[i] * layer2_weight_copy[i][k]
          error_hidden[k] = sum_error_weights * relu_derivative(l1_linear[k])

      for i in range(layer1_weight.shape[0]):
          for k in range(layer1_weight.shape[1]):
              grad_w1 = error_hidden[i] * xor_values[k]
              layer1_weight[i][k] -= learning_rate * grad_w1
          layer1_bias[i] -= learning_rate * error_hidden[i]
      

  # --- ERGEBNISSE NACH DEM TRAINING ---
  #print("Vorhersagen nach dem Training:")
  #for j in range(xor_input.shape[0]):
  #    _, _, pred = forward_pass(xor_input[j], layer1_weight, layer1_bias, layer2_weight, layer2_bias)
  #   print(f"Input: {xor_input[j]} -> Vorhersage: {pred[0]:.4f} (Soll: {xor_output[j][0]})")
  output = np.zeros((xor_input.shape[0], 10), dtype=np.float32)
  for i in range(100):
     for j in range(xor_input.shape[0]):
      _, _, pred = forward_pass(xor_input[j], layer1_weight, layer1_bias, layer2_weight, layer2_bias)
      loss = (pred[0] - xor_output[j][0])**2
      output[j][i] = loss
  average_loss = np.mean(output, axis=0)
  return average_loss