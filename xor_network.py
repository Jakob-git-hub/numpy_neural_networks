import numpy as np

def main(neurons, learning_rate,epochs):
  def get_weight_vector(input,neurons):
    weight_vector = np.random.uniform(low=-1.0, high=1.0, size=(input, neurons))
    return weight_vector
  def get_bias_vector(neurons):
    bias_vector = np.random.uniform(low=-1.0, high=1.0, size=(1,neurons))
    return bias_vector

  def relu(x):
      return np.maximum(0, x)
  def relu_derivative(x):
      return np.where(x > 0, 1.0, 0.0)
  def sigmoid(x):
      return 1 / (1 + np.exp(-x))


  def create_network(layer1_neurons, layer2_neurons, input_neurons):
    layer1_weight = get_weight_vector(input_neurons, layer1_neurons)
    layer1_bias = get_bias_vector(layer1_neurons)

    layer2_weight = get_weight_vector(layer1_weight.shape[1], layer2_neurons)
    layer2_bias = get_bias_vector(layer2_neurons)

    return layer1_weight, layer1_bias, layer2_weight, layer2_bias



  def forward_pass(xor_values, layer1_weight, layer1_bias, layer2_weight, layer2_bias):
    
    layer1_linear = np.dot(xor_values, layer1_weight) + layer1_bias
    layer1_output = relu(layer1_linear)
    layer2_output = np.dot(layer1_output, layer2_weight) + layer2_bias
    final_output = sigmoid(layer2_output)

    return layer1_linear, layer1_output, final_output

  layer1_neurons = neurons
  layer2_neurons = 1
  epochs = epochs

  learning_rate = learning_rate


  xor_input = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
  xor_output = np.array([[0],[1],[1],[0]], dtype=np.float32)
  input_neurons = xor_input.shape[1]

  layer1_weight, layer1_bias, layer2_weight, layer2_bias = create_network(layer1_neurons, layer2_neurons, input_neurons)

  

  for epoch in range(epochs):
    # 1. Forward Pass (für den GESAMTEN Batch auf einmal)
    # xor_input hat Shape (batch_size, in_dim)
    l1_linear, l1_out, output = forward_pass(xor_input, layer1_weight, layer1_bias, layer2_weight, layer2_bias)
    
    # 2. Backpropagation Layer 2 (Output-Schicht)
    # Shape: (batch_size, out_dim)
    error_output = (output - xor_output) * output * (1 - output)
    
    # Gradienten für Layer 2
    # Transponieren von l1_out: (hidden_dim, batch_size) @ (batch_size, out_dim) -> (hidden_dim, out_dim)
    grad_w2 = np.dot(l1_out.T, error_output)
    grad_b2 = np.sum(error_output, axis=0)  # Summe über alle Zeilen (den Batch)

    # 3. Backpropagation Layer 1 (Hidden-Schicht)
    # (batch_size, out_dim) @ (out_dim, hidden_dim) -> (batch_size, hidden_dim)
    error_hidden = np.dot(error_output, layer2_weight.T) * relu_derivative(l1_linear)
    
    # Gradienten für Layer 1
    # Transponieren von xor_input: (in_dim, batch_size) @ (batch_size, hidden_dim) -> (in_dim, hidden_dim)
    grad_w1 = np.dot(xor_input.T, error_hidden)
    grad_b1 = np.sum(error_hidden, axis=0)  # Summe über den Batch

    # 4. Gewichte und Biases aktualisieren
    layer2_weight -= learning_rate * grad_w2
    layer2_bias   -= learning_rate * grad_b2
    layer1_weight -= learning_rate * grad_w1
    layer1_bias   -= learning_rate * grad_b1
      
  test_epochs = 100

  # --- ERGEBNISSE NACH DEM TRAINING ---
  #print("Vorhersagen nach dem Training:")
  #for j in range(xor_input.shape[0]):
  #    _, _, pred = forward_pass(xor_input[j], layer1_weight, layer1_bias, layer2_weight, layer2_bias)
  #   print(f"Input: {xor_input[j]} -> Vorhersage: {pred[0]:.4f} (Soll: {xor_output[j][0]})")
  output = np.zeros((test_epochs, xor_output.shape[0], xor_output.shape[1]), dtype=np.float32)
  for i in range(test_epochs):

    _, _, pred = forward_pass(xor_input, layer1_weight, layer1_bias, layer2_weight, layer2_bias)
    loss = (pred - xor_output)**2
    output[i] = loss
  average_loss = np.mean(output)
  return average_loss

