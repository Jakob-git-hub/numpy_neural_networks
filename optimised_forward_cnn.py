import numpy as np

class neural_network:
  def __init__(self, neurons,input_dim, batch_size, output_dim):
    self.neurons = neurons
    self.batch_size = batch_size

    self.input_dim = input_dim
    self.output_dim = output_dim

    self.layer1_weight, self.layer1_bias, self.layer2_weight, self.layer2_bias = self.create_network(self.input_dim, self.neurons, self.output_dim)
    
  def get_weight_vector(self, input, neurons):
    weight_vector = np.random.uniform(low=-1.0, high=1.0, size=(input, neurons))
    return weight_vector
  def get_bias_vector(self, neurons):
    bias_vector = np.random.uniform(low=-1.0, high=1.0, size=(1,neurons))
    return bias_vector

  def relu(self, x):
      return np.maximum(0, x)
  def relu_derivative(self, x):
      return np.where(x > 0, 1.0, 0.0)
  def sigmoid(self, x):
      return 1 / (1 + np.exp(-x))
  def softmax(self,x):
    # Maximum abziehen für numerische Stabilität (verhindert Overflow)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


  def create_network(self, input_neurons, layer1_neurons, layer2_neurons):
    layer1_weight = self.get_weight_vector(input_neurons, layer1_neurons)
    layer1_bias = self.get_bias_vector(layer1_neurons)

    layer2_weight = self.get_weight_vector(layer1_weight.shape[1], layer2_neurons)
    layer2_bias = self.get_bias_vector(layer2_neurons)

    return layer1_weight, layer1_bias, layer2_weight, layer2_bias




  def forward_pass(self, input_data):
    
    layer1_linear = (input_data@self.layer1_weight) + self.layer1_bias
    layer1_output = self.relu(layer1_linear)
    layer2_output = (layer1_output@self.layer2_weight) + self.layer2_bias
    final_output = self.softmax(layer2_output)

    return layer1_linear, layer1_output, final_output


  def reset_weights(self):
    self.layer1_weight, self.layer1_bias, self.layer2_weight, self.layer2_bias = self.create_network(self.input_dim, self.neurons, self.output_dim)


  
  def fit(self, input_data, output_data , learning_rate, epochs):
    self.input_data = input_data
    self.output_data = output_data

    for i in range((input_data.shape[0]/self.batch_size)*epochs):
        batch = input_data[i*self.batch_size:(i+1)*self.batch_size]
        # 1. Forward Pass (für den GESAMTEN Batch auf einmal)
        # xor_input hat Shape (batch_size, in_dim)
        l1_linear, l1_out, output = self.forward_pass(batch)
        
        # 2. Backpropagation Layer 2 (Output-Schicht)
        # Shape: (batch_size, out_dim)
        error_output = (output - target) * output * (1 - output)
        
        # Gradienten für Layer 2
        # Transponieren von l1_out: (hidden_dim, batch_size) @ (batch_size, out_dim) -> (hidden_dim, out_dim)
        grad_w2 = np.dot(l1_out.T, error_output)
        grad_b2 = np.sum(error_output, axis=0)  # Summe über alle Zeilen (den Batch)

        # 3. Backpropagation Layer 1 (Hidden-Schicht)
        # (batch_size, out_dim) @ (out_dim, hidden_dim) -> (batch_size, hidden_dim)
        error_hidden = np.dot(error_output, self.layer2_weight.T) * self.relu_derivative(l1_linear)
        
        # Gradienten für Layer 1
        # Transponieren von input_data: (in_dim, batch_size) @ (batch_size, hidden_dim) -> (in_dim, hidden_dim)
        grad_w1 = np.dot(self.input_data.T, error_hidden)
        grad_b1 = np.sum(error_hidden, axis=0)  # Summe über den Batch

        # 4. Gewichte und Biases aktualisieren
        self.layer2_weight -= learning_rate * grad_w2
        self.layer2_bias   -= learning_rate * grad_b2
        self.layer1_weight -= learning_rate * grad_w1
        self.layer1_bias   -= learning_rate * grad_b1



  # --- ERGEBNISSE NACH DEM TRAINING ---
  def predict(self, input_data, output_data):
    for j in range(input_data.shape[0]):
        _, _, pred = self.forward_pass(input_data[j])
        print(f"Input: {input_data[j]} -> Vorhersage: {pred} (Soll: {output_data[j][0]})")

  def evaluate(self,test_epochs, test_input_data, test_output_data):
    output = np.zeros((test_epochs, test_output_data.shape[0], test_output_data.shape[1]), dtype=np.float32)
    for i in range(test_epochs):

        _, _, pred = self.forward_pass(test_input_data)
        loss = (pred - test_output_data)**2
        output[i] = loss
    average_loss = np.mean(output)
    return average_loss

