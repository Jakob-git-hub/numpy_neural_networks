import numpy as np

class neural_network:
  def __init__(self, neurons,input_dim,  output_dim):
    self.neurons = neurons

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
    if x.ndim == 1:
        x = x.reshape(1, -1)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


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

  def one_hot_encoder(self, labels): 
    one_hot_labels = np.zeros((labels.shape[0], self.output_dim))
    for index, number in enumerate(labels):
       one_hot_labels[index][number]= 1.0
    return one_hot_labels
     
  def flatten_input(self, array):
    new_shape = array.shape[:-2] + (array.shape[-2] * array.shape[-1],)
    flattened = array.reshape(new_shape)
    return flattened
  
  def fit(self, input_data, data_labels , learning_rate, batch_size, epochs):
    n_samples = batch_size * (input_data.shape[0] // batch_size)
    input_data = input_data[:n_samples]
    data_labels = data_labels[:n_samples]

    n_batches = input_data.shape[0] // batch_size
    input_data = self.flatten_input(input_data)
    data_labels = self.one_hot_encoder(data_labels)


    for j in range(epochs):
        indices = np.arange(input_data.shape[0])
        np.random.shuffle(indices)
        input_data = input_data[indices]
        data_labels = data_labels[indices]
        for i in range((n_batches)):

            batch = input_data[i*batch_size:(i+1)*batch_size]
            batch_labels = data_labels[i*batch_size:(i+1)*batch_size]
            # 1. Forward Pass (für den GESAMTEN Batch auf einmal)
            # xor_input hat Shape (batch_size, in_dim)
            l1_linear, l1_out, output = self.forward_pass(batch)
            
            # 2. Backpropagation Layer 2 (Output-Schicht)
            # Shape: (batch_size, out_dim)
            error_output = (output - batch_labels) / batch_size
            
            # Gradienten für Layer 2
            # Transponieren von l1_out: (hidden_dim, batch_size) @ (batch_size, out_dim) -> (hidden_dim, out_dim)
            grad_w2 = np.dot(l1_out.T, error_output)/batch_size
            grad_b2 = np.sum(error_output, axis=0)/batch_size  # Summe über alle Zeilen (den Batch)

            # 3. Backpropagation Layer 1 (Hidden-Schicht)
            # (batch_size, out_dim) @ (out_dim, hidden_dim) -> (batch_size, hidden_dim)
            error_hidden = np.dot(error_output, self.layer2_weight.T) * self.relu_derivative(l1_linear)
            
            # Gradienten für Layer 1
            # Transponieren von input_data: (in_dim, batch_size) @ (batch_size, hidden_dim) -> (in_dim, hidden_dim)
            grad_w1 = np.dot(batch.T, error_hidden)/batch_size  # Durchschnitt über den Batch
            grad_b1 = np.sum(error_hidden, axis=0)/batch_size # Summe über den Batch

            # 4. Gewichte und Biases aktualisieren
            self.layer2_weight -= learning_rate * grad_w2
            self.layer2_bias   -= learning_rate * grad_b2
            self.layer1_weight -= learning_rate * grad_w1
            self.layer1_bias   -= learning_rate * grad_b1



  # --- ERGEBNISSE NACH DEM TRAINING ---
  def predict(self, input_data):
    from matplotlib import pyplot as plt
    flattened_input = self.flatten_input(input_data)
    _, _, pred = self.forward_pass(flattened_input)
    print(f"Input: {input_data} -> Vorhersage: {np.argmax(pred)}, mit der Warscheinlichkeit: {pred}")
    plt.imshow(input_data, cmap='gray')  # cmap='gray' sorgt für die korrekte Schwarz-Weiß-Darstellung
    plt.axis('off')                     # Blendat die Achsenbeschriftung für eine sauberere Optik aus
    plt.show()
    


  def evaluate(self, test_input_data, test_output_data):
    test_input_data = self.flatten_input(test_input_data)
    test_output_data = self.one_hot_encoder(test_output_data)
   
    _, _, pred = self.forward_pass(test_input_data)

        
    # Categorical Cross-Entropy Loss
    eps = 1e-15
    pred_clipped = np.clip(pred, eps, 1 - eps)
    loss = -np.mean(np.sum(test_output_data * np.log(pred_clipped), axis=1))
    
  
    return loss

