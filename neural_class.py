import numpy as np
architecture = [[5,"relu"],[2,"relu"]]

class Layer:
    def __init__(self, input, neurons, activation):
        self.input = input
        self.neurons = neurons
        self.activation = activation
        self.weight = np.random.uniform(low=-1.0,high=1.0, size=(input, neurons))
        self.bias = np.random.uniform(low=-1.0, high=1.0, size=(1,neurons))


class neural_net:
    def __init__(self, features, architecture):
        self.architecture = architecture
        self.features = features
        self.model = self.init_weights()

    def init_weights(self):
        model = []
        current_dim = self.features
        for layer in self.architecture:
            model.append(Layer(current_dim,layer[0], layer[1]))
            current_dim = layer[0]
        return model  

    def relu(self, x):
        return np.maximum(0, x)
    def relu_d(self, x):
        return np.where(x > 0, 1.0, 0.0)
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def softmax(self,x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)


    def forward_pass(self, input_data):
        layer_buffer = {}
        layer_input = input_data

        for idx, layer in enumerate(self.model):

            layer_buffer[idx] = {}
            
            activation = getattr(self, layer.activation)

            layer_buffer[idx]["linear"] = (layer_input@layer.weight) + layer.bias       
            layer_buffer[idx]["activated"],  = activation(layer_buffer[idx]["linear"])
            layer_input = layer_buffer[idx]["activated"]

        return layer_input, layer_buffer 

    def backpropagation(self, labels, layer_buffer, output, batch_size):
        grad = {}
        error_signal = labels - output / batch_size
        for idx, layer in reversed(list(enumerate(self.model))):
            grad[idx] = {}
            grad[idx]["bias"] = np.sum(error_signal, axis=0, keepdims=True) / batch_size
            grad[idx]["weight"] = error_signal*layer_buffer[idx-1]["activated"]

    def fit(self, x_train,):
        


        
class data:
    def __init__(self,x,y,split,):
        x = self.flatten_input(x)
        y = self.one_hot_encoder(y)
        x_train = 

    def one_hot_encoder(self, labels): 
        one_hot_labels = np.zeros((labels.shape[0], self.output_dim))
        for index, number in enumerate(labels):
           one_hot_labels[index][number]= 1.0
        return one_hot_labels
         
    def flatten_input(self, array):
        new_shape = array.shape[:-2] + (array.shape[-2] * array.shape[-1],)
        flattened = array.reshape(new_shape)
        return flattened        

