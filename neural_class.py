import numpy as np
architecture = [[5,"relu"],[2,"relu"]]

class Layer:
    def __init__(self, input_dim, neurons, activation):
        self.input_dim = input_dim
        self.neurons = neurons
        self.activation = activation
        if activation in ("relu", "relu_sin"):
            scale = np.sqrt(2.0 / input_dim)
        else:
            scale = np.sqrt(1.0 / input_dim)

        self.weight = np.random.randn(input_dim, neurons) * scale
        self.bias = np.zeros((1, neurons))



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
    def relu_backward(self, x, upstream_gradient):
        return upstream_gradient * self.relu_d(x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_d(self, x):
        sigmoid = self.sigmoid(x)
        return sigmoid * (1 - sigmoid)
    def sigmoid_backward(self, x, upstream_gradient):
        return upstream_gradient * self.sigmoid_d(x)
    
    def sin(self, x):
        return np.sin(x)
    def sin_d(self, x):
        return np.cos(x)
    def sin_backward(self, x, upstream_gradient):
        return upstream_gradient * self.sin_d(x)
    
    def relu_sin(self, x):
        return (np.sin(x) + np.maximum(0,x))
    def relu_sin_d(self, x):
        return  np.select([x < 0, x > 0], [np.cos(x), 1 + np.cos(x)])
    def relu_sin_backward(self, x, upstream_gradient):
        return upstream_gradient * self.relu_sin_d(x)
    
    def arctan_sin(self, x):
        return (np.arctan(x) +np.sin(x))
    def arctan_sin_d(self, x):
        return  np.select([x < 0, x > 0], [1/(1+x**2), np.cos(x) + 1/(1+x**2)])
    def arctan_sin_backward(self, x, upstream_gradient):
        return upstream_gradient * self.arctan_sin_d(x)
    
    def softmax(self,x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    def softmax_d(self, x):
        s = self.softmax(x)
        return (
            s[:, :, np.newaxis]
            * (np.eye(s.shape[1])[np.newaxis, :, :] - s[:, np.newaxis, :])
        )
    def softmax_backward(self, x, upstream_gradient):
        s = self.softmax(x)
        jacobian_product = np.sum(upstream_gradient * s, axis=1, keepdims=True)
        return s * (upstream_gradient - jacobian_product)


    def forward_pass(self, input_data):
        layer_buffer = {"input": input_data}
        layer_input = input_data

        for idx, layer in enumerate(self.model):

            layer_buffer[idx] = {}

            activation = getattr(self, layer.activation)

            layer_buffer[idx]["linear"] = (layer_input@layer.weight) + layer.bias
            layer_buffer[idx]["activated"] = activation(layer_buffer[idx]["linear"])
            layer_input = layer_buffer[idx]["activated"]

        return layer_input, layer_buffer

    def backpropagation(self, labels, layer_buffer, output, batch_size):
        """Return weight and bias gradients for categorical cross-entropy."""
        grad = {}
        eps = 1e-15
        output_clipped = np.clip(output, eps, 1 - eps)
        error_signal = -labels / output_clipped / batch_size

        for idx, layer in reversed(list(enumerate(self.model))):
            grad[idx] = {}

            activation_backward = getattr(
                self, f"{layer.activation}_backward"
            )
            delta = activation_backward(
                layer_buffer[idx]["linear"], error_signal
            )

            if idx == 0:
                layer_input = layer_buffer.get("input")
            else:
                layer_input = layer_buffer[idx - 1]["activated"]

            if layer_input is None:
                raise ValueError(
                    "layer_buffer must contain the input under the 'input' key"
                )

            grad[idx]["weight"] = layer_input.T @ delta
            grad[idx]["bias"] = np.sum(delta, axis=0, keepdims=True)

            error_signal = delta @ layer.weight.T

        return grad

    def fit(self, dataset, learning_rate, epochs):

        for j in range(epochs):
            dataset.shuffle_data()
            for i in range(dataset.n_batches):
                batch = dataset.x_train[i*dataset.batch_size:(i+1)*dataset.batch_size]
                batch_labels = dataset.y_train[i*dataset.batch_size:(i+1)*dataset.batch_size]
                output, layer_buffer = self.forward_pass(batch)
                grads = self.backpropagation(batch_labels, layer_buffer, output, dataset.batch_size)
                for idx,layer in enumerate(self.model):
                    layer.weight -= learning_rate * grads[idx]["weight"]
                    layer.bias -= learning_rate * grads[idx]["bias"]

            Loss, Accuracy = self.evaluate(dataset)
  
            print(f"Episode: {j}, Loss: {Loss:.4f}, Accuracy: {Accuracy:.2%}")



    def evaluate(self, dataset):


        output,_ = self.forward_pass(dataset.x_test)
        
        eps = 1e-15
        pred_clipped = np.clip(output, eps, 1 - eps)
        pred_label = np.argmax(output, axis=1)
        loss = -np.mean(np.sum(dataset.y_test * np.log(pred_clipped), axis=1))
        accuracy = np.mean(pred_label == np.argmax(dataset.y_test, axis=1))
          
        return loss, accuracy
    
class dataset:  
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def prepare(self, output_dim, split, batch_size):
        self.normalize_input()
        self.flatten_input()
        self.one_hot_encoder(output_dim)
        self.split_data(split, batch_size)

    def one_hot_encoder(self, output_dim):
        one_hot_labels = np.zeros((self.y.shape[0], output_dim))
        for index, number in enumerate(self.y):
           one_hot_labels[index][number]= 1.0
        self.y = one_hot_labels
    def normalize_input(self):
        self.x = self.x / 255.0

    def flatten_input(self):
        new_shape = self.x.shape[:-2] + (self.x.shape[-2] * self.x.shape[-1],)
        flattened = self.x.reshape(new_shape)
        self.x = flattened

    def split_data(self, split, batch_size):
        self.n_batches = int((self.x.shape[0]*split)//batch_size)
        self.batch_size = batch_size
        split_index = int(self.n_batches*batch_size)
        
        self.x_train = self.x[0:split_index]
        self.y_train = self.y[0:split_index]
        self.x_test = self.x[split_index:]
        self.y_test = self.y[split_index:]

    def shuffle_data(self):
            indices = np.arange(len(self.x_train))
            np.random.shuffle(indices)
            self.x_train = self.x_train[indices]
            self.y_train = self.y_train[indices]
