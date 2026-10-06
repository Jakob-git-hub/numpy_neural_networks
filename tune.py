from optimised_forward_cnn import neural_network
import numpy as np
model = neural_network(neurons=64, input_dim=2, output_dim=1)

input_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
output_data = np.array([[0],[1],[1],[0]], dtype=np.float32)

model.fit(input_data, output_data,  learning_rate=0.5, epochs=2000)
model.predict(input_data, output_data)
evaluation = model.evaluate(100, input_data, output_data)
print("Final Weights and Biases:")
    
print("Evaluation Loss:", evaluation)