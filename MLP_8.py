import numpy as np

class Layer:
    # this class cannot see its neighboring classes.
    def __init__(self, n, input_size, activation):
        """ 
        Args:
        1. n(int): layer size or number of neurons
        2. input_size(int): incoming vector size
        3. activation(str): activation function name
        """
        self.weights_matrix = np.random.rand(n, input_size)
        self.bias_vector = np.random.rand(n)
        self.activation = activation

    def forward(self, vector_in):
        pre_activation = (self.weights_matrix @ vector_in) + self.bias_vector
        vector_out = self.activations(self.activation, pre_activation)
        return vector_out
    
    def backward(self, ):
        pass            
    
    def activations(self, name, vector):
        if name == "sigmoid":
            return self.sigmoid(vector)
        elif name == "softmax":
            return self.sigmoid(vector)
    
    def sigmoid(self, vector):
        neg_exp_vector = np.exp(-vector)   # exponentiate components
        return 1 / (1 + neg_exp_vector)
    
    def softmax(self, vector):
        exp_vec = np.exp(vector)   # exponentiate components
        return exp_vec / sum(exp_vec)
    
    def layer_info(self):
        print("Weight Matrix:\n",self.weights_matrix)
        print("Bias Vector:\n",self.bias_vector.T)
        print("Activation Function:",self.activation)
        print("----------------------------")

class NN:
    def __init__(self, Layers):
        """
        Args: 
        - Layers: A list of tuples where each tuple looks like: (int,"string")
            example NN: [10000, (100, "relu"), (100, "softmax"), 10]
                - 10000 : input shape
                - (100, "relu") : a 100 neurons layer with relu
                - (100, "softmax") : a 50 neurons layer with softmax
                - 10 : output shape
                - This means there are three layers(2 hidden + 1 output)
        """
        # Number of layers
        self.N_layers = len(Layers) - 1
        # User definition of the network:
        self.Layers_definition = Layers
        # List of Layer objects
        self.Layers = []

        self.valid_activations = ["sigmoid","softmax"]
        # Parsing & Initializing a dense network of Layer Objects:
        try:
            for i, layer_i in enumerate(self.Layers_definition[1:-1]):
                # Creating layers object:
                layer_size = layer_i[0]
                if i == 0:
                    prev_layer_size = self.Layers_definition[0]
                else:
                    prev_layer_size = self.Layers_definition[i-1]
                
                # Activation values validation & initialization:
                activation = layer_i[1]
                if activation not in self.valid_activations:
                    print(f"Only valid activation values = {self.valid_activations}")
                    return 0
                self.Layers.append(Layer(layer_size, prev_layer_size, activation))

        except:
            example = [10000, (100, "relu"), (100, "softmax"), 10]
            print(f"Correct syntax for initilization:\n\t example:{example}")
            
    def feed_forward(self, input_vector):
        for layer in self.Layers:
            output_vector = layer(input_vector)
        return output_vector

    def network_info(self):
        for i, layer in enumerate(self.Layers):
            print(f"\n-------Hidden Layer {i}-------")
            layer.layer_info()
            print()

# Test:
def main():
    layers = [3, (3, "sigmoid"), (3, "softmax"), 3]
    MLP = NN(layers)
    MLP.network_info()

if __name__=="__main__":
    main()
