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
            return self.softmax(vector)
    
    def sigmoid(self, vector):
        neg_exp_vector = np.exp(-vector)   # exponentiate components
        return 1 / (1 + neg_exp_vector)
    
    def softmax(self, vector):
        exp_vec = np.exp(vector)   # exponentiate components
        return exp_vec / sum(exp_vec)
    
    def layer_info(self):
        print("Size:\n",self.weights_matrix.shape[0])
        print("Weight Matrix:\n",self.weights_matrix)
        print("Bias Vector:\n",self.bias_vector.T)
        print("Activation Function:",self.activation)
        print("----------------------------")
        return self.weights_matrix.shape[0] + self.weights_matrix.shape[1] + self.bias_vector.shape[0]

class NN:
    def __init__(self, Layers):
        """
        Args: 
        - Layers: A list of tuples where each tuple looks like: (int,"string")
            example NN: [10000, (100, "relu"), (100, "softmax")]
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
        for i, layer_i in enumerate(self.Layers_definition[1:], start=1):
            # extract current layer’s size & activation
            layer_size, activation = layer_i

            # ALWAYS grab the immediately preceding entry
            prev_def = self.Layers_definition[i-1]
            # if the previous entry is a tuple, take its size; otherwise it’s an int
            prev_layer_size = prev_def[0] if isinstance(prev_def, tuple) else prev_def

            # activation check unchanged
            if activation not in self.valid_activations:
                print(f"Only valid activation values = {self.valid_activations}")
                return 0

            print(layer_size, prev_layer_size, activation)
            self.Layers.append(Layer(layer_size, prev_layer_size, activation))
            print(i)

        # except:
        #     example = [10000, (100, "relu"), (100, "softmax")]
        #     print(f"Correct syntax for initilization:\n\t example:{example}")
            
    def feed_forward(self, input_vector):
        print("\n######## FEED FORWARD: ########")
        for i, layer in enumerate(self.Layers):
            if i == 0:
                output_vector = layer.forward(input_vector)
            else:
                output_vector = layer.forward(output_vector)
            print(f"{i+1}th Layer's Output {output_vector}")
        return output_vector

    def network_info(self):
        total_parameters = 0
        for i, layer in enumerate(self.Layers):
            print(f"\n--------- Layer {i+1} ----------")
            total_parameters += layer.layer_info()
            print()
        print(f"Total Learnable Model Parameters = {total_parameters}")
# Test:
def main():
    layers = [10, (100, "sigmoid"), (100, "sigmoid"), (5, "softmax")]
    MLP = NN(layers)
    MLP.network_info()
    test_inp_vec = np.random.rand(10)
    print(f"\n\nInput Vector = {test_inp_vec}")
    MLP.feed_forward(test_inp_vec)

if __name__=="__main__":
    main()
