import numpy as np

################## Activation Functions & Derivatives ##################
def sigmoid(vector):
    # clipping for numerical stability
    vector = np.clip(vector, -500, 500) 
    neg_exp_vector = np.exp(-vector)
    return 1 / (1 + neg_exp_vector)

def sigmoid_derivative(a):
    # Takes activation 'a' (which is sigmoid(z)) as input
    return a * (1 - a)

def softmax(vector):
    # Subtract max for numerical stability
    vector = vector - np.max(vector, axis=0, keepdims=True)
    exp_vec = np.exp(vector)
    return exp_vec / np.sum(exp_vec, axis=0, keepdims=True)

def relu(vector):
    return np.maximum(0, vector)

def relu_derivative(z):
    # Takes pre-activation 'z' as input
    return (z > 0).astype(float)

def cross_entropy(y_pred, y_one_hot):
    # making sure inputs are numpy arrays
    y_pred = np.array(y_pred)
    y_one_hot = np.array(y_one_hot)
    # Clipping for numerical stability (avoid log(0))
    epsilon = 1e-12
    y_pred_clipped = np.clip(y_pred, epsilon, 1. - epsilon)
    loss = -np.sum(y_one_hot * np.log(y_pred_clipped))
    return loss # Return total loss for the sample


class Layer:
    # this class cannot see its neighboring classes so interactions will be in NN
    def __init__(self, n, input_size, activation, initialization_method = "xavier"):
        """ 
        Args:
        1. n(int): layer size(number of neurons)
        2. input_size(int): incoming vector size
        3. activation(str): activation function name
        """
        self.params_init_method = initialization_method
        if self.params_init_method == "xavier":
            # Corrected Xavier for uniform distribution
            limit = np.sqrt(6 / (input_size + n))
            self.weights_matrix = np.random.uniform(low=-limit, high=limit, size=(n, input_size))
        else:
            # Simple random initialization (can lead to issues)
            self.weights_matrix = np.random.rand(n, input_size) * 0.01 # Scale down
            print("Non Xavier Initialization was chosen for this layer...")
        
        # Bias usually initialized to zeros
        self.bias_vector = np.zeros((n, 1)) 
        self.activation = activation

        # --- Store gradients within the layer ---
        self.dL_dW = np.zeros_like(self.weights_matrix)
        self.dL_db = np.zeros_like(self.bias_vector)


    def layer_forward(self, vector_in):
        """
        Performs the forward pass for this layer.
        Args:
            vector_in (np.array): Input activations from the previous layer (a^{l-1}). Shape (input_size, 1).
        Returns:
            tuple: (pre_activation (z^l), activation (a^l))
                   Shapes: (n, 1), (n, 1)
        """
        # Ensure input is a column vector
        if vector_in.ndim == 1:
             vector_in = vector_in.reshape(-1, 1)
             
        pre_activation = (self.weights_matrix @ vector_in) + self.bias_vector
        
        if self.activation == "sigmoid":
            vector_out = sigmoid(pre_activation)
        elif self.activation == "softmax":
            vector_out = softmax(pre_activation)
        elif self.activation == "relu":
            vector_out = relu(pre_activation)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")
            
        return pre_activation, vector_out

    # Removing old backward methods as backprop logic will be centralized in NN
    # def last_layer_backward(self, A, Y, Z, avg_loss): ...
    # def dA_dZ(self, dL_dA): ...

    def layer_info(self):
        print("Size:\n",self.weights_matrix.shape[0])
        print("Weight Matrix Shape:\n",self.weights_matrix.shape)
        # print("Weight Matrix:\n",self.weights_matrix) # Often too large to print
        print("Bias Vector Shape:\n",self.bias_vector.shape)
        # print("Bias Vector:\n",self.bias_vector.T)
        print("Activation Function:",self.activation)
        print("----------------------------")
        return self.weights_matrix.shape[0] * self.weights_matrix.shape[1] + self.bias_vector.shape[0]

class NN:
    def __init__(self, Layers, debug = True, one_hot_encoding = None):
        """
        Args: 
        - Layers: A list where the first element is input size (int) 
                  and subsequent elements are tuples (layer_size, activation_str).
            example NN: [784, (128, "relu"), (10, "softmax")] 
        - one_hot_encoding(dictionary): Maps class labels to one-hot vectors.
        """
        self.N_layers = len(Layers) - 1
        self.Layers_definition = Layers
        self.Layers = []
        self.debug = debug

        self.valid_activations = ["sigmoid","softmax", "relu"]
        
        # Parsing & Initializing a dense network of Layer Objects:
        input_size = Layers[0] # Get the initial input size
        for i, layer_def in enumerate(Layers[1:]):
            layer_size, activation = layer_def
            
            # activation fn sanity check:
            if activation not in self.valid_activations:
                 raise ValueError(f"Activation '{activation}' not valid. Use one of {self.valid_activations}")

            self.debug and print(f"Initializing Layer {i+1}: Size={layer_size}, InputSize={input_size}, Activation='{activation}'")
            current_layer = Layer(layer_size, input_size, activation)
            self.Layers.append(current_layer)
            input_size = layer_size # Input size for the next layer is the current layer's size
            
        # Default One-Hot Encoding for MNIST (example)
        if one_hot_encoding == None:
            # Assuming output size is the size of the last layer
            output_dim = Layers[-1][0] 
            self.one_hot_encoding = {i: np.eye(output_dim)[i].reshape(-1,1) for i in range(output_dim)}
            self.debug and print(f"Initialized default one-hot encoding for {output_dim} classes.")
        else:
            self.one_hot_encoding = one_hot_encoding

    def forward_propogation(self, x):
        """
        Performs a full forward pass and caches intermediate values.
        Args:
            x (np.array): Input vector. Shape (input_feature_size,) or (input_feature_size, 1).
        Returns:
            tuple: (y_pred, cache)
                y_pred (np.array): Final output activation (a^L). Shape (output_size, 1).
                cache (dict): Stores intermediate values {'A0': x, 'Z1': z1, 'A1': a1, ...}
        """
        # Ensure input is a column vector
        if x.ndim == 1:
             x = x.reshape(-1, 1)
             
        print("\n######## FEED FORWARD: ########") if self.debug else None
        cache = {}
        current_a = x
        cache['A0'] = current_a # Cache input activation

        for i, layer in enumerate(self.Layers):
            layer_idx = i + 1
            z, current_a = layer.layer_forward(current_a) # Get pre-activation (z) and activation (a)
            
            cache[f'Z{layer_idx}'] = z
            cache[f'A{layer_idx}'] = current_a
            
            if self.debug:
                 print(f"Layer {layer_idx} Output Shape: {current_a.shape}")
                 # print(f"Layer {layer_idx} Output: {current_a.T}") # Often too verbose
                 
        y_pred = current_a
        return y_pred, cache

    def back_propogation(self, y_true_label, cache):
        """
        Performs backpropagation for a single training example.
        Args:
            y_true_label (int or string): The true class label (e.g., 7).
            cache (dict): The cache returned by forward_propogation.
                            {'A0': x, 'Z1': z1, 'A1': a1, ... 'ZL': zL, 'AL': aL}
        Returns:
            tuple: (grads_W, grads_b) 
                   Dictionaries containing gradients for each layer, keyed by layer index (1 to L).
                   grads_W = {1: dL_dW1, 2: dL_dW2, ...}
                   grads_b = {1: dL_db1, 2: dL_db2, ...}
                   (These gradients are NOT yet stored in the layers themselves)
        """
        print("\n######## BACKPROPAGATION: ########") if self.debug else None
        
        grads_W = {}
        grads_b = {}
        L = len(self.Layers) # Total number of layers

        # 1. Get one-hot encoded true label
        try:
            y_one_hot = self.one_hot_encoding[y_true_label]
        except KeyError:
             raise KeyError(f"Label '{y_true_label}' not found in one_hot_encoding dictionary.")
             
        # Ensure y_one_hot is a column vector
        if y_one_hot.ndim == 1:
            y_one_hot = y_one_hot.reshape(-1, 1)

        # 2. Initialize Backprop with Output Layer (Layer L)
        # Retrieve final activation a^L and pre-activation z^L from cache
        aL = cache[f'A{L}']
        zL = cache[f'Z{L}']
        last_layer = self.Layers[L-1] # Access last Layer object (index L-1)

        # Calculate delta for the output layer (delta^L)
        if last_layer.activation == "softmax":
             # Using the shortcut for Softmax + Cross-Entropy
             delta_L = aL - y_one_hot 
        elif last_layer.activation == "sigmoid":
             # Assuming Cross-Entropy loss for binary/multi-label with sigmoid outputs
             # dL/da = -(y/a) + (1-y)/(1-a)
             dL_daL = -(y_one_hot / aL) + ((1 - y_one_hot) / (1 - aL))
             daL_dzL = sigmoid_derivative(aL) # aL = sigmoid(zL)
             delta_L = dL_daL * daL_dzL # Element-wise product
        # Add elif for ReLU if it could be an output layer (less common)
        # elif last_layer.activation == "relu": ... 
        else:
             raise NotImplementedError(f"Backprop for output activation '{last_layer.activation}' not implemented yet.")
        
        self.debug and print(f"Delta {L} Shape: {delta_L.shape}")
        
        # Calculate Gradients for the Output Layer (Layer L)
        a_prev = cache[f'A{L-1}'] # Activation from the previous layer
        dL_dW_L = delta_L @ a_prev.T
        dL_db_L = delta_L
        
        grads_W[L] = dL_dW_L
        grads_b[L] = dL_db_L
        # Store gradients in the layer object itself as well
        last_layer.dL_dW = dL_dW_L
        last_layer.dL_db = dL_db_L


        # 3. Loop backwards through remaining layers (L-1 down to 1)
        delta_current = delta_L # Start with delta from the layer ahead (output layer)
        for l in range(L - 1, 0, -1): # l goes from L-1 down to 1
            layer_idx_ahead = l + 1 # Index of the layer *ahead* (l+1)
            layer_idx_current = l     # Index of the current layer (l)

            # Get parameters and cached values for the current layer (l)
            current_layer = self.Layers[layer_idx_current - 1] # Layer object at index l-1
            W_ahead = self.Layers[layer_idx_ahead - 1].weights_matrix # W^{l+1}
            
            z_current = cache[f'Z{layer_idx_current}'] # z^l
            a_prev = cache[f'A{layer_idx_current - 1}'] # a^{l-1} (from layer l-1, or input A0)

            # Calculate activation derivative for the current layer (l)
            if current_layer.activation == "sigmoid":
                activation_derivative = sigmoid_derivative(cache[f'A{layer_idx_current}']) # Use a^l
            elif current_layer.activation == "relu":
                activation_derivative = relu_derivative(z_current) # Use z^l
            # Add other activation derivatives if needed
            else:
                raise NotImplementedError(f"Derivative for activation '{current_layer.activation}' not implemented.")

            # Calculate delta for the current layer (delta^l)
            # delta^l = (W^{l+1}^T @ delta^{l+1}) * sigma'(z^l)
            delta_current = (W_ahead.T @ delta_current) * activation_derivative # Element-wise product
            
            self.debug and print(f"Delta {layer_idx_current} Shape: {delta_current.shape}")

            # Gradients for the current layer (l)
            dL_dW_l = delta_current @ a_prev.T
            dL_db_l = delta_current

            grads_W[layer_idx_current] = dL_dW_l
            grads_b[layer_idx_current] = dL_db_l
            # Store gradients in the layer object
            current_layer.dL_dW = dL_dW_l
            current_layer.dL_db = dL_db_l

        return grads_W, grads_b # Return gradients for all layers {1: dLdW1, 2: dLdW2,...}

    def update_params(self, learning_rate):
        # Update W, b using gradients stored in each layer during the back_propagation
        print("\n######## UPDATING PARAMETERS: ########") if self.debug else None
        for i, layer in enumerate(self.Layers):
            layer_i = i + 1
            layer.weights_matrix -= learning_rate * layer.dL_dW
            layer.bias_vector -= learning_rate * layer.dL_db
            self.debug and print(f"Updated Layer {layer_i}")


    ############# Batch methods (placeholders, need implementation) #############
    def forward(self, x_batch):  # Batch Forward
        """input_batch = List of input vectors/arrays"""
        outputs = []
        caches = []
        for x in x_batch:
             y_pred, cache = self.forward_propogation(x)
             outputs.append(y_pred)
             caches.append(cache)
        return outputs, caches # Returns list of outputs and list of caches
    
    def backward(self, y_batch_labels, caches):
        """
        Vectorized batch back-propagation.
        - y_batch_labels: list of labels (ints)
        - caches: list of dicts, each with 'A0','Z1','A1',…,'ZL','AL'
        """
        L = len(self.Layers)    # number of layers
        m = len(y_batch_labels) # batch size

        # building one-hot label matrix Y: shape (output_dim, m)
        Y = np.hstack([self.one_hot_encoding[y] for y in y_batch_labels])  # (n^l, m)

        # Stacking all A's and Z's into dicts of 2D arrays:
        A = { 0: np.hstack([c['A0'] for c in caches]) }    # A0: (input_dim, m)
        Z = {}
        for l in range(1, L+1):
            Z[l] = np.hstack([c[f'Z{l}'] for c in caches])  # Zl: (n_l, m)
            A[l] = np.hstack([c[f'A{l}'] for c in caches])  # Al: (n_l, m)
        # Gradient dictionaries:
        dW = {}
        db = {}

        # Output-layer delta/error signal
        last = self.Layers[-1]
        if last.activation == "softmax":
            delta = A[L] - Y    # shape -> (n^L, m)
        else:   # raising not implemented:
            raise NotImplementedError("Only softmax+CE supported in vectorized form")

        # 5) Gradients for layer L
        dW[L] = (delta @ A[L-1].T) / m  # (n^l, n^(l-1))
        db[L] = np.sum(delta, axis=1, keepdims=True) / m # (n^l, 1)

        for l in range(L-1, 0, -1):
            W_next = self.Layers[l].weights_matrix # W^{l+1}
            # derivative of activation at layer l
            if   self.Layers[l-1].activation == "sigmoid":
                d_act = sigmoid_derivative(A[l])# shape (n_l, m)
            elif self.Layers[l-1].activation == "relu":
                d_act = relu_derivative(Z[l])   # shape (n_l, m)
            else:# raising not implemented:
                raise NotImplementedError
            delta = (W_next.T @ delta) * d_act  # (n_l, m)
            # gradients
            dW[l] = (delta @ A[l-1].T) / m # (n_l, n^(l-1))
            db[l] = np.sum(delta, axis=1, keepdims=True) / m

        # storing layers:
        for l in range(1, L+1):
            self.Layers[l-1].dL_dW = dW[l]
            self.Layers[l-1].dL_db = db[l]

        return dW, db
            

    def train(self, x_batch, y_batch_labels, learning_rate):
        """ 
        Performs one training step on a batch: forward, backward (average grads), update.
        Args:
            x_batch (list): List of input vectors for the batch.
            y_batch_labels (list): List of true class labels for the batch.
            learning_rate (float): The learning rate for the update step.
        Returns:
            float: Average loss over the batch.
        """
        # 1. Forward pass for the whole batch
        y_preds, caches = self.forward(x_batch) # Get list of predictions and caches
        
        # 2. Calculate loss for the batch (optional, but good for monitoring)
        total_loss = 0
        for i in range(len(y_batch_labels)):
            y_true_one_hot = self.one_hot_encoding[y_batch_labels[i]]
            total_loss += cross_entropy(y_preds[i], y_true_one_hot)
        average_loss = total_loss / len(y_batch_labels)
        print(f"\nBatch Average Loss: {average_loss:.4f}") if self.debug else None
        
        # 3. Backward pass for the whole batch (computes and stores average gradients in layers)
        self.backward(y_batch_labels, caches)
        
        # 4. Update parameters using the stored average gradients
        self.update_params(learning_rate)

        return average_loss

    def network_info(self):
        total_parameters = 0
        print("\n######## NETWORK INFO: ########")
        print(f"Input Size: {self.Layers_definition[0]}")
        for i, layer in enumerate(self.Layers):
            print(f"\n--------- Layer {i+1} ----------")
            total_parameters += layer.layer_info()
            print()
        print(f"Total Learnable Model Parameters = {total_parameters}")

# inputs_labels_to_lists was for PyTorch DataLoader, might not be used directly with numpy-based batching
def inputs_labels_to_lists(dataloader):
    """
    Converts batched inputs and labels into two Python lists:
    - One list of flattened image pixel intensities
    - One list of labels
    
    Args:
    dataloader : torch.utils.data.DataLoader
        DataLoader providing (inputs, labels) batches.
        
    Returns:
    tuple(list, list)
        (flattened_inputs, labels_list)
    """
    flattened_inputs = []
    labels_list = []

    for inputs, labels in dataloader:
        # inputs: (batch_size, 1, 28, 28)
        inputs = inputs.squeeze(1)               # (batch_size, 28, 28)
        inputs = inputs.view(inputs.size(0), -1)  # (batch_size, 784)
        
        batch_inputs = inputs.tolist()            # list of lists
        batch_labels = labels.tolist()            # list of integers
        
        flattened_inputs.extend(batch_inputs)
        labels_list.extend(batch_labels)
    
    return flattened_inputs, labels_list

def test():
    # Example: 784 input features (e.g., MNIST), 128 hidden (Sigmoid), 10 output (Softmax)
    layers_def = [784, (128, "sigmoid"), (10, "softmax")] 
    MLP = NN(layers_def, debug=True) # Enable debug prints
    MLP.network_info()

    # Testing Single Forward/Backward Pass
    print("\n--- Testing Single Pass ---")
    # Create a dummy input vector (e.g., flattened image)
    test_inp_vec = np.random.rand(784) 
    # Create a dummy true label (needs to be a key in one_hot_encoding)
    true_label = 3 

    print(f"\nInput Vector Shape = {test_inp_vec.shape}")
    y_pred, cache = MLP.forward_propogation(test_inp_vec)
    print(f"\nPrediction (y_pred) shape: {y_pred.shape}")
    # print("Prediction (y_pred):\n", y_pred.T)
    print(f"Predicted class (highest probability): {np.argmax(y_pred)}")
    
    loss = cross_entropy(y_pred, MLP.one_hot_encoding[true_label])
    print(f"\nLoss for this sample: {loss:.4f}")

    # Perform backpropagation
    grads_W, grads_b = MLP.back_propogation(true_label, cache) 
    
    # print shapes of gradients for verification
    # print("\nGradients calculated (Shapes):")
    # for l in range(1, len(MLP.Layers) + 1):
    #    print(f" Layer {l}: dW shape {grads_W[l].shape}, db shape {grads_b[l].shape}")
    # print(f" Layer {l}: Stored dW shape {MLP.Layers[l-1].dL_dW.shape}, Stored db shape {MLP.Layers[l-1].dL_db.shape}")
        
    # --- Test Training on a Dummy Batch ---
    print("\n--- Testing Batch Training ---")
    # Create a dummy batch
    batch_size = 4
    dummy_x_batch = [np.random.rand(784) for _ in range(batch_size)]
    dummy_y_batch_labels = [np.random.randint(0, 10) for _ in range(batch_size)] # Random labels 0-9
    
    learning_rate = 0.01
    print(f"Training on batch of size {batch_size} with LR={learning_rate}")
    avg_loss = MLP.train(dummy_x_batch, dummy_y_batch_labels, learning_rate)
    print(f"Training step completed. Average Loss: {avg_loss:.4f}")

    # You could run another forward pass here to see if the loss decreased slightly


if __name__=="__main__":
    test()