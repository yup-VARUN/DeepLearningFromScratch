import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys # Import sys to print progress

# ===================== Utility Functions ===================== #

def relu(x):
    """ Rectified Linear Unit activation function """
    return np.maximum(0, x)

def softmax(x):
    """ Softmax activation function """
    # Subtract max for numerical stability to prevent overflow with large exponents
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Helper function for cross-entropy loss
def cross_entropy_loss(predictions, targets_one_hot):
    """ Computes cross-entropy loss """
    # Ensure predictions are not exactly zero or one to avoid log(0) or log(inf)
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    # Loss is - (1/N) * sum(targets * log(predictions))
    N = predictions.shape[0]
    loss = -np.sum(targets_one_hot * np.log(predictions)) / N
    return loss

# ===================== Data Loading ===================== #
def dataloader(train_dataset, test_dataset, batch_size=64):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)
    print("Training samples:", len(train_dataset))
    print("Testing samples:", len(test_dataset))
    return dataloader(train_dataset, test_dataset)

# ===================== CNN Structure ===================== #
class CNN:
    def __init__(self, input_size, num_filters, kernel_size, fc_output_size, lr):
        """
        Initializes the CNN model.

        Args:
            input_size (tuple): (channels, height, width) of the input image.
            num_filters (int): Number of filters in the convolutional layer.
            kernel_size (int): Size of the square convolutional kernel.
            fc_output_size (int): Number of output classes for the fully connected layer (e.g., 10 for MNIST).
            lr (float): Learning rate for parameter updates.
        """
        self.input_shape = input_size # (C_in, H_in, W_in)
        self.num_filters = num_filters # C_out
        self.kernel_size = kernel_size # K_h, K_w
        self.fc_output_size = fc_output_size # Number of classes
        self.lr = lr

        # --- Convolutional Layer Initialization ---
        C_in, H_in, W_in = self.input_shape
        K_h, K_w = self.kernel_size, self.kernel_size

        # Calculate output dimensions after convolution (assuming stride 1, valid padding)
        self.conv_output_height = H_in - K_h + 1
        self.conv_output_width = W_in - K_w + 1

        # Shape of convolutional weights: (num_filters, C_in, K_h, K_w) -> (C_out, C_in, K_h, K_w)
        # User variable name: c_weights
        # Initialize with small random values (e.g., using He initialization scaling for ReLU)
        # Factor is sqrt(2 / fan_in)
        fan_in = C_in * K_h * K_w
        std_dev = np.sqrt(2.0 / fan_in)
        self.c_weights = np.random.randn(self.num_filters, C_in, self.kernel_size, self.kernel_size) * std_dev
        # Shape of convolutional biases: (num_filters,) -> (C_out,)
        # User variable name: c_bias
        self.c_bias = np.zeros(self.num_filters)

        # --- Fully Connected Layer Initialization ---
        # Calculate the size of the flattened output after convolution (and pooling, if any - none here)
        self.flattened_size = self.num_filters * self.conv_output_height * self.conv_output_width

        # Shape of fully connected weights: (flattened_size, fc_output_size)
        # User variable name: weights
        # Initialize with small random values
        fan_in_fc = self.flattened_size
        std_dev_fc = np.sqrt(2.0 / fan_in_fc) # Using He initialization for weights connected to ReLU output
        self.weights = np.random.randn(self.flattened_size, self.fc_output_size) * std_dev_fc
        # Shape of fully connected biases: (fc_output_size,)
        # User variable name: bias
        self.bias = np.zeros(self.fc_output_size)

        # Variables to store intermediate values for backward pass (using user's names)
        self.c_z1 = None        # Output of convolution before activation
        self.c_a1 = None        # Output of ReLU activation
        self.flattened_a1 = None # Output after flattening
        self.z2 = None          # Output of fully connected layer before activation
        self.a2 = None          # Output of Softmax activation (predictions)


    def conv2d_forward(self, image, kernel, bias):
        """ Helper function for 2D convolution (Forward Pass - assumes stride 1, valid padding) """
        # image shape (N, C_in, H_in, W_in)
        # kernel shape (C_out, C_in, K_h, K_w)
        # bias shape (C_out,)

        N, C_in, H_in, W_in = image.shape
        C_out, _, K_h, K_w = kernel.shape

        H_out = H_in - K_h + 1
        W_out = W_in - K_w + 1

        output = np.zeros((N, C_out, H_out, W_out))

        # Simple loop-based convolution (can be slow for larger inputs/batches)
        for n in range(N):
            for co in range(C_out):
                for ho in range(H_out):
                    for wo in range(W_out):
                        h_start = ho
                        h_end = ho + K_h
                        w_start = wo
                        w_end = wo + K_w
                        image_patch = image[n, :, h_start:h_end, w_start:w_end] # Shape (C_in, K_h, K_w)
                        output[n, co, ho, wo] = np.sum(image_patch * kernel[co]) + bias[co]

        return output

    def conv2d_backward_weights(self, image, grad_output):
        """ Helper function for 2D convolution (Backward Pass - Weights Gradient) """
        # image shape (N, C_in, H_in, W_in)
        # grad_output shape (N, C_out, H_out, W_out)

        N, C_in, H_in, W_in = image.shape
        _, C_out, H_out, W_out = grad_output.shape
        K_h, K_w = self.kernel_size, self.kernel_size

        dL_dcW = np.zeros((C_out, C_in, K_h, K_w))

        # Calculate gradient dL_dcW by cross-correlating input image with grad_output
        # Sum over batch dimension and spatial output dimensions
        for co in range(C_out):
            for ci in range(C_in):
                for kh in range(K_h):
                    for kw in range(K_w):
                        # Select corresponding patches from the input image
                        image_patch = image[:, ci, kh : kh + H_out, kw : kw + W_out] # Shape (N, H_out, W_out)
                        # Select corresponding patches from the gradient
                        grad_patch = grad_output[:, co, :, :] # Shape (N, H_out, W_out)
                        # Perform element-wise multiplication and sum over N, H_out, W_out
                        dL_dcW[co, ci, kh, kw] = np.sum(image_patch * grad_patch)

        # Average gradient over the batch
        dL_dcW /= N
        return dL_dcW

    def conv2d_backward_bias(self, grad_output):
        """ Helper function for 2D convolution (Backward Pass - Bias Gradient) """
        # grad_output shape (N, C_out, H_out, W_out)
        # Sum over N, H_out, W_out and average over N
        dL_dc_bias = np.sum(grad_output, axis=(0, 2, 3)) / grad_output.shape[0]
        return dL_dc_bias


    def forward(self, x):
        """ Forward propogation """
        # x shape: (batch_size, channels, height, width) - e.g., (N, 1, 28, 28) for MNIST

        # 1. Convolutional Layer
        # Computing convolution result
        # User variable name: c_z1
        self.c_z1 = self.conv2d_forward(x, self.c_weights, self.c_bias)

        # 2. ReLU Activation
        # Apply ReLU element-wise
        # User variable name: c_a1
        self.c_a1 = relu(self.c_z1)

        # 3. Flattening
        # Reshape c_a1 from (N, C_out, H_out, W_out) to (N, C_out * H_out * W_out)
        N = self.c_a1.shape[0]
        # User variable name: flattened_a1
        self.flattened_a1 = self.c_a1.reshape(N, self.flattened_size)

        # 4. Fully Connected Layer
        # Perform linear transformation: z2 = flattened_a1 @ weights + bias
        # User variable name: z2
        self.z2 = np.dot(self.flattened_a1, self.weights) + self.bias

        # 5. Softmax Activation
        # Apply softmax to get probabilities
        # User variable name: a2 (predictions)
        self.a2 = softmax(self.z2)

        # The output of the forward pass is the predicted probabilities
        outputs = self.a2

        return outputs

    def backward(self, x, y, pred):
        """ Backward propagation """
        # x: Original input batch (N, C_in, H_in, W_in)
        # y: True labels (N,)
        # pred: Predicted probabilities from forward pass (self.a2) (N, num_classes)

        N = x.shape[0] # Batch size
        num_classes = self.fc_output_size

        # 1. one-hot encode the labels
        # Create a zero array of shape (N, num_classes)
        # User variable name: one_hot_y
        one_hot_y = np.zeros((N, num_classes))
        # Set the element at the true class index to 1 for each sample in the batch
        one_hot_y[np.arange(N), y] = 1

        # 2. Calculate softmax cross-entropy loss gradient
        # The gradient of the loss with respect to the output of the softmax layer (a2)
        # This is a simplified gradient when using combined Softmax and Cross-Entropy loss
        # dL/da2 = pred - one_hot_y
        # User variable name: dL_da2
        dL_da2 = pred - one_hot_y # Shape (N, num_classes)

        # 3. Calculate fully connected layer gradient
        # The gradient with respect to z2 (output before softmax) is the same as dL_da2
        dL_dz2 = dL_da2 # Shape (N, num_classes)

        # Gradient with respect to fully connected weights (dL_dW)
        # dL/dW = (flattened_a1).T @ dL_dz2
        # Shape: (flattened_size, N) @ (N, num_classes) -> (flattened_size, num_classes)
        # Average gradient over the batch
        # User variable name: dL_dW
        dL_dW = np.dot(self.flattened_a1.T, dL_dz2) / N

        # Gradient with respect to fully connected biases (dL_dB)
        # dL/dB = sum(dL_dz2) over the batch dimension
        # Shape: (num_classes,)
        # Average gradient over the batch
        # User variable name: dL_dB
        dL_dB = np.sum(dL_dz2, axis=0) / N

        # Gradient propogated back through the fully connected layer to the flattened layer
        # dL/d_flattened_a1 = dL_dz2 @ weights.T
        # Shape: (N, num_classes) @ (num_classes, flattened_size) -> (N, flattened_size)
        # User variable name: dL_dflattened
        dL_dflattened = np.dot(dL_dz2, self.weights.T) # Shape (N, flattened_size)

        # 4. Backpropagate through ReLU
        # Reshape the gradient from the flattened layer back to the shape of the ReLU output (c_a1)
        # Shape: (N, flattened_size) -> (N, C_out, H_out, W_out)
        # This gradient is with respect to c_a1 (output of ReLU)
        # User variable name: dL_drelu_out
        dL_drelu_out = dL_dflattened.reshape(self.c_a1.shape) # Shape (N, C_out, H_out, W_out)

        # Apply the gradient of the ReLU activation function
        # The derivative of ReLU is 1 if the input (c_z1) was > 0, and 0 otherwise.
        # dL/dc_z1 = dL/dc_a1 * d(relu)/dc_z1
        # Element-wise multiplication by the mask (c_z1 > 0)
        # This gradient is with respect to c_z1 (output of convolution before ReLU)
        # User variable name: dL_dc_out
        dL_dc_out = dL_drelu_out * (self.c_z1 > 0) # Shape (N, C_out, H_out, W_out)

        # 5. Calculate convolution kernel gradient
        # Gradient with respect to convolutional weights (dL_dcW)
        # This requires cross-correlation between the original input x and dL_dc_out
        # User variable name: dL_dcW
        dL_dcW = self.conv2d_backward_weights(x, dL_dc_out) # Shape (C_out, C_in, K_h, K_w)

        # Gradient with respect to convolutional biases
        # This requires summing dL_dc_out over the batch and spatial dimensions, and averaging over the batch
        # User variable name for bias gradient was not explicitly listed for step 5's output.
        # We will calculate it and apply it to self.c_bias.
        conv_bias_gradient = self.conv2d_backward_bias(dL_dc_out) # Shape (C_out,)


        # 6. Update parameters
        # Update convolutional weights using the calculated gradient and learning rate
        self.c_weights -= self.lr * dL_dcW

        # Update convolutional biases using the calculated gradient and learning rate
        self.c_bias -= self.lr * conv_bias_gradient # Using the internally calculated bias gradient

        # Update fully connected weights using the calculated gradient and learning rate
        self.weights -= self.lr * dL_dW

        # Update fully connected biases using the calculated gradient and learning rate
        self.bias -= self.lr * dL_dB

        # The backward function typically updates parameters and doesn't return a value.


    def train(self, x, y):
        """
        Performs a single training step (forward pass, loss calculation, backward pass, parameter update).

        Args:
            x: Input batch (N, C_in, H_in, W_in)
            y: True labels (N,)

        Returns:
            The calculated loss for the batch.
        """
        # call forward function to get predictions
        # User variable name for predictions: a2 (stored internally in self.a2)
        # I've used local var - pred to hold the return value of forward for clarity
        pred = self.forward(x) # here pred is the same as "self.a2" after the forward pass

        # calculate loss
        # Need one-hot encoded labels for cross-entropy loss calculation
        N = x.shape[0]
        num_classes = self.fc_output_size
        # User variable name: one_hot_y
        one_hot_y = np.zeros((N, num_classes))
        one_hot_y[np.arange(N), y] = 1

        # Use the cross_entropy_loss helper function to calculate the loss for the batch
        # User variable name for loss: loss
        loss = cross_entropy_loss(pred, one_hot_y)

        # call backward function to compute gradients and update parameters
        self.backward(x, y, pred) # Pass original input x, true labels y, and predictions pred

        # Return the calculated loss for monitoring training progress
        return loss


if __name__ == "__main__":
    # Define hyperparameters for the CNN and training process
    input_size = (1, 28, 28) # MNIST is 1 channel, 28x28 pixels
    num_filters = 1
    kernel_size = 12
    fc_output_size = 8
    learning_rate = 0.0005
    batch_size = 64
    num_epochs = 5

    # Load the MNIST dataset using the defined data loading functions
    train_loader, test_loader = load_data()

    # Initialize the CNN model with the specified hyperparameters
    cnn_model = CNN(input_size, num_filters, kernel_size, fc_output_size, learning_rate)

    # Print model architecture details
    print("\nCNN Model Initialized:")
    print(f"\tInput Shape: {cnn_model.input_shape}")
    print(f"\tNum Filters: {cnn_model.num_filters}")
    print(f"\tKernel Size: {cnn_model.kernel_size}")
    print(f"\tConv Output Spatial Dimensions: ({cnn_model.conv_output_height}, {cnn_model.conv_output_width})")
    print(f"\tFlattened Size (Input to FC): {cnn_model.flattened_size}")
    print(f"\tFC Output Size (Num Classes): {cnn_model.fc_output_size}")
    print(f"\tLearning Rate: {cnn_model.lr}")
    print(f"\tConv Weights Shape: {cnn_model.c_weights.shape}")
    print(f"\tConv Bias Shape: {cnn_model.c_bias.shape}")
    print(f"\tFC Weights Shape: {cnn_model.weights.shape}")
    print(f"\tFC Bias Shape: {cnn_model.bias.shape}")


    # --- Training Loop ---
    print("\nStarting Training...")
    for epoch in range(num_epochs):
        # Keep track of the total loss for the epoch
        epoch_loss = 0
        # Iterate over the training data in batches
        batch_count = 0
        for images, labels in train_loader:
            # Convert PyTorch tensors from DataLoader to NumPy arrays
            # Ensure the data types are appropriate for NumPy computations
            images_np = images.numpy().astype(np.float32)
            labels_np = labels.numpy().astype(np.int64) # Labels are typically integer class indices

            # Perform a single training step (forward, loss, backward, update)
            loss = cnn_model.train(images_np, labels_np)
            epoch_loss += loss
            batch_count += 1

            # Print training progress periodically
            if (batch_count % 100) == 0: # Print every 100 batches
                 # Use sys.stdout.write and flush for printing on the same line
                 sys.stdout.write(f'\r  Epoch {epoch+1}/{num_epochs}, Batch {batch_count}/{len(train_loader)}, Loss: {loss:.4f}')
                 sys.stdout.flush()

        # Calculate and print the average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'\r  Epoch {epoch+1}/{num_epochs} Training Complete, Avg Loss: {avg_epoch_loss:.4f}')


    print("\nTraining Finished.")

    # --- Testing Loop ---
    print("\nStarting Testing...")
    correct_predictions = 0
    total_samples = 0

    # Iterate over the testing data in batches
    for images, labels in test_loader:
        # Convert PyTorch tensors to NumPy arrays
        images_np = images.numpy().astype(np.float32)
        labels_np = labels.numpy().astype(np.int64)

        # Perform a forward pass to get predictions (probabilities) for the test batch
        predictions = cnn_model.forward(images_np) # predictions shape (N, num_classes)

        # Get the predicted class index for each sample by finding the index with the highest probability
        predicted_classes = np.argmax(predictions, axis=1) # shape (N,)

        # Compare predicted classes with true labels and count correct predictions
        correct_predictions += np.sum(predicted_classes == labels_np)
        total_samples += len(labels_np) # Add the number of samples in the current batch to total

    # Calculate and print the final testing accuracy
    accuracy = correct_predictions / total_samples
    print(f"Testing Complete. Accuracy: {accuracy:.4f}")