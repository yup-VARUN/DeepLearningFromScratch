import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ===================== Utility Functions ===================== #

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def cross_entropy_loss(predictions, targets_one_hot):
    # avoiding log(0) or log(inf)
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

        # Xavier initialization for the C layer too:
        fan_in  = C_in * K_h * K_w
        fan_out = self.num_filters * K_h * K_w
        limit   = np.sqrt(6.0 / (fan_in + fan_out))
        # weights ∼ Uniform(–limit, +limit)
        self.c_weights = np.random.uniform(
            -limit, limit,
            size=(self.num_filters, C_in, self.kernel_size, self.kernel_size)
        )
        # biases still zeros
        self.c_bias = np.zeros(self.num_filters)

        ####### Fully Connected Layer Initialization #######
        # Calculate the size of the flattened output after convolution (and pooling, if any - none here)
        self.flattened_size = self.num_filters * self.conv_output_height * self.conv_output_width

        # Xavier
        fan_in_fc  = self.flattened_size
        fan_out_fc = self.fc_output_size
        limit = np.sqrt(6.0 / (fan_in_fc + fan_out_fc))
        # weights ∼ Uniform(–limit, +limit)
        self.weights = np.random.uniform(-limit, limit, size=(fan_in_fc, fan_out_fc))
        # biases still zeros
        self.bias = np.zeros(self.fc_output_size)

        # Variables to store intermediate values for backward pass
        self.c_z1 = None
        self.c_a1 = None
        self.flattened_a1 = None
        self.z2 = None
        self.a2 = None


    def conv2d_forward(self, image, kernel, bias):
        """Helper function for 2D convolution (Forward Pass - assuming stride 1 and validted padding)
        image shape (N, C_in, H_in, W_in)
        kernel shape (C_out, C_in, K_h, K_w)
        bias shape (C_out,)"""

        N, C_in, H_in, W_in = image.shape
        C_out, _, K_h, K_w = kernel.shape

        H_out = H_in - K_h + 1
        W_out = W_in - K_w + 1

        output = np.zeros((N, C_out, H_out, W_out))

        # main convolution loop:
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
        """ Helper function for 2D convolution (Backward Pass - Weights Gradient) 
            image shape (N, C_in, H_in, W_in)
            grad_output shape (N, C_out, H_out, W_out)
        """

        N, C_in, H_in, W_in = image.shape
        _, C_out, H_out, W_out = grad_output.shape # we don't need the first element
        K_h, K_w = self.kernel_size, self.kernel_size

        dL_dcW = np.zeros((C_out, C_in, K_h, K_w))

        # dL_dcW calculation by cross-correlating input image with grad_output:
        # Summing over batch dimension and spatial output dimensions
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
        """ Helper function for 2D convolution (Backward Pass - Bias Gradient)
        grad_output shape (N, C_out, H_out, W_out)
        Sum over N, H_out, W_out and average over N
        """
        dL_dc_bias = np.sum(grad_output, axis=(0, 2, 3)) / grad_output.shape[0]
        return dL_dc_bias


    def forward(self, x):
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
        """ Backward propagation 
        x: Original input batch (N, C_in, H_in, W_in)
        y: True labels (N,)
        pred: Predicted probabilities from forward pass (self.a2) (N, num_classes) """
        N = x.shape[0] # Batch size
        num_classes = self.fc_output_size

        # 1. one-hot encode the labels
        # Create a zero array of shape (N, num_classes)
        one_hot_y = np.zeros((N, num_classes))
        one_hot_y[np.arange(N), y] = 1

        # 2. Calculate softmax cross-entropy loss gradient
        # The gradient of the loss with respect to the output of the softmax layer (a2)
        # This is a simplified gradient when using combined Softmax and Cross-Entropy loss
        # dL/da2 = pred - one_hot_y
        dL_da2 = pred - one_hot_y # Shape (N, num_classes)

        # 3. Calculate fully connected layer gradient
        # grad wrt z2 (output before softmax) is the same as dL_dA2(instead of L I've been using 2 in CNN.py)
        dL_dz2 = dL_da2 # Shape (N, num_classes)
        dL_dW = np.dot(self.flattened_a1.T, dL_dz2) / N
        dL_dB = np.sum(dL_dz2, axis=0) / N

        # Gradient propogated back through the fully connected layer to the flattened layer
        dL_dflattened = np.dot(dL_dz2, self.weights.T) # Shape (N, flattened_size)

        # 4. Backpropagate through ReLU
        dL_drelu_out = dL_dflattened.reshape(self.c_a1.shape) # Shape (N, C_out, H_out, W_out)

        # Apply the gradient of the ReLU activation function
        # The derivative of ReLU is 1 if the input (c_z1) was > 0, and 0 otherwise.
        # dL/dc_z1 = dL/dc_a1 * d(relu)/dc_z1
        # Doing selective element-wise multiplication by the mask (c_z1 > 0)
        dL_dc_out = dL_drelu_out * (self.c_z1 > 0) # Shape (N, C_out, H_out, W_out)

        # 5. Calculate convolution kernel gradient
        # Gradient with respect to convolutional weights (dL_dcW)
        # cross-correlation between the original input x and dL_dc_out
        dL_dcW = self.conv2d_backward_weights(x, dL_dc_out) # Shape (C_out, C_in, K_h, K_w)

        # Gradient with respect to convolutional biases
        # summing dL_dc_out over the batch and spatial dimensions, and averaging over the batch
        # calculate and apply directly to self.c_bias
        conv_bias_gradient = self.conv2d_backward_bias(dL_dc_out) # Shape (C_out,)
        # 6. Pparameters Updation:
        self.c_weights -= self.lr * dL_dcW
        self.c_bias -= self.lr * conv_bias_gradient # Using the internally calculated bias gradient
        self.weights -= self.lr * dL_dW
        self.bias -= self.lr * dL_dB

    def train(self, x, y):
        """
        Performs a single training step (forward pass, loss calculation, backward pass, parameter update).

        Args:
            x: Input batch (N, C_in, H_in, W_in)
            y: True labels (N,)

        Returns:
            The calculated loss for the batch.
        """
        pred = self.forward(x) # here pred is the same as "self.a2" after the forward pass
        # Need one-hot encoded labels for cross-entropy loss calculation
        N = x.shape[0]
        num_classes = self.fc_output_size
        one_hot_y = np.zeros((N, num_classes))
        one_hot_y[np.arange(N), y] = 1
        loss = cross_entropy_loss(pred, one_hot_y)
        self.backward(x, y, pred) # Pass original input x, true labels y, and predictions pred
        return loss


if __name__ == "__main__":
    # Hyperparams :))
    input_size = (1, 28, 28) # 28x28 sized matrix
    num_filters = 1
    kernel_size = 12
    fc_output_size = 10
    learning_rate = 0.001
    batch_size = 128
    num_epochs = 5

    train_loader, test_loader = load_data()

    cnn_model = CNN(input_size, num_filters, kernel_size, fc_output_size, learning_rate)

    # printing model architecture details
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


    ##### Training Loop #####
    print("\nStarting Training...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        batch_count = 0
        for images, labels in train_loader:
            # PyTorch tensors -> NumPy arrays
            # making sure dtypes are appropriate for NumPy computations
            images_np = images.numpy().astype(np.float32)
            labels_np = labels.numpy().astype(np.int64) # Labels are typically integer class indices

            # Single training step (forward, loss, backward, update)
            loss = cnn_model.train(images_np, labels_np)
            epoch_loss += loss
            batch_count += 1
            if (batch_count % 100) == 0: # Print every 100 batches
                 print(f' Epoch {epoch+1}/{num_epochs}, Batch {batch_count}/{len(train_loader)}, Loss: {loss:.4f}')

        # Calculate and print the average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'\r  Epoch {epoch+1}/{num_epochs} Training Complete, Avg Loss: {avg_epoch_loss:.4f}')


    print("\nTraining Finished.")

    ##### Testing Loop #####
    print("\nStarting Testing...")
    correct_predictions = 0
    total_samples = 0

    # Iterate over the testing data in batches
    for images, labels in test_loader:
        # Convert PyTorch tensors to NumPy arrays
        images_np = images.numpy().astype(np.float32)
        labels_np = labels.numpy().astype(np.int64)
        predictions = cnn_model.forward(images_np) # predictions shape (N, num_classes)
        predicted_classes = np.argmax(predictions, axis=1) # shape (N,)
        correct_predictions += np.sum(predicted_classes == labels_np)
        total_samples += len(labels_np) # Add the number of samples in the current batch to total

    # Calculate and print the final testing accuracy
    accuracy = correct_predictions / total_samples
    print(f"Testing Complete. Accuracy: {accuracy:.4f}")