import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from MLP_helper import *

# I've removed the activation functions since they're already in the ML_helper.py which is being imported

def dataloader(train_dataset, test_dataset, batch_size=128):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_data(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)
    print("The number of training data:", len(train_dataset))
    print("The number of testing data:", len(test_dataset))
    return dataloader(train_dataset, test_dataset, batch_size=batch_size)


def main():
    # Hyperparameters :)
    input_size = 28*28
    num_epochs = 100
    learning_rate = 0.009
    batch_size = 128

    train_loader, test_loader = load_data(batch_size)

    model = NN([input_size, (160, "sigmoid"), (10, "softmax")], debug=False) # debug printing -> false
    model.network_info()

    # training the model
    print(f"\n--- Starting Training for {num_epochs} Epochs ---")
    for epoch in range(num_epochs):
        epoch_total_loss = 0.0
        num_batches = 0
        for i, (inputs_tensor, labels_tensor) in enumerate(train_loader):
            # inputs_tensor shape -> (batch_size, 1, 28, 28)
            # Converting PyTorch Tensors to list of numpy arrays & flattening
            inputs_numpy_batch = inputs_tensor.view(inputs_tensor.size(0), -1).numpy() # Shape (batch_size, 784)
            x_batch_list = [img_vector for img_vector in inputs_numpy_batch]           # List of (784,) arrays

            # Convert labels to list of integers
            y_batch_list = labels_tensor.numpy().tolist() # List of integer labels

            # 2. Perform training step using NN.train
            # NN.train handles forward, backward (calculating avg grads), and update
            batch_avg_loss = model.train(x_batch_list, y_batch_list, learning_rate)
            
            epoch_total_loss += batch_avg_loss
            num_batches += 1

            # Optional: Print progress within epoch
            if (i + 1) % 100 == 0:
                 print(f"  Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Avg Batch Loss: {batch_avg_loss:.4f}")

        # Print average loss for the epoch
        avg_epoch_loss = epoch_total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} Completed, Average Training Loss: {avg_epoch_loss:.4f}")

    # Finally, evaluate the model on the test set
    print("\n--- Evaluating Model on Test Set ---")
    correct_pred = 0
    total_pred = 0

    # Evaluation loop using PyTorch DataLoader
    for inputs_tensor, labels_tensor in test_loader:
        inputs_numpy_batch = inputs_tensor.view(inputs_tensor.size(0), -1).numpy()
        x_batch_list = [img_vector for img_vector in inputs_numpy_batch]
        true_labels = labels_tensor.numpy()
        pred_output_list, _ = model.forward(x_batch_list) 
        
        # Stacking the list of output vectors into a single batch
        # Each pred in list is (10, 1), stack -> (batch_size, 10, 1), squeeze -> (batch_size, 10)
        pred_output_batch = np.stack(pred_output_list).squeeze(axis=-1)
        
        # Get the index of the max probability (predicted class)
        predicted_labels = np.argmax(pred_output_batch, axis=1) # Shape (batch_size,)
        correct_pred += np.sum(predicted_labels == true_labels) # batch loss calculation
        total_pred += len(true_labels) # or inputs_tensor.size(0)

    # Calculating and printing final accuracy:
    test_accuracy = correct_pred / total_pred
    print(f"\nTest Accuracy: {correct_pred}/{total_pred} = {test_accuracy:.4f}")

if __name__ == "__main__":  # Program entry
    main()  

