import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def dataloader(train_dataset, test_dataset, batch_size=128):
    """
    Creates DataLoader objects for both training and testing datasets.
    
    Parameters:
    -----------
    train_dataset : torch.utils.data.Dataset
        The training dataset for training model
    test_dataset : torch.utils.data.Dataset
        The testing dataset for testing model
    batch_size : int, optional (default=128)
        Number of samples per batch
        
    Returns:
    --------
    tuple(DataLoader, DataLoader)
        Train and test dataloaders
        
    Example:
    --------
    # If you want to use a different batch size:
    train_loader, test_loader = dataloader(train_dataset, test_dataset, batch_size=64)
    """
    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True)  # shuffle=True for random batch sampling
    
    test_loader = DataLoader(dataset=test_dataset, 
                           batch_size=batch_size, 
                           shuffle=False)  # shuffle=False for consistent testing

    for i in iter(test_loader):
        # each element in test_loader iterable contains two tensors
        # i[0] -> first tensor contains normalized pixel intensities) first image
        # i[1] -> second tensor contains labels
        first_image = i[0][0, 0, :, :]
        print(first_image)
        print(i[1])
        break

    print("\n\n\n")

    flattened_images = []

    for images, labels in train_loader:
        # images shape: (batch_size, 1, 28, 28)
        
        # Remove the channel dimension (1 -> nothing)
        images = images.squeeze(1)  # Now shape is (batch_size, 28, 28)
        
        # Flatten each image to (batch_size, 784)
        images = images.view(images.size(0), -1)
        
        # Convert to nested list
        batch_list = images.tolist()  # list of lists (each inner list length 784)
        
        # Append to overall list
        flattened_images.extend(batch_list)
        print(flattened_images[0])
        break



    return train_loader, test_loader


def load_data():
    # Define the preprocessing transformations
    transform = transforms.Compose([transforms.ToTensor(),  # Convert images to tensor and scale to [0, 1]
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    # Load training data
    train_dataset = torchvision.datasets.MNIST(
        root="./data/mnist",  # Data will be downloaded here
        train=True,  # Specify training dataset
        download=True,  # Download if not already present
        transform=transform  # Apply the preprocessing
    )
    
    # Load testing data
    test_dataset = torchvision.datasets.MNIST(
        root="./data/mnist",
        train=False,  # Specify test dataset
        download=True,
        transform=transform
    )

    print("The number of training data:", len(train_dataset))  # Should print 60000
    print("The number of testing data:", len(test_dataset))   # Should print 10000

    return dataloader(train_dataset, test_dataset)  # using above designed dataloader() function for here

load_data()