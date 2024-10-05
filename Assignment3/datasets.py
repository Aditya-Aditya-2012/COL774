import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# Custom Dataset class to handle CIFAR100 from local .pkl files
class CIFAR100Dataset(Dataset):
    def __init__(self, data_path, transform=None):
        # Load the data from the pickle file
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)  # `self.data` is a list of tuples (image, label)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Access the image and label from the tuple
        image, label = self.data[idx]
        
        # Apply transformations to the image (if any)
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Function to compute mean and std for normalization
def compute_mean_std(cifar100_dataset):
    # Stack all image channels along the depth axis
    all_images = torch.stack([cifar100_dataset[i][0] for i in range(len(cifar100_dataset))], dim=0)
    
    # Compute the mean and std for each channel
    mean = torch.mean(all_images, dim=[0, 2, 3])  # mean across batch, height, and width (leave channels)
    std = torch.std(all_images, dim=[0, 2, 3])  # std across batch, height, and width (leave channels)
    
    return mean, std

# Function to load the dataset
def load_dataset(batch_size, path_train, path_test):
    # Load the raw dataset to compute mean and std
    raw_train_dataset = CIFAR100Dataset(data_path=path_train)
    
    # Compute the mean and std from the raw dataset
    mean, std = compute_mean_std(raw_train_dataset)
    print(f"Computed Mean: {mean}")
    print(f"Computed Std: {std}")
    
    # Use computed mean and std for normalization
    normalizer = transforms.Normalize(mean=mean, std=std)
    
    # Train transform with AutoAugment
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),  # Applying AutoAugment
        normalizer
    ])
    
    # Test transform with normalization only
    transform_test = transforms.Compose([
        normalizer
    ])
    
    # Load data with transformations
    train_dataset = CIFAR100Dataset(data_path=path_train, transform=transform_train)
    test_dataset = CIFAR100Dataset(data_path=path_test, transform=transform_test)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

# Example usage:
train_path= '/home/civil/btech/ce1210494/A3_data/train.pkl'
test_path = '/home/civil/btech/ce1210494/A3_data/test.pkl'
rain_loader, test_loader = load_dataset(batch_size=64, path_train=train_path, path_test=test_path)
