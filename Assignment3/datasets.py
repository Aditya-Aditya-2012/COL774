import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pickle

# Custom Dataset class to handle CIFAR100 from local .pkl files
class CIFAR100Dataset(Dataset):
    def __init__(self, data_path=None, data_list=None, transform=None):
        if data_path:
            # Load the data from the pickle file
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)  # `self.data` is a list of tuples (image, label)
        elif data_list is not None:
            self.data = data_list
        else:
            self.data = []
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

# Function to get data augmentation transforms
def get_transforms(mean, std):
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transform_train, transform_val

# Function to load the dataset and split into train and validation sets
def load_dataset(batch_size, path_train, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    raw_train_dataset = CIFAR100Dataset(data_path=path_train)
    
    mean, std = compute_mean_std(raw_train_dataset)
    print(f"Computed Mean: {mean}")
    print(f"Computed Std: {std}")
    
    transform_train, transform_val = get_transforms(mean, std)
    
    full_dataset = CIFAR100Dataset(data_path=path_train, transform=None)
    
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.shuffle(indices)  # This shuffle will be consistent due to the set seed
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_data = [full_dataset[i] for i in train_indices]
    val_data = [full_dataset[i] for i in val_indices]
    
    train_dataset = CIFAR100Dataset(data_list=train_data, transform=transform_train)
    val_dataset = CIFAR100Dataset(data_list=val_data, transform=transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    return train_loader, val_loader
