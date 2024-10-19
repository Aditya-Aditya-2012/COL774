import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from datasets import CIFAR100Dataset, get_transforms, compute_mean_std

# Configuration parameters
batch_size = 256  # Adjust as needed
num_iterations = 10  # Number of iterations to measure average loading time
num_workers_list = [1, 4, 8, 10, 16, 20, 30, 40]  # List of num_workers values to test

# Path to the training data
train_data_path = '/home/civil/btech/ce1210494/A3_data/train.pkl'

# Load a sample dataset to compute mean and std for normalization
sample_dataset = CIFAR100Dataset(data_path=train_data_path)
mean, std = compute_mean_std(sample_dataset)

# Define transformations
transform_train, _ = get_transforms(mean, std)

# Load the entire dataset with transformations applied
dataset = CIFAR100Dataset(data_path=train_data_path, transform=transform_train)

# Prepare to store the results for plotting
average_times = []

# Test data loading for different values of num_workers
for num_workers in num_workers_list:
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # Measure time for data loading
    start_time = time.time()
    for i, (images, labels) in enumerate(data_loader):
        if i >= num_iterations:
            break  # Limit to the specified number of iterations
    end_time = time.time()
    
    # Calculate average time per batch
    avg_time_per_batch = (end_time - start_time) / num_iterations
    average_times.append(avg_time_per_batch)
    
    print(f"Num Workers: {num_workers}, Average Time per Batch: {avg_time_per_batch:.4f} seconds")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(num_workers_list, average_times, marker='o', linestyle='-')
plt.xlabel('Number of Workers')
plt.ylabel('Average Time per Batch (seconds)')
plt.title('Data Loading Time vs. Number of Workers')
plt.grid(True)
plt.xticks(num_workers_list)
plt.savefig('data_loading_time_vs_workers.png')
plt.show()
