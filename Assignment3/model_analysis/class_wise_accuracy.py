import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import numpy as np
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from datasets import CIFAR100Dataset, compute_mean_std
from models.pyramidnet import ShakePyramidNet

torch.manual_seed(0)

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print("Data loading completed.")
    return data

def main():
    if len(sys.argv) != 3:
        print("Usage: python class_wise_accuracy.py model.pth train.pkl")
        sys.exit(1)

    model_path = sys.argv[1]
    train_file = sys.argv[2]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    print("Loading the model...")
    model = ShakePyramidNet(depth=110, alpha=270, label=100)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load and prepare training data
    raw_train = CIFAR100Dataset(data_path=train_file)
    mean, std = compute_mean_std(raw_train)

    print("Loading and preparing training data...")
    train_data = load_data(train_file)
    
    # Assuming train_data is a list of (image_tensor, label) tuples
    images = torch.stack([img for img, _ in train_data])
    labels = torch.tensor([label for _, label in train_data])

    # Normalize the images
    normalize = transforms.Normalize(mean=mean, std=std)
    normalized_images = normalize(images)
    
    train_dataset = TensorDataset(normalized_images, labels)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

    # Initialize class-wise counters
    correct_predictions = defaultdict(int)
    total_predictions = defaultdict(int)

    # Evaluate class-wise accuracy
    print("Evaluating class-wise accuracy...")
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            for target, pred in zip(targets, predicted):
                total_predictions[target.item()] += 1
                if target == pred:
                    correct_predictions[target.item()] += 1

    # Calculate accuracy for each class
    class_accuracies = {cls: (correct_predictions[cls] / total_predictions[cls]) * 100 
                        for cls in range(100)}

    # Save results to a text file
    with open("class_wise_accuracy.txt", "w") as f:
        for cls, accuracy in class_accuracies.items():
            f.write(f"Class {cls}: {accuracy:.2f}%\n")

    # Plotting class-wise accuracies with improved readability
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())

    plt.figure(figsize=(20, 8))
    plt.bar(classes, accuracies, color='skyblue')
    plt.xlabel("Class")
    plt.ylabel("Accuracy (%)")
    plt.title("Class-wise Accuracy on Training Set")
    plt.xticks(ticks=classes, labels=classes, rotation=45, ha="right", fontsize=10)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save plot
    plt.tight_layout()
    plt.savefig("class_wise_accuracy.png")
    plt.show()

    print("Class-wise accuracy calculation completed. Results saved to 'class_wise_accuracy.txt' and 'class_wise_accuracy.png'.")

if __name__ == '__main__':
    main()
