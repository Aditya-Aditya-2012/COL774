import sys
MAINPATH = ".."
sys.path.append(MAINPATH)

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datasets import CIFAR100Dataset, get_transforms, load_dataset
from models.pyramidnet import ShakePyramidNet
from utility.initialize import initialize

# Set a fixed seed for reproducibility
initialize(seed=0)

def evaluate_class_wise_accuracy(model, val_loader, device):
    correct_predictions = defaultdict(int)
    total_predictions = defaultdict(int)

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            for target, pred in zip(targets, predicted):
                total_predictions[target.item()] += 1
                if target == pred:
                    correct_predictions[target.item()] += 1

    class_accuracies = {cls: (correct_predictions[cls] / total_predictions[cls]) * 100 
                        for cls in range(100)}
    return class_accuracies

def main():
    if len(sys.argv) != 3:
        print("Usage: python repeated_class_accuracy.py model.pth train.pkl")
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

    # Prepare validation data loader
    batch_size = 256
    _, val_loader = load_dataset(batch_size, path_train=train_file)

    # Perform 10 runs of class-wise accuracy calculation
    all_accuracies = []
    print("Calculating class-wise accuracy over 10 runs...")
    for i in range(10):
        class_accuracies = evaluate_class_wise_accuracy(model, val_loader, device)
        all_accuracies.append(class_accuracies)

    # Calculate mean and std of accuracies across 10 runs
    accuracies_array = np.array([[all_accuracies[j][cls] for cls in range(100)] for j in range(10)])
    mean_accuracies = np.mean(accuracies_array, axis=0)
    std_accuracies = np.std(accuracies_array, axis=0)
    threshold = mean_accuracies #- std_accuracies

    # Identify classes consistently below the threshold
    underperforming_classes = []
    for cls in range(100):
        if all(accuracies_array[:, cls] < threshold[cls]):
            underperforming_classes.append(cls)

    # Save underperforming classes to text file and pickle file
    with open("underperforming_classes.txt", "w") as f:
        f.write("Classes consistently below (mean - std) accuracy threshold over 10 runs:\n")
        for cls in underperforming_classes:
            f.write(f"Class {cls}\n")

    with open("underperforming_classes.pkl", "wb") as f:
        pickle.dump(underperforming_classes, f)

    print(f"Underperforming classes saved to 'underperforming_classes.txt' and 'underperforming_classes.pkl'.")

if __name__ == '__main__':
    main()
