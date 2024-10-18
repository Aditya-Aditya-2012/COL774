import sys
MAINPATH = ".."  # nopep8
sys.path.append(MAINPATH)  # nopep8

import torch
import numpy as np
import pickle
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from datasets import CIFAR100Dataset, compute_mean_std, get_transforms, load_dataset
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
    # raw_train = CIFAR100Dataset(data_path=train_file)
    # mean, std = compute_mean_std(raw_train)

    # print("Loading and preparing training data...")
    # train_data = load_data(train_file)
    
    # # Assuming train_data is a list of (image_tensor, label) tuples
    # images = torch.stack([img for img, _ in train_data])
    # labels = torch.tensor([label for _, label in train_data])

    # # Normalize the images
    # normalize = transforms.Normalize(mean=mean, std=std)
    # normalized_images = normalize(images)
    
    # train_dataset = TensorDataset(normalized_images, labels)
    # train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    train_loader, val_loader = load_dataset(256, path_train=train_file, seed=0)
    # Initialize class-wise counters
    correct_predictions = defaultdict(int)
    total_predictions = defaultdict(int)

    # Evaluate class-wise accuracy
    print("Evaluating class-wise accuracy...")
    with torch.no_grad():
        for inputs, targets in val_loader:
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

    # Compute mean and standard deviation of the accuracies
    accuracies = np.array(list(class_accuracies.values()))
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    threshold = mean_accuracy - 2*std_accuracy

    # Identify classes with accuracy <= mean - std
    underperforming_classes = [cls for cls, acc in class_accuracies.items() if acc <= threshold]

    # Save underperforming classes to a text file
    with open("underperforming_classes.txt", "w") as f:
        f.write("Classes with accuracy <= mean - 2*std:\n")
        for cls in underperforming_classes:
            f.write(f"Class {cls}: {class_accuracies[cls]:.2f}%\n")

    # Save the underperforming classes to a .pkl file
    with open("underperforming_classes.pkl", "wb") as f:
        pickle.dump(underperforming_classes, f)

    # Sort classes by accuracy
    sorted_accuracies = sorted(class_accuracies.items(), key=lambda item: item[1])

    # Save sorted accuracies to a text file
    with open("sorted_class_accuracies.txt", "w") as f:
        f.write("Classes sorted by accuracy (ascending):\n")
        for cls, acc in sorted_accuracies:
            f.write(f"Class {cls}: {acc:.2f}%\n")

    print(f"Class-wise accuracy calculation completed.")
    print(f"Results saved to 'underperforming_classes_val.txt', 'underperforming_classes_val.pkl', and 'sorted_class_accuracies_val.txt'.")

if __name__ == '__main__':
    main()
