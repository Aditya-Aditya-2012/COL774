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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

torch.manual_seed(0)

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print("Data loading completed.")
    return data

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion Matrix'):
    plt.figure(figsize=(20, 18))  # Increased figure size for better readability
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")

    sns.heatmap(cm, annot=False, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=20)
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('True', fontsize=16)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)  # Higher resolution
    plt.close()

def plot_class_wise_accuracy(class_accuracies, class_names=None):
    # Sort classes by accuracy
    sorted_classes = sorted(class_accuracies.items(), key=lambda item: item[1], reverse=True)
    classes, accuracies = zip(*sorted_classes)

    if class_names:
        classes_labels = [class_names[cls] for cls in classes]
    else:
        classes_labels = [str(cls) for cls in classes]

    plt.figure(figsize=(20, 10))  # Increased figure size
    bars = plt.barh(range(len(classes)), accuracies, color='skyblue')
    plt.xlabel('Accuracy (%)', fontsize=16)
    plt.title('Class-Wise Accuracy', fontsize=20)
    plt.yticks(range(len(classes)), classes_labels, fontsize=8)
    plt.gca().invert_yaxis()  # Highest accuracy at the top

    # Annotate bars with accuracy values
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{acc:.2f}%', va='center', fontsize=6)

    plt.tight_layout()
    plt.savefig('class_accuracy_val.png', dpi=300)  # Higher resolution
    plt.close()

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
    train_loader, val_loader = load_dataset(256, path_train=train_file, seed=0)

    correct_predictions = defaultdict(int)
    total_predictions = defaultdict(int)
    all_targets = []
    all_preds = []

    # If CIFAR-100, get class names
    cifar100 = CIFAR100Dataset()  # Assuming this class provides class names
    class_names = cifar100.classes if hasattr(cifar100, 'classes') else None

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

            # For confusion matrix
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Calculate accuracy for each class
    class_accuracies = {cls: (correct_predictions[cls] / total_predictions[cls]) * 100 
                        for cls in range(100)}

    # Compute mean and standard deviation of the accuracies
    accuracies = np.array(list(class_accuracies.values()))
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    threshold = mean_accuracy - 2 * std_accuracy

    # Identify classes with accuracy <= mean - 2*std
    underperforming_classes = [cls for cls, acc in class_accuracies.items() if acc <= threshold]

    # Save underperforming classes to a text file
    with open("underperforming_classes.txt", "w") as f:
        f.write("Classes with accuracy <= mean - 2*std:\n")
        for cls in underperforming_classes:
            class_label = class_names[cls] if class_names else cls
            f.write(f"Class {cls} ({class_label}): {class_accuracies[cls]:.2f}%\n")

    # Save the underperforming classes to a .pkl file
    with open("underperforming_classes.pkl", "wb") as f:
        pickle.dump(underperforming_classes, f)

    # Sort classes by accuracy
    sorted_accuracies = sorted(class_accuracies.items(), key=lambda item: item[1])

    # Save sorted accuracies to a text file
    with open("sorted_class_accuracies.txt", "w") as f:
        f.write("Classes sorted by accuracy (ascending):\n")
        for cls, acc in sorted_accuracies:
            class_label = class_names[cls] if class_names else cls
            f.write(f"Class {cls} ({class_label}): {acc:.2f}%\n")

    print(f"Class-wise accuracy calculation completed.")
    print(f"Results saved to 'underperforming_classes.txt', 'underperforming_classes.pkl', and 'sorted_class_accuracies.txt'.")

    # Plot class-wise accuracy
    plot_class_wise_accuracy(class_accuracies, class_names=class_names)

    # Plot confusion matrix
    print("Plotting confusion matrix...")
    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm, classes=class_names if class_names else range(100))

if __name__ == '__main__':
    main()
