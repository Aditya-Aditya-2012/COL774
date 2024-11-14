import sys
import torch
import numpy as np
import pickle
import pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from datasets import CIFAR100Dataset, compute_mean_std, get_transforms, load_dataset
from models.pyramidnet import ShakePyramidNet
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

torch.manual_seed(0)

def calculate_classwise_accuracy(df, pred_col):
    """Calculate accuracy for each class."""
    accuracy_dict = {}
    grouped = df.groupby(pred_col)

    for name, group in grouped:
        accuracy = (group['Label'] == group[pred_col]).sum() / len(group)
        accuracy_dict[name] = accuracy

    return accuracy_dict

def score(actual_labels, predicted_labels, accuracy_threshold=0.9, gamma=5.0) -> float:
    """
    Custom metric to evaluate model performance.
    
    Parameters:
        - actual_labels: Array of true class labels.
        - predicted_labels: Array of predicted class labels.
        - accuracy_threshold: Threshold for class accuracy.
        - gamma: Weighting factor for low accuracy classifications.
    
    Returns:
        - A single float representing the overall performance of the model.
    """
    # Create a DataFrame to hold actual and predicted labels
    df = pd.DataFrame({
        'Label': actual_labels,
        'Predicted_label': predicted_labels
    })

    # Calculate classwise accuracy
    accuracy_per_class = calculate_classwise_accuracy(df, 'Predicted_label')

    all_classes = list(range(100))
    sum_of_correctly_classified_high_accuracy = 0
    sum_of_correctly_classified_low_accuracy = 0

    for cls in all_classes:
        total = len(df[df['Predicted_label'] == cls])
        correct = (df[df['Predicted_label'] == cls]['Predicted_label'] == df[df['Predicted_label'] == cls]['Label']).sum()
        class_accuracy = accuracy_per_class.get(cls, 0.0)
        
        if class_accuracy >= accuracy_threshold:
            sum_of_correctly_classified_high_accuracy += total
        else:
            sum_of_correctly_classified_low_accuracy += total

    # Calculate final score
    final_score = sum_of_correctly_classified_high_accuracy - gamma * sum_of_correctly_classified_low_accuracy

    return float(final_score)

def tta_predict(model, inputs, num_augmentations=5):
    predictions = []
    for _ in range(num_augmentations):
        augmented_inputs = transforms.RandomHorizontalFlip()(inputs)
        outputs = model(augmented_inputs)
        predictions.append(torch.nn.functional.softmax(outputs, dim=1))
    return torch.stack(predictions).mean(dim=0)

def main():
    if len(sys.argv) != 5:
        print("Usage: python predict_val.py model.pth test.pkl alpha gamma underperforming_classes.pkl")
        sys.exit(1)

    model_path = sys.argv[1]
    train_file = sys.argv[2]
    alpha = float(sys.argv[3])
    gamma = float(sys.argv[4])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    print("Loading the model...")
    model = ShakePyramidNet(depth=110, alpha=270, label=100)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load and prepare training data
    train_loader, val_loader = load_dataset(256, path_train=train_file, seed=0)
    class_numbers = [58]

    all_targets = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            probs = tta_predict(model, inputs)
            max_probs, predicted = torch.max(probs, 1)
            
            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    final_predictions = []
    cnt68 = 0
    for i in range(len(all_preds)):
        first_prob, _ = torch.max(probs, 1)  # First probability
        second_prob = probs[i].topk(2)[0][1].item()  # Second probability
        
        if all_preds[i] == 68:
            cnt68 += 1
            final_predictions.append(all_preds[i])
        else:
            final_predictions.append(-1)

        # Print the probabilities for correct and wrong predictions
        print(f"First Probability: {first_prob}, Second Probability: {second_prob}, Correct: {all_preds[i] == all_targets[i]}")


    final_score = score(np.array(all_targets), np.array(final_predictions), accuracy_threshold=alpha, gamma=gamma)
    print(f"Final Score: {final_score}")
    print(f'class 68 coutn: {cnt68}')

if __name__ == '__main__':
    main()
