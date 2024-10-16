import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import numpy as np
import sys
import pickle
import pandas as pd
from datasets import CIFAR100Dataset, compute_mean_std
from models.pyramidnet import ShakePyramidNet
from train import WideResNet
torch.manual_seed(0)

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print("Data loading completed.")
    return data

def tta_predict(model, inputs, num_augmentations=5):
    predictions = []
    for _ in range(num_augmentations):
        augmented_inputs = transforms.RandomHorizontalFlip()(inputs)
        outputs = model(augmented_inputs)
        predictions.append(torch.nn.functional.softmax(outputs, dim=1))
    return torch.stack(predictions).mean(dim=0)

def main():
    if len(sys.argv) != 5:
        print("Usage: python predict.py model.pth test.pkl alpha gamma")
        sys.exit(1)

    model_path = sys.argv[1]
    test_file = sys.argv[2]
    alpha = float(sys.argv[3])
    gamma = float(sys.argv[4])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_test = CIFAR100Dataset(data_path=test_file)
    mean, std = compute_mean_std(raw_test)
    # Load the model
    print("Loading the model...")
    # model = WideResNet(depth=28, num_classes=100, widen_factor=10, dropRate=0.3)
    model = ShakePyramidNet(depth=110, alpha=270, label=100)
    model.load_state_dict(torch.load(model_path, map_location=device))
    # model = ModelWithTemperature(model)
    model = model.to(device)
    model.eval()

    # Load and prepare test data
    print("Loading and preparing test data...")
    test_data = load_data(test_file)
    
    # Assuming test_data is a list of (image_tensor, id) tuples
    images = torch.stack([img for img, _ in test_data])
    ids = [id for _, id in test_data]
    
    # Normalize the images
    normalize = transforms.Normalize(mean=mean, std=std)
    normalized_images = normalize(images)
    
    test_dataset = TensorDataset(normalized_images)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Make predictions
    print("Making predictions...")
    predictions = []
    confidences = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs[0].to(device)
            probs = tta_predict(model, inputs)
            
            max_probs, predicted = torch.max(probs, 1)
            predictions.extend(predicted.cpu().numpy())
            confidences.extend(max_probs.cpu().numpy())

    # Dynamic threshold calculation
    sorted_confidences = sorted(confidences, reverse=True)
    confidence_threshold = sorted_confidences[int(len(sorted_confidences) * 0.7263)]  

    # Apply confidence threshold and create submission
    print("Creating submission file...")
    final_predictions = []
    for pred, conf in zip(predictions, confidences):
        if conf >= confidence_threshold:
            final_predictions.append(pred)
        else:
            final_predictions.append(-1)

    submission = pd.DataFrame({'ID': ids, 'Predicted_label': final_predictions})
    submission.to_csv('submission.csv', index=False)
    print("Prediction completed. Submission file 'submission.csv' created.")

if __name__ == '__main__':
    main()
