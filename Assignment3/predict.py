import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import numpy as np
import sys
import pickle
import pandas as pd
from train import WideResNet, ModelWithTemperature  # Importing the model classes from train.py

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print("Data loading completed.")
    return data

def main():
    if len(sys.argv) != 5:
        print("Usage: python predict.py model.pth test.pkl alpha gamma")
        sys.exit(1)

    model_path = sys.argv[1]
    test_file = sys.argv[2]
    alpha = float(sys.argv[3])
    gamma = float(sys.argv[4])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    print("Loading the model...")
    model = WideResNet(depth=28, num_classes=100, widen_factor=10, dropRate=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = ModelWithTemperature(model)
    model = model.to(device)
    model.eval()

    # Load and prepare test data
    print("Loading and preparing test data...")
    test_data = load_data(test_file)
    
    # Assuming test_data is a list of (image_tensor, id) tuples
    images = torch.stack([img for img, _ in test_data])
    ids = [id for _, id in test_data]
    
    # Normalize the images
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalized_images = normalize(images)
    
    test_dataset = TensorDataset(normalized_images)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Make predictions
    print("Making predictions...")
    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs[0].to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            max_probs, predicted = torch.max(probabilities, 1)
            
            # Apply confidence threshold
            predicted[max_probs < alpha] = -1
            
            predictions.extend(predicted.cpu().numpy())

    # Create submission file
    print("Creating submission file...")
    submission = pd.DataFrame({'ID': ids, 'Predicted_label': predictions})
    submission.to_csv('submission.csv', index=False)
    print("Prediction completed. Submission file 'submission.csv' created.")

if __name__ == '__main__':
    main()
