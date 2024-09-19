import os
import argparse
import torch
import torch.nn as nn
import pickle
import numpy as np
from testloader import CustomImageDataset, transform  # Import CustomImageDataset and transform from your testloader.py
from torch.utils.data import DataLoader

torch.manual_seed(0)

# Define the CNN architecture (same as in training script)
class CNNBinaryClassifier(nn.Module):
    def __init__(self):
        super(CNNBinaryClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 12 * 25, 1)  # Adjust based on your input size

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def test_model(test_loader, load_weights_path, save_predictions_path):
    # Create model and load weights
    model = CNNBinaryClassifier()
    model.load_state_dict(torch.load(load_weights_path))
    model.eval()  # Set the model to evaluation mode

    predictions = []
    
    with torch.no_grad():
        for inputs in test_loader:  # Only one value expected in the batch
            inputs = inputs.float()
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).squeeze().numpy()
            predictions.extend(preds)
    
    # Convert predictions to binary class labels (0 or 1)
    binary_predictions = [1 if pred > 0.5 else 0 for pred in predictions]

    # Save predictions as a pickle file
    with open(save_predictions_path, 'wb') as f:
        pickle.dump(np.array(binary_predictions), f)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset_root', required=True, help='Path to the test dataset root folder')
    parser.add_argument('--load_weights_path', required=True, help='Path to load the model weights')
    parser.add_argument('--save_predictions_path', required=True, help='Path to save the predictions file')

    args = parser.parse_args()

    # Use the CustomImageDataset from testloader.py
    dataset = CustomImageDataset(root_dir=args.test_dataset_root, csv=os.path.join(args.test_dataset_root, "public_test.csv"), transform=transform)

    # Create DataLoader
    test_loader = DataLoader(dataset, batch_size=128)

    # Call the testing function
    test_model(test_loader, args.load_weights_path, args.save_predictions_path)
