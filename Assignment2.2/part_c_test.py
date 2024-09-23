import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from testloader import CustomImageDataset, transform 
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
import pickle

torch.manual_seed(0)

# Define the basic CNN architecture
class CNNMultiClassClassifier(nn.Module):
    def __init__(self):
        super(CNNMultiClassClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)

        self.fc1 = nn.Linear(128 * 11 * 24, 512) 
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, 8)  # Output layer for 8 classes

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=8):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet18_custom(num_classes=8):
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes)

def test_model(model, test_loader, load_weights_path, save_predictions_path):
    # Create model and load weights
    model.load_state_dict(torch.load(load_weights_path))
    model.eval()  # Set the model to evaluation mode

    all_predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.float()
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()  # Get the class with the highest probability
            all_predictions.extend(preds)

    # Save predictions as a pickle file
    with open(save_predictions_path, 'wb') as f:
        pickle.dump(np.array(all_predictions), f)

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
    test_loader = DataLoader(dataset, batch_size=128, shuffle=False)

    # Call the testing function
    test_model(resnet18_custom(), test_loader, args.load_weights_path, args.save_predictions_path)