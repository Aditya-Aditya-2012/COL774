import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from trainloader import CustomImageDataset, transform 
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import time

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

# Define AlexNet architecture
class AlexNet(nn.Module):
    def __init__(self, num_classes=8):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


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


class VGGNet(nn.Module):
    def __init__(self, num_classes=8):
        super(VGGNet, self).__init__()
        self.features = nn.Sequential(
            # Input: 1x50x50
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output: 64x25x25

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output: 128x12x12

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output: 256x6x6

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output: 512x3x3
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def calculate_accuracy(model, data_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation during inference
        for inputs, labels in data_loader:
            inputs = inputs.float()
            labels = labels.long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def train_model(model, train_loader, save_weights_path):
    model = model.float()  # Ensure model is using float32
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    
    # Simple validation split (80% train, 20% validation)
    dataset = train_loader.dataset
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_sampler = Subset(dataset, train_indices)
    val_sampler = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_sampler, batch_size=train_loader.batch_size, shuffle=True)
    val_loader = DataLoader(val_sampler, batch_size=train_loader.batch_size, shuffle=False)
    
    training_duration = 25 * 60  # 25 minutes in seconds
    start_time = time.time()

    best_val_accuracy = 0

    epoch = 0
    while time.time() - start_time < training_duration:
        epoch += 1
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.float()  # Ensure inputs are float32
            labels = labels.long()  # Ensure labels are long

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        
        # Evaluate on validation set
        val_accuracy = calculate_accuracy(model, val_loader)
        
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch}, Time: {elapsed_time:.2f}s, Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%")
        
        scheduler.step(val_accuracy)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_weights_path)
            print(f"Saved new best model with validation accuracy: {best_val_accuracy:.2f}%")


    # Load the best model
    model.load_state_dict(torch.load(save_weights_path))
    final_val_accuracy = calculate_accuracy(model, val_loader)
    print(f"Final Validation accuracy for the model: {final_val_accuracy:.2f}%")
    
    return model, final_val_accuracy

# Main function to train and compare models
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_root', required=True, help='Path to the train dataset root folder')
    parser.add_argument('--save_weights_path', required=True, help='Path to save the model weights')

    args = parser.parse_args()

    # Use the CustomImageDataset from trainloader.py
    dataset = CustomImageDataset(root_dir=args.train_dataset_root, csv=os.path.join(args.train_dataset_root, "public_train.csv"), transform=transform)

    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)  

    # Define the models to be trained
    models = {
        "Basic CNN": CNNMultiClassClassifier(),
        "AlexNet": AlexNet(),
        "ResNet": resnet18_custom(),
        "VGGNet": VGGNet(),
    }

    accuracy_results = {}

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        trained_model, train_accuracy = train_model(model, train_loader, args.save_weights_path)
        
        # Store the accuracy for comparison
        accuracy_results[model_name] = train_accuracy

    # Output the accuracy comparison
    print("\nAccuracy comparison:")
    print(accuracy_results)
    best_model_name = max(accuracy_results, key=accuracy_results.get)
    for model_name, accuracy in accuracy_results.items():
        print(f"{model_name}: {accuracy}%")

    # Print the best model
    print(f"\nBest model: {best_model_name} with {accuracy_results[best_model_name]}% accuracy")


# python part_c_train.py --train_dataset_root "C:\Users\HP\OneDrive - IIT Delhi\IIT Delhi\Semester 7\COL774\Assignments\A2.2\dataset_for_A2.2\dataset_for_A2.2\multi_dataset"--save_weights_path "weights.pkl"

# Training Basic CNN...
# Epoch 1, Time: 19.22s, Loss: 2.2019, Train Accuracy: 19.18%, Val Accuracy: 24.22%
# Saved new best model with validation accuracy: 24.22%
# Epoch 2, Time: 46.87s, Loss: 1.5809, Train Accuracy: 41.02%, Val Accuracy: 48.44%
# Saved new best model with validation accuracy: 48.44%
# Epoch 3, Time: 73.96s, Loss: 1.2907, Train Accuracy: 54.57%, Val Accuracy: 53.44%
# Saved new best model with validation accuracy: 53.44%
# Epoch 4, Time: 100.74s, Loss: 1.1394, Train Accuracy: 59.80%, Val Accuracy: 60.94%
# Saved new best model with validation accuracy: 60.94%
# Epoch 5, Time: 119.41s, Loss: 1.0142, Train Accuracy: 62.54%, Val Accuracy: 66.72%
# Saved new best model with validation accuracy: 66.72%
# Epoch 6, Time: 137.78s, Loss: 0.9006, Train Accuracy: 67.93%, Val Accuracy: 62.97%
# Epoch 7, Time: 156.21s, Loss: 0.7975, Train Accuracy: 71.09%, Val Accuracy: 67.81%
# Saved new best model with validation accuracy: 67.81%
# Epoch 8, Time: 174.87s, Loss: 0.6946, Train Accuracy: 75.39%, Val Accuracy: 71.56%
# Saved new best model with validation accuracy: 71.56%
# Epoch 9, Time: 193.57s, Loss: 0.6082, Train Accuracy: 78.24%, Val Accuracy: 70.16%
# Epoch 10, Time: 211.89s, Loss: 0.5452, Train Accuracy: 80.39%, Val Accuracy: 72.03%
# Saved new best model with validation accuracy: 72.03%
# Epoch 11, Time: 230.49s, Loss: 0.4820, Train Accuracy: 82.03%, Val Accuracy: 73.44%
# Saved new best model with validation accuracy: 73.44%
# Epoch 12, Time: 249.19s, Loss: 0.4447, Train Accuracy: 84.38%, Val Accuracy: 76.88%
# Saved new best model with validation accuracy: 76.88%
# Epoch 13, Time: 267.94s, Loss: 0.3556, Train Accuracy: 87.15%, Val Accuracy: 76.56%
# Epoch 14, Time: 286.65s, Loss: 0.3289, Train Accuracy: 87.66%, Val Accuracy: 72.66%
# Epoch 15, Time: 304.96s, Loss: 0.2732, Train Accuracy: 90.27%, Val Accuracy: 78.75%
# Saved new best model with validation accuracy: 78.75%
# Epoch 16, Time: 323.65s, Loss: 0.2225, Train Accuracy: 92.23%, Val Accuracy: 77.03%
# Epoch 17, Time: 341.91s, Loss: 0.1612, Train Accuracy: 95.08%, Val Accuracy: 73.59%
# Epoch 18, Time: 360.01s, Loss: 0.1688, Train Accuracy: 93.87%, Val Accuracy: 77.66%
# Epoch 19, Time: 378.10s, Loss: 0.1137, Train Accuracy: 96.29%, Val Accuracy: 76.09%
# Epoch 00019: reducing learning rate of group 0 to 1.0000e-04.
# Epoch 20, Time: 397.02s, Loss: 0.0747, Train Accuracy: 98.01%, Val Accuracy: 79.69%
# Saved new best model with validation accuracy: 79.69%
# Epoch 21, Time: 418.48s, Loss: 0.0424, Train Accuracy: 99.38%, Val Accuracy: 80.00%
# Saved new best model with validation accuracy: 80.00%
# Epoch 22, Time: 438.31s, Loss: 0.0324, Train Accuracy: 99.57%, Val Accuracy: 80.47%
# Saved new best model with validation accuracy: 80.47%
# Epoch 23, Time: 458.33s, Loss: 0.0248, Train Accuracy: 99.84%, Val Accuracy: 80.00%
# Epoch 24, Time: 477.72s, Loss: 0.0207, Train Accuracy: 99.84%, Val Accuracy: 80.31%
# Epoch 25, Time: 497.08s, Loss: 0.0168, Train Accuracy: 99.88%, Val Accuracy: 80.16%
# Epoch 26, Time: 517.18s, Loss: 0.0145, Train Accuracy: 99.92%, Val Accuracy: 80.16%
# Epoch 00026: reducing learning rate of group 0 to 1.0000e-05.
# Epoch 27, Time: 536.47s, Loss: 0.0120, Train Accuracy: 99.96%, Val Accuracy: 80.47%
# Epoch 28, Time: 554.57s, Loss: 0.0114, Train Accuracy: 100.00%, Val Accuracy: 80.16%
# Epoch 29, Time: 572.66s, Loss: 0.0113, Train Accuracy: 99.96%, Val Accuracy: 79.69%
# Epoch 30, Time: 590.73s, Loss: 0.0111, Train Accuracy: 100.00%, Val Accuracy: 80.16%
# Epoch 00030: reducing learning rate of group 0 to 1.0000e-06.
# Epoch 31, Time: 609.00s, Loss: 0.0108, Train Accuracy: 100.00%, Val Accuracy: 80.16%
# Epoch 32, Time: 627.15s, Loss: 0.0108, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 33, Time: 645.16s, Loss: 0.0108, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 34, Time: 663.16s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 00034: reducing learning rate of group 0 to 1.0000e-07.
# Epoch 35, Time: 681.38s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 36, Time: 699.55s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 37, Time: 717.79s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 38, Time: 736.00s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 00038: reducing learning rate of group 0 to 1.0000e-08.
# Epoch 39, Time: 754.20s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 40, Time: 772.42s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 41, Time: 790.51s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 42, Time: 808.84s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 43, Time: 826.95s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 44, Time: 845.10s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 45, Time: 863.12s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 46, Time: 881.28s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 47, Time: 899.56s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 48, Time: 917.64s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 49, Time: 936.01s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 50, Time: 954.14s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 51, Time: 972.11s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 52, Time: 990.29s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 53, Time: 1008.28s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 54, Time: 1026.52s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 55, Time: 1044.68s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 56, Time: 1062.68s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 57, Time: 1080.72s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 58, Time: 1098.82s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 59, Time: 1117.22s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 60, Time: 1135.27s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 61, Time: 1153.34s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 62, Time: 1171.27s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 63, Time: 1189.29s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 64, Time: 1207.44s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 65, Time: 1225.46s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 66, Time: 1243.59s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 67, Time: 1261.65s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 68, Time: 1279.68s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 69, Time: 1297.87s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 70, Time: 1315.89s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 71, Time: 1333.85s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 72, Time: 1352.13s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 73, Time: 1370.40s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 74, Time: 1388.46s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 75, Time: 1406.57s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 76, Time: 1424.73s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 77, Time: 1442.92s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 78, Time: 1461.01s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 79, Time: 1479.03s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 80, Time: 1497.08s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Epoch 81, Time: 1515.17s, Loss: 0.0107, Train Accuracy: 100.00%, Val Accuracy: 80.31%
# Final Validation accuracy for the model: 80.47%
# Training AlexNet...
# Epoch 1, Time: 55.62s, Loss: 2.0774, Train Accuracy: 12.38%, Val Accuracy: 11.88%
# Saved new best model with validation accuracy: 11.88%
# Epoch 2, Time: 112.79s, Loss: 1.9164, Train Accuracy: 23.91%, Val Accuracy: 31.72%
# Saved new best model with validation accuracy: 31.72%
# Epoch 3, Time: 172.76s, Loss: 1.7346, Train Accuracy: 34.77%, Val Accuracy: 35.16%
# Saved new best model with validation accuracy: 35.16%
# Epoch 4, Time: 232.80s, Loss: 1.4672, Train Accuracy: 43.48%, Val Accuracy: 48.28%
# Saved new best model with validation accuracy: 48.28%
# Epoch 5, Time: 288.26s, Loss: 1.3099, Train Accuracy: 50.04%, Val Accuracy: 47.34%
# Epoch 6, Time: 345.48s, Loss: 1.1848, Train Accuracy: 53.40%, Val Accuracy: 57.03%
# Saved new best model with validation accuracy: 57.03%
# Epoch 7, Time: 400.98s, Loss: 1.0247, Train Accuracy: 61.52%, Val Accuracy: 60.00%
# Saved new best model with validation accuracy: 60.00%
# Epoch 8, Time: 457.40s, Loss: 1.0147, Train Accuracy: 61.37%, Val Accuracy: 60.94%
# Saved new best model with validation accuracy: 60.94%
# Epoch 9, Time: 513.17s, Loss: 1.1549, Train Accuracy: 57.03%, Val Accuracy: 62.19%
# Saved new best model with validation accuracy: 62.19%
# Epoch 10, Time: 569.00s, Loss: 0.9683, Train Accuracy: 63.75%, Val Accuracy: 67.03%
# Saved new best model with validation accuracy: 67.03%
# Epoch 11, Time: 623.77s, Loss: 0.9200, Train Accuracy: 65.86%, Val Accuracy: 67.81%
# Saved new best model with validation accuracy: 67.81%
# Epoch 12, Time: 679.40s, Loss: 0.8929, Train Accuracy: 65.00%, Val Accuracy: 66.72%
# Epoch 13, Time: 733.81s, Loss: 0.8720, Train Accuracy: 65.90%, Val Accuracy: 67.50%
# Epoch 14, Time: 789.04s, Loss: 0.8168, Train Accuracy: 68.63%, Val Accuracy: 65.16%
# Epoch 15, Time: 844.34s, Loss: 0.7841, Train Accuracy: 70.08%, Val Accuracy: 70.94%
# Saved new best model with validation accuracy: 70.94%
# Epoch 16, Time: 903.88s, Loss: 0.7180, Train Accuracy: 72.70%, Val Accuracy: 72.34%
# Saved new best model with validation accuracy: 72.34%
# Epoch 17, Time: 976.64s, Loss: 0.7277, Train Accuracy: 72.42%, Val Accuracy: 67.97%
# Epoch 18, Time: 1046.15s, Loss: 0.6696, Train Accuracy: 74.49%, Val Accuracy: 69.22%
# Epoch 19, Time: 1116.61s, Loss: 0.6692, Train Accuracy: 74.49%, Val Accuracy: 73.59%
# Saved new best model with validation accuracy: 73.59%
# Epoch 20, Time: 1187.67s, Loss: 0.6267, Train Accuracy: 76.37%, Val Accuracy: 73.75%
# Saved new best model with validation accuracy: 73.75%
# Epoch 21, Time: 1259.00s, Loss: 0.6315, Train Accuracy: 76.13%, Val Accuracy: 66.88%
# Epoch 22, Time: 1327.78s, Loss: 0.6225, Train Accuracy: 76.88%, Val Accuracy: 74.06%
# Saved new best model with validation accuracy: 74.06%
# Epoch 23, Time: 1398.84s, Loss: 0.5604, Train Accuracy: 79.61%, Val Accuracy: 73.44%
# Epoch 24, Time: 1470.19s, Loss: 0.5093, Train Accuracy: 81.05%, Val Accuracy: 78.28%
# Saved new best model with validation accuracy: 78.28%
# Epoch 25, Time: 1540.20s, Loss: 0.5594, Train Accuracy: 78.12%, Val Accuracy: 77.34%
# Final Validation accuracy for the model: 78.28%
# Training ResNet...
# Epoch 1, Time: 28.69s, Loss: 1.2956, Train Accuracy: 51.37%, Val Accuracy: 13.59%
# Saved new best model with validation accuracy: 13.59%
# Epoch 2, Time: 57.09s, Loss: 0.7750, Train Accuracy: 71.02%, Val Accuracy: 61.88%
# Saved new best model with validation accuracy: 61.88%
# Epoch 3, Time: 85.43s, Loss: 0.6048, Train Accuracy: 77.77%, Val Accuracy: 59.22%
# Epoch 4, Time: 113.82s, Loss: 0.4660, Train Accuracy: 83.01%, Val Accuracy: 61.88%
# Epoch 5, Time: 142.61s, Loss: 0.3936, Train Accuracy: 85.16%, Val Accuracy: 40.31%
# Epoch 6, Time: 169.81s, Loss: 0.3329, Train Accuracy: 87.89%, Val Accuracy: 67.19%
# Saved new best model with validation accuracy: 67.19%
# Epoch 7, Time: 198.33s, Loss: 0.2359, Train Accuracy: 91.56%, Val Accuracy: 52.66%
# Epoch 8, Time: 225.63s, Loss: 0.2288, Train Accuracy: 92.19%, Val Accuracy: 32.34%
# Epoch 9, Time: 253.21s, Loss: 0.1863, Train Accuracy: 93.36%, Val Accuracy: 57.50%
# Epoch 10, Time: 277.38s, Loss: 0.1594, Train Accuracy: 94.10%, Val Accuracy: 62.81%
# Epoch 00010: reducing learning rate of group 0 to 1.0000e-04.
# Epoch 11, Time: 301.08s, Loss: 0.0749, Train Accuracy: 97.62%, Val Accuracy: 87.50%
# Saved new best model with validation accuracy: 87.50%
# Epoch 12, Time: 325.02s, Loss: 0.0391, Train Accuracy: 99.10%, Val Accuracy: 87.19%
# Epoch 13, Time: 348.68s, Loss: 0.0254, Train Accuracy: 99.53%, Val Accuracy: 85.94%
# Epoch 14, Time: 372.39s, Loss: 0.0204, Train Accuracy: 99.88%, Val Accuracy: 90.78%
# Saved new best model with validation accuracy: 90.78%
# Epoch 15, Time: 396.47s, Loss: 0.0137, Train Accuracy: 100.00%, Val Accuracy: 88.91%
# Epoch 16, Time: 420.14s, Loss: 0.0118, Train Accuracy: 100.00%, Val Accuracy: 91.72%
# Saved new best model with validation accuracy: 91.72%
# Epoch 17, Time: 444.30s, Loss: 0.0096, Train Accuracy: 100.00%, Val Accuracy: 91.09%
# Epoch 18, Time: 467.90s, Loss: 0.0086, Train Accuracy: 100.00%, Val Accuracy: 89.69%
# Epoch 19, Time: 491.41s, Loss: 0.0091, Train Accuracy: 99.96%, Val Accuracy: 91.56%
# Epoch 20, Time: 515.83s, Loss: 0.0067, Train Accuracy: 100.00%, Val Accuracy: 90.78%
# Epoch 00020: reducing learning rate of group 0 to 1.0000e-05.
# Epoch 21, Time: 539.77s, Loss: 0.0057, Train Accuracy: 100.00%, Val Accuracy: 92.03%
# Saved new best model with validation accuracy: 92.03%
# Epoch 22, Time: 567.90s, Loss: 0.0053, Train Accuracy: 100.00%, Val Accuracy: 91.72%
# Epoch 23, Time: 597.01s, Loss: 0.0054, Train Accuracy: 100.00%, Val Accuracy: 92.19%
# Saved new best model with validation accuracy: 92.19%
# Epoch 24, Time: 625.49s, Loss: 0.0061, Train Accuracy: 100.00%, Val Accuracy: 92.50%
# Saved new best model with validation accuracy: 92.50%
# Epoch 25, Time: 653.80s, Loss: 0.0057, Train Accuracy: 100.00%, Val Accuracy: 92.34%
# Epoch 26, Time: 681.82s, Loss: 0.0052, Train Accuracy: 100.00%, Val Accuracy: 92.34%
# Epoch 27, Time: 707.89s, Loss: 0.0052, Train Accuracy: 100.00%, Val Accuracy: 92.34%
# Epoch 28, Time: 735.77s, Loss: 0.0049, Train Accuracy: 100.00%, Val Accuracy: 92.34%
# Epoch 00028: reducing learning rate of group 0 to 1.0000e-06.
# Epoch 29, Time: 763.80s, Loss: 0.0048, Train Accuracy: 100.00%, Val Accuracy: 92.34%
# Epoch 30, Time: 791.97s, Loss: 0.0046, Train Accuracy: 100.00%, Val Accuracy: 92.34%
# Epoch 31, Time: 816.84s, Loss: 0.0048, Train Accuracy: 100.00%, Val Accuracy: 92.34%
# Epoch 32, Time: 844.94s, Loss: 0.0051, Train Accuracy: 100.00%, Val Accuracy: 92.34%
# Epoch 00032: reducing learning rate of group 0 to 1.0000e-07.
# Epoch 33, Time: 924.56s, Loss: 0.0048, Train Accuracy: 100.00%, Val Accuracy: 92.34%
# Epoch 34, Time: 1010.40s, Loss: 0.0054, Train Accuracy: 100.00%, Val Accuracy: 92.34%
# Epoch 35, Time: 1094.42s, Loss: 0.0053, Train Accuracy: 100.00%, Val Accuracy: 92.34%
# Epoch 36, Time: 1129.22s, Loss: 0.0044, Train Accuracy: 100.00%, Val Accuracy: 92.19%
# Epoch 00036: reducing learning rate of group 0 to 1.0000e-08.
# Epoch 37, Time: 1168.81s, Loss: 0.0052, Train Accuracy: 100.00%, Val Accuracy: 92.50%
# Epoch 38, Time: 1254.10s, Loss: 0.0044, Train Accuracy: 100.00%, Val Accuracy: 92.19%
# Epoch 39, Time: 1342.02s, Loss: 0.0049, Train Accuracy: 100.00%, Val Accuracy: 92.34%
# Epoch 40, Time: 1430.60s, Loss: 0.0056, Train Accuracy: 100.00%, Val Accuracy: 92.34%
# Epoch 41, Time: 1497.88s, Loss: 0.0044, Train Accuracy: 100.00%, Val Accuracy: 92.19%
# Epoch 42, Time: 1587.88s, Loss: 0.0041, Train Accuracy: 100.00%, Val Accuracy: 92.34%
# Final Validation accuracy for the model: 92.50%
# Training VGGNet...
# Epoch 1, Time: 326.62s, Loss: 1.8382, Train Accuracy: 31.09%, Val Accuracy: 11.25%
# Saved new best model with validation accuracy: 11.25%
# Epoch 2, Time: 653.69s, Loss: 1.2460, Train Accuracy: 49.61%, Val Accuracy: 42.97%
# Saved new best model with validation accuracy: 42.97%
# Epoch 3, Time: 983.13s, Loss: 1.0182, Train Accuracy: 58.40%, Val Accuracy: 54.06%
# Saved new best model with validation accuracy: 54.06%
# Epoch 4, Time: 1302.02s, Loss: 0.8502, Train Accuracy: 67.66%, Val Accuracy: 70.00%
# Saved new best model with validation accuracy: 70.00%
# Epoch 5, Time: 1412.47s, Loss: 0.7641, Train Accuracy: 71.25%, Val Accuracy: 63.59%
# Epoch 6, Time: 1521.69s, Loss: 0.6825, Train Accuracy: 74.96%, Val Accuracy: 67.19%
# Final Validation accuracy for the model: 70.00%

# Accuracy comparison:
# {'Basic CNN': 80.46875, 'AlexNet': 78.28125, 'ResNet': 92.5, 'VGGNet': 70.0}
# Basic CNN: 80.46875%
# AlexNet: 78.28125%
# ResNet: 92.5%
# VGGNet: 70.0%

# Best model: ResNet with 92.5% accuracy
