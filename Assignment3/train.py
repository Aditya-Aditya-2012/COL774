import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import pickle
import sys
import os

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet152(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet152, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 8, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 36, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print("Data loading completed.")
    return data

class ConfidenceLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super(ConfidenceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, targets):
        ce_loss = self.ce_loss(outputs, targets)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, dim=1)
        
        correct_mask = (predicted == targets)
        confident_mask = (confidence >= self.alpha)
        
        high_accuracy_loss = ce_loss[correct_mask & confident_mask].mean()
        low_accuracy_loss = self.gamma * ce_loss[~correct_mask & confident_mask].mean()
        
        return high_accuracy_loss + low_accuracy_loss

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    print("Starting training...")
    model.train()
    for epoch in range(num_epochs):
        print(f'Starting Epoch {epoch+1}/{num_epochs}...')
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            print(f"Processing batch {batch_idx+1}/{len(train_loader)}...")
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"Batch {batch_idx+1} loss: {loss.item()}")
        print(f'Epoch {epoch+1} completed. Average Loss: {running_loss/len(train_loader)}')

def main():
    if len(sys.argv) != 4:
        print("Usage: python train.py train.pkl alpha gamma")
        sys.exit(1)

    train_file = sys.argv[1]
    alpha = float(sys.argv[2])
    gamma = float(sys.argv[3])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading and preparing data...")
    train_data = load_data(train_file)
    
    # Assuming train_data is a list of (image_tensor, label) tuples
    images = torch.stack([img for img, _ in train_data])
    labels = torch.tensor([label for _, label in train_data])
    
    # Normalize the images
    print("Normalizing the images...")
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalized_images = normalize(images)
    
    print("Creating train dataset and dataloader...")
    train_dataset = TensorDataset(normalized_images, labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print("Data preparation completed.")
    
    print("Initializing the model...")
    model = ResNet152().to(device)
    criterion = ConfidenceLoss(alpha, gamma)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting model training...")
    train_model(model, train_loader, criterion, optimizer, device)
    
    print("Training completed. Saving the model...")
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved successfully.")

if __name__ == '__main__':
    main()
