import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision.transforms as transforms
import numpy as np
import time
import sys
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print("Data loading completed.")
    return data

# Wide ResNet implementation (same as before)
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = nn.functional.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = nn.functional.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

def calculate_accuracy(model, data_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation during inference
        for inputs, labels in data_loader:
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def train_model(model, train_loader, save_weights_path):
    model = model.float()  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    
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
    
    training_duration = 5 * 60 * 60  
    start_time = time.time()

    best_val_accuracy = 0

    epoch = 0
    while epoch<125:
        epoch += 1
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.float().to(device) 
            labels = labels.long().to(device)  

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


    model.load_state_dict(torch.load(save_weights_path))
    final_val_accuracy = calculate_accuracy(model, val_loader)
    print(f"Final Validation accuracy for the model: {final_val_accuracy:.2f}%")
    
    return model, final_val_accuracy

def main():
    if len(sys.argv) != 4:
        print("Usage: python train.py train.pkl alpha gamma")
        sys.exit(1)

    train_file = sys.argv[1]
    alpha = float(sys.argv[2])
    gamma = float(sys.argv[3])
    
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WideResNet(depth=28, num_classes=100, widen_factor=10, dropRate=0.3).to(device)
    
    print("Starting model training...")
    save_weights_path = 'best_model.pth'
    model, final_val_accuracy = train_model(model, train_loader, save_weights_path)

    print(final_val_accuracy)
    
    print("Training completed. Saving the model...")
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved successfully.")

if __name__ == '__main__':
    main()


# Epoch 106, Time: 7357.09s, Loss: 0.0065, Train Accuracy: 99.94%, Val Accuracy: 64.91%
# Epoch 107, Time: 7426.20s, Loss: 0.0072, Train Accuracy: 99.89%, Val Accuracy: 64.59%
# Epoch 108, Time: 7495.29s, Loss: 0.0068, Train Accuracy: 99.93%, Val Accuracy: 64.90%
# Epoch 109, Time: 7564.38s, Loss: 0.0063, Train Accuracy: 99.95%, Val Accuracy: 64.64%
# Epoch 110, Time: 7633.48s, Loss: 0.0066, Train Accuracy: 99.94%, Val Accuracy: 64.51%
# Epoch 111, Time: 7702.57s, Loss: 0.0069, Train Accuracy: 99.92%, Val Accuracy: 64.67%
# Epoch 112, Time: 7771.68s, Loss: 0.0066, Train Accuracy: 99.95%, Val Accuracy: 64.76%
# Saved new best model with validation accuracy: 65.10%