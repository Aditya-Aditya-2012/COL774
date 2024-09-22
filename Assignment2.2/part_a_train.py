import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from trainloader import CustomImageDataset, transform  
from torch.utils.data import DataLoader

torch.manual_seed(0)

# Define the CNN architecture
class CNNBinaryClassifier(nn.Module):
    def __init__(self):
        super(CNNBinaryClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 12 * 25, 1)  

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def train_model(train_loader, save_weights_path):
    # Create model, define optimizer and loss function
    model = CNNBinaryClassifier().float()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the CNN model
    num_epochs = 8
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.float()
            labels = labels.float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Save model state after each epoch
        torch.save(model.state_dict(), save_weights_path)
        # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_root', required=True, help='Path to the train dataset root folder')
    parser.add_argument('--save_weights_path', required=True, help='Path to save the model weights')

    args = parser.parse_args()

    # Use the CustomImageDataset from trainloader.py
    dataset = CustomImageDataset(root_dir=args.train_dataset_root, csv=os.path.join(args.train_dataset_root, "public_train.csv"), transform=transform)

    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=128, shuffle=False)

    # Call the training function
    train_model(train_loader, args.save_weights_path)


# Epoch 1/8, Loss: 0.7932, Test Accuracy: 50.00%
# Epoch 2/8, Loss: 0.6900, Test Accuracy: 50.00%
# Epoch 3/8, Loss: 0.6763, Test Accuracy: 72.00%
# Epoch 4/8, Loss: 0.6385, Test Accuracy: 79.50%
# Epoch 5/8, Loss: 0.5711, Test Accuracy: 79.00%
# Epoch 6/8, Loss: 0.4872, Test Accuracy: 77.50%
# Epoch 7/8, Loss: 0.4220, Test Accuracy: 79.00%
# Epoch 8/8, Loss: 0.3831, Test Accuracy: 80.00%

