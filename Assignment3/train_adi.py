import os
import argparse
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
torch.manual_seed(0)
# from datasets import load_dataset
from models.pyramidnet import ShakePyramidNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print("Data loading completed.")
    return data


# Helper function to calculate accuracy
def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def train_model(model, train_loader, save_weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.float().to(device)
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    optimizer = optim.SGD(model.parameters(),
                    lr=0.1,
                    momentum=0.9,
                    nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [300 // 2, 300 * 3 // 4])
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    
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
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training loop
    training_duration = 4.5 * 60 * 60
    start_time = time.time()
    best_val_accuracy = 0
    epoch = 0

    while time.time() - start_time < training_duration and epoch<300:
        epoch += 1
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for inputs, labels in train_loader:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())

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
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)

        # Evaluate on validation set
        val_accuracy = calculate_accuracy(model, val_loader)
        val_loss = 0.0
        for inputs, labels in val_loader:
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch}, Time: {elapsed_time:.2f}s, Train Loss: {epoch_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        scheduler.step(val_accuracy)

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_weights_path)
            print(f"Saved new best model with validation accuracy: {best_val_accuracy:.2f}%")

        epochs_range = range(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 6))

        # Plot accuracy
        plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
        plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.savefig('loss_pyramid_final_plateau.png')

    model.load_state_dict(torch.load(save_weights_path))
    final_val_accuracy = calculate_accuracy(model, val_loader)

    print(f"Final Validation accuracy for the calibrated model: {final_val_accuracy:.2f}%")
    # Plotting training and validation curves

    return model, best_val_accuracy


def main():
    if len(sys.argv) != 4:
        print("Usage: python train.py train.pkl alpha gamma")
        sys.exit(1)

    train_file = sys.argv[1]
    alpha = float(sys.argv[2])
    gamma = float(sys.argv[3])

    normalize = transforms.Normalize(mean=[0.5069, 0.4865, 0.4406], std=[0.2671, 0.2563, 0.2760])
    # transform_train = transforms.Compose([
    #     # transforms.RandomCrop(32, padding=4),
    #     # transforms.RandomHorizontalFlip(),  
    #     normalize
    # ])

    train_data = load_data(train_file)
    
    # Assuming train_data is a list of (image_tensor, label) tuples
    images = torch.stack([img for img, _ in train_data])
    labels = torch.tensor([label for _, label in train_data])

    normalized_images = normalize(images)
    
    print("Creating train dataset and dataloader...")
    train_dataset = TensorDataset(normalized_images, labels)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # train_loader, test_loader = load_dataset(batch_size=64, path_train=train_file, path_test=train_file)
    model = ShakePyramidNet(depth=110, alpha=270, label=100)
    # model = torch.nn.DataParallel(model).cuda()

    print("Starting model training...")
    save_weights_path = 'best_model_pyramid_plateau.pth'
    model, final_val_accuracy = train_model(model, train_loader, save_weights_path)

    # print("Training completed. Saving the model...")
    # torch.save(model.state_dict(), 'model.pth')
    print("Model saved successfully.")


if __name__ == "__main__":
    main()
