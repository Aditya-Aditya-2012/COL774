import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision.transforms as transforms
import numpy as np
import time
import sys
import pickle
import matplotlib.pyplot as plt
from models.wideresnet import WideResNet
from models.pyramidnet import ShakePyramidNet
from models.smooth_ce import smooth_crossentropy
from models.sam import SAM
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.initialize import initialize

initialize(seed=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_val_split(train_data, split_ratio=0.2):
    dataset = train_data.dataset
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(split_ratio * dataset_size))
    
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_sampler = Subset(dataset, train_indices)
    val_sampler = Subset(dataset, val_indices)
    
    train_set = DataLoader(train_sampler, batch_size=train_data.batch_size, shuffle=True)
    val_set = DataLoader(val_sampler, batch_size=train_data.batch_size, shuffle=False)

    return train_set, val_set


def make_plots(train_loss, val_loss, plt_name):
    epochs_range = range(1, len(train_loss) + 1)
    plt.figure(figsize=(10, 6))

    # Plot accuracy
    plt.plot(epochs_range, train_loss, label='Training Accuracy')
    plt.plot(epochs_range, val_loss, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(plt_name+'.png')



def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print("Data loading completed.")
    return data

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

def train_model(model, train_loader, save_weights_path, epoch_cap):
    model = model.float().to(device)
    criterion = nn.CrossEntropyLoss().cuda()
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, adaptive=True, momentum=0.9, lr=0.05, weight_decay=0.0005)
    scheduler = StepLR(optimizer, 0.05, epoch_cap)

    # optimizer = optim.SGD(model.parameters(),
    #                 lr=0.1,
    #                 momentum=0.9,
    #                 nesterov=True)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [300 // 2, 300 * 3 // 4])
    
    # Simple validation split (80% train, 20% validation)
    train_loader, val_loader = train_val_split(train_loader, split_ratio=0.2)
    
    train_accuracies = []
    val_accuracies = []

    training_duration = 4.5 * 60 * 60  
    start_time = time.time()

    best_val_accuracy = 0

    epoch = 0
    while time.time() - start_time < training_duration and epoch<epoch_cap:
        epoch += 1
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.float().to(device) 
            labels = labels.long().to(device) 

            #first forward-backward step
            enable_running_stats(model)
            predictions=model(inputs)
            loss=smooth_crossentropy(predictions, labels)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True) 

            #second forward-backward step
            disable_running_stats(model)
            smooth_crossentropy(model(inputs), labels).mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                scheduler(epoch)

            running_loss += loss.mean().item()
            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)
        
        # Evaluate on validation set
        val_accuracy = calculate_accuracy(model, val_loader)
        val_accuracies.append(val_accuracy)
        make_plots(train_accuracies, val_accuracies, 'pyramidnet')
        
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch}, Time: {elapsed_time:.2f}s, Train Loss: {epoch_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%")
        
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_weights_path)
            print(f"Saved new best model with validation accuracy: {best_val_accuracy:.2f}%")

    model.load_state_dict(torch.load(save_weights_path))
    final_val_accuracy = calculate_accuracy(model, val_loader)
    print(f"Final Validation accuracy for the model: {final_val_accuracy:.2f}%")
    
    return model, final_val_accuracy

def get_model(model_name):
    if model_name == 'wideresnet':
        model = WideResNet(depth=28, num_classes=100, widen_factor=10, dropRate=0.3).float().to(device)
    if model_name == 'pyramidnet':
        model = ShakePyramidNet(depth=110, alpha=270, label=100).float().to(device)
    return model

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
    normalize = transforms.Normalize(mean=[0.5069, 0.4865, 0.4406], std=[0.2671, 0.2563, 0.2760])
    normalized_images = normalize(images)
    
    print("Creating train dataset and dataloader...")
    train_dataset = TensorDataset(normalized_images, labels)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    print("Initializing the model...")
    model = get_model('pyramidnet')
    # model = get_model('wideresnet')
    
    print("Starting model training...")
    save_weights_path = 'model.pth'
    model, final_val_accuracy = train_model(model, train_loader, save_weights_path, epoch_cap=300)

    print(final_val_accuracy)

if __name__ == '__main__':
    main()