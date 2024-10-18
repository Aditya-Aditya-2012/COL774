import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from make_plot import make_plots
from datasets import CIFAR100Dataset, compute_mean_std, get_transforms, load_dataset
from models.pyramidnet import ShakePyramidNet
from models.smooth_ce import smooth_crossentropy
from models.sam import SAM
from utility.plateauLR import ReduceLROnPlateau
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.initialize import initialize

initialize(seed=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def calculate_loss(model, data_loader, loss_function):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():  # Disable gradient computation during inference
        for inputs, labels in data_loader:
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            
            # Get model predictions
            outputs = model(inputs)
            
            # Compute loss for the batch
            loss = loss_function(outputs, labels)
            
            # Accumulate loss and number of samples
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

    # Calculate average loss over all samples
    average_loss = total_loss / total_samples
    return average_loss


def train_model(model, train_loader, val_loader, save_weights_path, epoch_cap, plot_name):
    model = model.float().to(device)
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, adaptive=True, momentum=0.9, lr=0.05, weight_decay=0.0005)
    # scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.2, min_lr=1e-10)
    scheduler = StepLR(optimizer, 0.05, epoch_cap)

    train_accuracies = []
    val_accuracies = []

    training_duration = 5.5 * 60 * 60  
    start_time = time.time()

    best_val_accuracy = 0

    epoch = 0
    while time.time() - start_time < training_duration and epoch < epoch_cap:
        epoch += 1
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.float().to(device) 
            labels = labels.long().to(device) 

            # First forward-backward step
            enable_running_stats(model)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, labels)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True) 

            # Second forward-backward step
            disable_running_stats(model)
            smooth_crossentropy(model(inputs), labels).mean().backward()
            optimizer.second_step(zero_grad=True)

            running_loss += loss.mean().item()
            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_loss = calculate_loss(model, val_loader, smooth_crossentropy)

        with torch.no_grad():
            scheduler(epoch)
            # scheduler(val_loss)

        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)
        
        # Evaluate on validation set
        val_accuracy = calculate_accuracy(model, val_loader)
        val_accuracies.append(val_accuracy)
        make_plots(train_accuracies, val_accuracies, plot_name)
        
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch}, Time: {elapsed_time:.2f}s, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_weights_path)
            print(f"Saved new best model with validation accuracy: {best_val_accuracy:.2f}%")

    return model, best_val_accuracy

def get_model(model_name):
    if model_name == 'pyramidnet':
        model = ShakePyramidNet(depth=110, alpha=270, label=100).float().to(device)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    return model

def main():
    if len(sys.argv) != 4:
        print("Usage: python train.py train.pkl alpha gamma")
        sys.exit(1)

    train_file = sys.argv[1]
    alpha = float(sys.argv[2])
    gamma = float(sys.argv[3])

    batch_size = 256  # Adjust batch size as needed

    print("Loading and preparing data with data augmentation...")
    train_loader, val_loader = load_dataset(batch_size, path_train=train_file, seed=0)

    print("Initializing the model...")
    model = get_model('pyramidnet') 

    print("Starting model training...")
    save_weights_path = 'model_18_10.pth'
    model, final_val_accuracy = train_model(model, train_loader, val_loader, save_weights_path, epoch_cap=300, plot_name='pyramidnet_step')

    print(f"Final Validation Accuracy: {final_val_accuracy:.2f}%")

if __name__ == '__main__':
    main()
