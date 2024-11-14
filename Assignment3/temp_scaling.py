import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import pickle
import pandas as pd
from datasets import CIFAR100Dataset, compute_mean_std, get_transforms, load_dataset
from models.pyramidnet import ShakePyramidNet
from models.smooth_ce import smooth_crossentropy
from utility.plateauLR import ReduceLROnPlateau
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.initialize import initialize
import logging

# Initialize seed and device
initialize(seed=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def load_data(file_path):
    logging.info(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    logging.info("Data loading completed.")
    return data

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Initialize temperature

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

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

def calibrate_model(model, val_loader, device):
    """
    Calibrate the model by optimizing the temperature parameter to minimize NLL on validation set.
    """
    model.eval()
    logits = []
    labels = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.float().to(device)
            targets = targets.long().to(device)
            outputs = model(inputs)
            logits.append(outputs)
            labels.append(targets)
    logits = torch.cat(logits)
    labels = torch.cat(labels)
    
    # Initialize temperature scaling model
    temperature_model = ModelWithTemperature(model)
    temperature_model.to(device)
    
    # Define optimizer for temperature parameter
    optimizer = torch.optim.LBFGS([temperature_model.temperature], lr=0.01, max_iter=50)
    
    # Define loss function (Negative Log Likelihood)
    loss_function = nn.CrossEntropyLoss()
    
    def eval_closure():
        optimizer.zero_grad()
        loss = loss_function(logits / temperature_model.temperature, labels)
        loss.backward()
        return loss
    
    # Optimize temperature
    optimizer.step(eval_closure)
    
    logging.info(f"Optimal temperature: {temperature_model.temperature.item():.4f}")
    return temperature_model

def get_model(model_name):
    if model_name == 'pyramidnet':
        model = ShakePyramidNet(depth=110, alpha=270, label=100).float().to(device)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    return model

def main():
    setup_logging()
    
    if len(sys.argv) != 4:
        logging.error("Usage: python calibrate.py model.pth val.pkl calibrated_model.pth")
        sys.exit(1)

    model_path = sys.argv[1]
    val_file = sys.argv[2]
    calibrated_model_path = sys.argv[3]

    logging.info(f"Using device: {device}")
    
    # Load validation data
    logging.info("Loading validation data...")
    train_loader, val_loader = load_dataset(256, path_train=val_file, seed=0)
    # Compute mean and std from validation set for normalization
    
    
    # Initialize the model
    logging.info("Loading the model...")
    model = get_model('pyramidnet')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Calibrate the model
    logging.info("Starting calibration...")
    temperature_model = calibrate_model(model, val_loader, device)
    
    # Save the calibrated model
    logging.info(f"Saving calibrated model to {calibrated_model_path}...")
    torch.save(temperature_model.state_dict(), calibrated_model_path)
    logging.info("Calibration completed and model saved.")

if __name__ == '__main__':
    main()
