import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import numpy as np
import sys
import pickle
import pandas as pd
from train_wideresnet import WideResNet, ModelWithTemperature  # Importing the model classes from train.py

import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar


def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print("Data loading completed.")
    return data

def make_predictions(threshold, data_loader, device, model):
    threshold = float(threshold)
    predictions = [] #prediction of each exapmple
    # delta = [] #max - second highest
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs[0].to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        
            max_probs, predicted = torch.max(probabilities, 1)
            
            # Apply confidence threshold
            predicted[max_probs < threshold] = -1
            
            predictions.extend(predicted.cpu().numpy())
    return predictions


def score(threshold, train_loader, device, model, labels, alpha, gamma):
    threshold = float(threshold)
    predictions = make_predictions(threshold, train_loader, device, model)
    prediction_set = np.zeros(100)
    accuracy_for_class = np.zeros(100)
    
    print(len(predictions))
    for i in range(len(predictions)):
        if predictions[i] != -1:
            prediction_set[predictions[i]] += 1
            if predictions[i] == labels[i]:
                accuracy_for_class[predictions[i]] += 1
            
    for i in range(100):
        # print(accuracy_for_class[i])
        if prediction_set[i] > 0:
            accuracy_for_class[i] /= prediction_set[i]

    high_acc_count = 0
    low_acc_count = 0
    for i in range(100):
        # print(accuracy_for_class[i])
        if accuracy_for_class[i]>=alpha:
            high_acc_count += prediction_set[i]
        else:
            low_acc_count += prediction_set[i]

    print(high_acc_count, low_acc_count)
    return -(high_acc_count - gamma*low_acc_count)
                        

def main():
    model_path = sys.argv[1]
    test_file = sys.argv[2]
    train_file = sys.argv[3]
    alpha = float(sys.argv[4])
    gamma = float(sys.argv[5])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    print("Loading the model...")
    model = WideResNet(depth=28, num_classes=100, widen_factor=10, dropRate=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = ModelWithTemperature(model)
    model = model.to(device)
    model.eval()

    ###Train Data used to find optimum alpha
    # Load and prepare train data
    print("Loading and preparing training data...")
    train_data = load_data(train_file)

    # Assuming train_data is a list of (image_tensor, label) tuples
    images_train = torch.stack([img for img, _ in train_data])
    labels_train = torch.tensor([label for _, label in train_data])
    
    # Normalize the images
    print("Normalizing the images...")
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalized_images_train = normalize(images_train)
    
    print("Creating train dataset and dataloader...")
    train_dataset = TensorDataset(normalized_images_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    print("Data preparation completed.")
    
    # Make predictions
    # print("Finding optimum threshold on train.")
    # predictions = make_predictions(threshold, train_loader, device, model)
    # score_to_maximise = score(threshold, train_loader, device, model, labels_train, alpha, gamma)

    # threshold = 0.80
    print("running the optimiser")
    # result = minimize(score, threshold, args=(train_loader, device, model, labels_train, alpha, gamma))
    # optimal_threshold = result.x[0]

    result = minimize_scalar(
        score,
        bounds=(0.0, 1.0),
        args=(train_loader, device, model, labels_train, alpha, gamma),
        method='bounded'
    )
    optimal_threshold = result.x
    print(optimal_threshold)

    # Load and prepare test data
    print("Loading and preparing test data...")
    test_data = load_data(test_file)
    
    # Assuming test_data is a list of (image_tensor, id) tuples
    images = torch.stack([img for img, _ in test_data])
    ids = [id for _, id in test_data]
    
    # Normalize the images
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalized_images = normalize(images)
    
    test_dataset = TensorDataset(normalized_images)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # threshold = 9.7
    # Make predictions
    print("Making predictions...")
    predictions = make_predictions(optimal_threshold, test_loader, device, model)
    
    # Create submission file
    print("Creating submission file...")
    submission = pd.DataFrame({'ID': ids, 'Predicted_label': predictions})
    submission.to_csv('submission.csv', index=False)
    print("Prediction completed. Submission file 'submission.csv' created.")

if __name__ == '__main__':
    main()