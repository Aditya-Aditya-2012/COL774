import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import numpy as np
import sys
import pickle
import pandas as pd
from datasets import CIFAR100Dataset, compute_mean_std
from models.pyramidnet import ShakePyramidNet
torch.manual_seed(0)

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print("Data loading completed.")
    return data

def tta_predict(model, inputs, num_augmentations=5):
    predictions = []
    for _ in range(num_augmentations):
        augmented_inputs = transforms.RandomHorizontalFlip()(inputs)
        outputs = model(augmented_inputs)
        # outputs = model(inputs)
        predictions.append(torch.nn.functional.softmax(outputs, dim=1))
    
    stacked_predictions = torch.stack(predictions)
    mean_prediction = stacked_predictions.mean(dim=0)
    std_prediction = stacked_predictions.std(dim=0)  # Calculate standard deviation

    return mean_prediction, std_prediction


def main():
    if len(sys.argv) != 5:
        print("Usage: python predict.py model.pth test.pkl alpha gamma ")
        sys.exit(1)

    model_path = sys.argv[1]
    test_file = sys.argv[2]
    alpha = float(sys.argv[3])
    gamma = float(sys.argv[4])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    raw_test = CIFAR100Dataset(data_path=test_file)
    mean, std = compute_mean_std(raw_test)
    # Load the model
    print("Loading the model...")
    # model = WideResNet(depth=28, num_classes=100, widen_factor=10, dropRate=0.3)
    model = ShakePyramidNet(depth=110, alpha=270, label=100)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load and prepare test data
    print("Loading and preparing test data...")
    test_data = load_data(test_file)
    
    # Assuming test_data is a list of (image_tensor, id) tuples
    images = torch.stack([img for img, _ in test_data])
    ids = [id for _, id in test_data]
    
    # Normalize the images
    normalize = transforms.Normalize(mean=mean, std=std)
    normalized_images = normalize(images)
    
    test_dataset = TensorDataset(normalized_images)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Make predictions
    print("Making predictions...")
    predictions = []
    confidences = []
    std_deviations = []
    ZScore = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs[0].to(device)
            probabilities, std_prediction = tta_predict(model, inputs)
            
            std_dev = torch.std(probabilities, dim=1)
            mean_prob_across_classes = torch.mean(probabilities, dim=1)
            max_probs, predicted = torch.max(probabilities, 1)

            zsc = (max_probs - mean_prob_across_classes)/std_dev

            std_deviations.extend(std_dev.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            confidences.extend(max_probs.cpu().numpy())
            ZScore.extend(zsc.cpu().numpy())

    # print(f"Length of S_dev = {len(std_deviations)}") # 20000
    # print(f"Max of S_dev = {max(std_deviations)} Min of S_dev = {min(std_deviations)}") # 0.0995953157544136 ; 0.01871304400265217
    # print(f"Length of ZScore = {len(ZScore)}") # 20000
    # print(f"Max of ZScore = {max(ZScore)} Min of ZScore = {min(ZScore)}") # 9.899999618530273 ; 4.328963279724121

    # Apply confidence threshold and underperforming classes filter
    print("Creating submission file...")
    final_predictions = []
    for i in range(len(std_deviations)):
        if ZScore[i] >= 9.899965:
            final_predictions.append(predictions[i])
        else:
            final_predictions.append(-1)

    # Save the results to a CSV file
    submission = pd.DataFrame({'ID': ids, 'Predicted_label': final_predictions})
    submission.to_csv('submission.csv', index=False)
    print("Prediction completed. Submission file 'submission.csv' created.")

if __name__ == '__main__':
    main()