import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
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

torch.manual_seed(0)

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def load_data(file_path):
    logging.info(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    logging.info("Data loading completed.")
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

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

def main():
    setup_logging()
    
    if len(sys.argv) != 6:
        logging.error("Usage: python predict.py model.pth val.pkl test.pkl calibrated_model.pth")
        sys.exit(1)

    model_path = sys.argv[1]
    test_file = sys.argv[2]
    alpha = float(sys.argv[3])
    gamma = float(sys.argv[4])
    calibrated_model_path = sys.argv[5]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load validation data for calibration
    logging.info("Loading calibration data...")
    
    # Load the model
    logging.info("Loading the model...")
    model = ShakePyramidNet(depth=110, alpha=270, label=100)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = ModelWithTemperature(model)
    model.to(device)
    model.eval()
    
    # Load calibrated temperature
    logging.info("Loading calibrated temperature...")
    calibrated_model = torch.load(calibrated_model_path, map_location=device)
    model.temperature.data = calibrated_model['temperature']
    
    # Load and prepare test data
    logging.info("Loading and preparing test data...")
    test_data = load_data(test_file)
    
    images = torch.stack([img for img, _ in test_data])
    ids = [id for _, id in test_data]
    
    raw_test = CIFAR100Dataset(data_path=test_file)
    mean, std = compute_mean_std(raw_test)

    normalize = transforms.Normalize(mean=mean, std=std)
    normalized_images = normalize(images)
    test_dataset = TensorDataset(normalized_images)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # TTA transforms

    # Make predictions
    logging.info("Making predictions...")
    predictions = []
    confidences = []
    std_deviations = []
    ZScore = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            inputs = batch[0].to(device)
            probabilities, std_prediction = tta_predict(model, inputs)
            
            # Calculate statistics
            std_dev = torch.std(probabilities, dim=1)
            mean_prob_across_classes = torch.mean(probabilities, dim=1)
            max_probs, predicted = torch.max(probabilities, 1)

            zsc = (max_probs - mean_prob_across_classes) / std_dev

            # Collect results
            std_deviations.extend(std_dev.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            confidences.extend(max_probs.cpu().numpy())
            ZScore.extend(zsc.cpu().numpy())
            
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(test_loader):
                logging.info(f"Processed {batch_idx + 1}/{len(test_loader)} batches.")

    # Convert ZScore to numpy array for efficient processing
    ZScore_np = np.array(ZScore)
    
    # Determine the number of samples
    num_samples = len(ZScore_np)
    logging.info(f"Total number of samples: {num_samples}")
    
    # Define top_k
    top_k = 900
    if num_samples < top_k:
        logging.warning(f"Number of samples ({num_samples}) is less than top_k ({top_k}). Selecting all samples.")
        top_indices = np.arange(num_samples)
    else:
        # Get indices of top_k ZScores
        top_indices = np.argsort(ZScore_np)[-top_k:]
    
    logging.info(f"Selecting top {top_k} predictions based on ZScore.")
    
    # Initialize final_predictions with -1
    final_predictions = np.full(len(predictions), -1, dtype=int)
    
    # Assign predictions to top_indices
    final_predictions[top_indices] = np.array(predictions)[top_indices]
    
    # Optional: Save the top_k ZScores to a separate file
    # top_zscores = ZScore_np[top_indices]
    # with open('top_zscores.txt', 'w') as file:
    #     for z in top_zscores:
    #         file.write(f"{z}\n")
    
    # Save the results to a CSV file
    logging.info("Creating submission file...")
    submission = pd.DataFrame({'ID': ids, 'Predicted_label': final_predictions})
    submission.to_csv('submission10.csv', index=False)
    logging.info("Prediction completed. Submission file 'submission.csv' created.")

if __name__ == '__main__':
    main()