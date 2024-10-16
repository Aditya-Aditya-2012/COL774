import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import numpy as np
import sys
import pickle
import pandas as pd
import torch.nn.functional as F
from datasets import CIFAR100Dataset, compute_mean_std, get_transforms
from models.pyramidnet import ShakePyramidNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, inputs):
        logits = self.model(inputs)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        # Temperature scaling
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, valid_loader):
        nll_criterion = nn.CrossEntropyLoss().to(device)
        ece_criterion = _ECELoss().to(device)

        logits_list = []
        labels_list = []

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)

            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)

        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        print(f'Optimal temperature: {self.temperature.item()}')

        # Calculate ECE after temperature scaling
        ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print(f'ECE after temperature scaling: {ece}')

        return self

class _ECELoss(nn.Module):
    """Calculates Expected Calibration Error (ECE) of a model."""
    def __init__(self, n_bins=15):
        super().__init__()
        self.n_bins = n_bins

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=logits.device)

        for i in range(self.n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]
            mask = (confidences > lower) & (confidences <= upper)
            prop = mask.float().mean()
            if prop.item() > 0:
                accuracy = accuracies[mask].float().mean()
                avg_confidence = confidences[mask].mean()
                ece += torch.abs(avg_confidence - accuracy) * prop

        return ece

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print("Data loading completed.")
    return data

def tta_predict(model, inputs, num_augmentations=10):
    predictions = []
    tta_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    for _ in range(num_augmentations):
        augmented_inputs = torch.stack([tta_transform(img.cpu()) for img in inputs])
        augmented_inputs = augmented_inputs.to(device)
        outputs = model(augmented_inputs)
        predictions.append(F.softmax(outputs, dim=1))
    return torch.stack(predictions).mean(dim=0)

def find_optimal_threshold(model, val_loader):
    model.eval()
    all_confidences = []
    all_correct = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, 1)
            correct = predictions.eq(labels).float()

            all_confidences.extend(confidences.cpu().numpy())
            all_correct.extend(correct.cpu().numpy())

    # Find the threshold that maximizes F1 score
    thresholds = np.linspace(0.5, 1.0, 100)
    best_threshold = 0.0
    best_f1 = -1

    for threshold in thresholds:
        mask = np.array(all_confidences) >= threshold
        if np.sum(mask) == 0:
            continue
        selected_correct = np.array(all_correct)[mask]
        precision = selected_correct.mean() if selected_correct.size > 0 else 0
        recall = np.sum(mask) / len(all_correct)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f'Optimal confidence threshold: {best_threshold}')
    return best_threshold

def main():
    if len(sys.argv) != 5:
        print("Usage: python predict.py model.pth test.pkl alpha gamma")
        sys.exit(1)

    model_path = sys.argv[1]
    test_file = sys.argv[2]
    alpha = float(sys.argv[3])
    gamma = float(sys.argv[4])

    batch_size = 256  # Same as in training

    # Load and prepare test data
    print("Loading and preparing test data...")
    test_data = load_data(test_file)
    images = torch.stack([img for img, _ in test_data])
    ids = [id for _, id in test_data]

    # Compute mean and std from training data
    print("Computing mean and std from training data...")
    train_file = 'train.pkl'  # Replace with your actual training data file path
    raw_train = CIFAR100Dataset(data_path=train_file)
    mean, std = compute_mean_std(raw_train)

    # Normalize the images
    print("Normalizing test images...")
    normalize = transforms.Normalize(mean=mean, std=std)
    normalized_images = torch.stack([normalize(img) for img in images])

    test_dataset = TensorDataset(normalized_images)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Load the model
    print("Loading the model...")
    model = ShakePyramidNet(depth=110, alpha=270, label=100)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Prepare validation loader using the same method as in training
    print("Loading and preparing validation data...")
    transform_train, transform_val = get_transforms(mean, std)

    # Load full dataset without transforms
    full_dataset = CIFAR100Dataset(data_path=train_file, transform=None)
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))  # 80-20 split
    np.random.seed(0)  # Ensure reproducibility
    np.random.shuffle(indices)
    val_indices = indices[:split]

    # Create validation dataset
    val_data = [full_dataset[i] for i in val_indices]
    val_dataset = CIFAR100Dataset(data_list=val_data, transform=transform_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Calibrate the model using validation data
    print("Calibrating model with temperature scaling...")
    model = ModelWithTemperature(model)
    model.set_temperature(val_loader)

    # Find optimal threshold
    print("Finding optimal confidence threshold...")
    confidence_threshold = find_optimal_threshold(model, val_loader)

    # Make predictions
    print("Making predictions...")
    predictions = []
    confidences = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs[0].to(device)
            probs = tta_predict(model, inputs, num_augmentations=10)
            max_probs, predicted = torch.max(probs, 1)
            predictions.extend(predicted.cpu().numpy())
            confidences.extend(max_probs.cpu().numpy())

    # Apply confidence threshold and create submission
    print("Creating submission file...")
    final_predictions = []
    for pred, conf in zip(predictions, confidences):
        if conf >= confidence_threshold:
            final_predictions.append(pred)
        else:
            final_predictions.append(-1)  # Abstain

    submission = pd.DataFrame({'ID': ids, 'Predicted_label': final_predictions})
    submission.to_csv('submission.csv', index=False)
    print("Prediction completed. Submission file 'submission.csv' created.")

if __name__ == '__main__':
    main()
