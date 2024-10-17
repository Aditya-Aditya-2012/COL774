import sys
MAINPATH = ".."  # nopep8
sys.path.append(MAINPATH)  # nopep8
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Import from parent directory
from datasets import CIFAR100Dataset, compute_mean_std, get_transforms, load_dataset

# Import from sibling directories
from models.pyramidnet import ShakePyramidNet
from utility.initialize import initialize

initialize(seed=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_confusion_matrix(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

train_file = '/home/civil/btech/ce1210494/A3_data/train.pkl'
model_path = '/home/civil/btech/ce1210494/a3models/model_pyr_SAM_newstepLR.pth'

model = ShakePyramidNet(depth=110, alpha=270, label=100)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
batch_size=256
train_loader, val_loader = load_dataset(batch_size, path_train=train_file)
plot_confusion_matrix(model, val_loader)

