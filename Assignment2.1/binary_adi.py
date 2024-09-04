import os
import numpy as np
import pickle
import argparse
from preprocessor import *

np.random.seed(0)

def load_data(dataset_root):
    train_csv = os.path.join(dataset_root, 'train.csv')
    val_csv = os.path.join(dataset_root, 'val.csv')

    train_dataset = CustomImageDataset(root_dir=dataset_root, csv=train_csv, transform=numpy_transform)
    val_dataset = CustomImageDataset(root_dir=dataset_root, csv=val_csv, transform=numpy_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=256)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)

    return train_loader, val_loader

def init_params() :
    weights = {
            "fc1": np.random.randn(625, 512) * np.sqrt(2.0 / 625),
            "fc2": np.random.randn(512, 256) * np.sqrt(2.0 / 512),
            "fc3": np.random.randn(256, 128) * np.sqrt(2.0 / 256),
            "fc4": np.random.randn(128, 1) * np.sqrt(2.0 / 128)
        }
    bias = {
            "b1": np.zeros((512,), dtype=np.float64),
            "b2": np.zeros((256,), dtype=np.float64),
            "b3": np.zeros((128,), dtype=np.float64),
            "b4": np.zeros((1,), dtype=np.float64)
        }
    
    params = {
            "weights" : weights,
            "bias" : bias
            }
    
    return params

def sigmoid(z) :
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
        return z * (1 - z)

def forward_prop(X, params) :
    z1 = X @ params["weights"]["fc1"] + params["bias"]["b1"]
    a1 = sigmoid(z1)

    z2 = z1 @ params["weights"]["fc2"] + params["bias"]["b2"]
    a2 = sigmoid(z2)

    z3 = z2 @ params["weights"]["fc3"] + params["bias"]["b3"]
    a3 = sigmoid(z3)

    z4 = z3 @ params["weights"]["fc4"] + params["bias"]["b4"]
    a4 = sigmoid(z4)

    return z1, a1, z2, a2, z3, a3, z4, a4

def back_prop(z1, a1, z2, a2, z3, a3, z4, a4, Y, X, params, lr) :
    m = Y.shape[0] 

    output_error = Y - a4
    output_delta = output_error @ sigmoid_derivative(a4)

    he_3 = output_delta @ params["weights"]["fc4"].T
    hd_3 = he_3 * sigmoid_derivative(a3)

    he_2 = hd_3 @ params["weights"]["fc3"].T
    hd_2 = he_2 * sigmoid_derivative(a2)

    he_1 = hd_2 @ params["weights"]["fc2"].T
    hd_1 = he_1 * sigmoid_derivative(a1)

    params["weights"]["fc4"] -= lr * a3.T @ output_delta / m
    params["bias"]["b4"] -= lr * np.sum(output_delta, axis=0, keepdims=False).T / m

    params["weights"]["fc3"] -= lr * a2.T @ hd_3 / m
    params["bias"]["b3"] -= lr * np.sum(hd_3, axis=0, keepdims=False).T / m

    params["weights"]["fc2"] -= lr * a1.T @ hd_2 / m
    params["bias"]["b2"] -= lr * np.sum(hd_2, axis=0, keepdims=False).T / m

    params["weights"]["fc1"] -= lr * X.T @ hd_1 / m
    params["bias"]["b1"] -= lr * np.sum(hd_1, axis=0, keepdims=False).T / m

    return params

def compute_loss(y_true, y_pred):
        m = y_true.shape[0]
        loss = -(1/m) * np.sum(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
        return loss

def save_weights(params, path, i):
    with open(f'{path}_{i}.pkl', 'wb') as f:
        pickle.dump(params, f)

def train(epochs, train_loader, valid_loader, lr, path) :
    params = init_params()
    for i in range(epochs) :
        epoch_loss = 0.
        for X_train, Y_train in train_loader :
            z1, a1, z2, a2, z3, a3, z4, a4 = forward_prop(X_train, params)
            params = back_prop(z1, a1, z2, a2, z3, a3, z4, a4, Y_train, X_train, params, lr)
            epoch_loss += compute_loss(Y_train, a4)
        
        print(f'epoch : {i} loss : {epoch_loss}')
        save_weights(params, path, i+1)

    valid_loss = 0.
    for X_val, Y_val in valid_loader :
        z1, a1, z2, a2, z3, a3, z4, a4 = forward_prop(X_val, params)
        valid_loss = compute_loss(Y_val, a4)
    
    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a neural network for binary classification.')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of the dataset.')
    parser.add_argument('--save_weights_path', type=str, required=True, help='Path to save the weights.')

    args = parser.parse_args()

    train_loader, valid_loader = load_data(args.dataset_root)
    params = train(epochs = 5, train_loader=train_loader, valid_loader=valid_loader, lr = 0.001, path=args.save_weights_path)


    









