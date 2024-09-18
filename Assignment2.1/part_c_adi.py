import os
import numpy as np
import pickle
import argparse
from preprocessor import *
import time

np.random.seed(0)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_data(dataset_root):
    train_csv = os.path.join(dataset_root, 'train.csv')
    val_csv = os.path.join(dataset_root, 'val.csv')

    train_dataset = CustomImageDataset(root_dir=dataset_root, csv=train_csv, transform=numpy_transform)
    val_dataset = CustomImageDataset(root_dir=dataset_root, csv=val_csv, transform=numpy_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=512)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)

    return train_loader, val_loader

def init_params():
    weights = {f"fc{i}": np.random.randn(input_size, output_size) * np.sqrt(2/input_size)
               for i, (input_size, output_size) in enumerate([(625, 512), (512, 256), (256, 128), (128, 32), (32, 8)], 1)}
    biases = {f"b{i}": np.zeros((output_size,), dtype=np.float64) for i, output_size in enumerate([512, 256, 128, 32, 8], 1)}
    return {"weights": weights, "biases": biases}

def init_optimizers(params):
    velocities = {layer: np.zeros_like(value) for layer, value in params["weights"].items()}
    momentums = {layer: np.zeros_like(value) for layer, value in params["weights"].items()}
    rms_cache = {layer: np.zeros_like(value) for layer, value in params["weights"].items()}

    # Also for biases
    momentums.update({layer: np.zeros_like(value) for layer, value in params["biases"].items()})
    rms_cache.update({layer: np.zeros_like(value) for layer, value in params["biases"].items()})

    return velocities, momentums, rms_cache

def softmax(X):
    e = np.exp(X - np.max(X, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def forward_prop(X, params):
    z1 = (X @ params["weights"]["fc1"]) + params["biases"]["b1"]
    a1 = sigmoid(z1)
    z2 = (a1 @ params["weights"]["fc2"]) + params["biases"]["b2"]
    a2 = sigmoid(z2)
    z3 = (a2 @ params["weights"]["fc3"]) + params["biases"]["b3"]
    a3 = sigmoid(z3)
    z4 = (a3 @ params["weights"]["fc4"]) + params["biases"]["b4"]
    a4 = sigmoid(z4)
    z5 = (a4 @ params["weights"]["fc5"]) + params["biases"]["b5"]
    a5 = softmax(z5)
    return z1, a1, z2, a2, z3, a3, z4, a4, z5, a5

def update_adam(grad, param, m, v, t, lr, beta1, beta2, epsilon=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return param, m, v

def back_prop(z1, a1, z2, a2, z3, a3, z4, a4, z5, a5, X, Y, params, lr, optimizer, velocities, momentums, rms_cache, t):
    m = Y.shape[0]

    output_delta = a5 - Y
    grads = {
        "w5": a4.T @ output_delta / m,
        "b5": np.mean(output_delta, axis=0),
        "w4": a3.T @ (output_delta @ params["weights"]["fc5"].T * sigmoid_derivative(z4)) / m,
        "b4": np.mean(output_delta @ params["weights"]["fc5"].T * sigmoid_derivative(z4), axis=0),
        "w3": a2.T @ (output_delta @ params["weights"]["fc5"].T * sigmoid_derivative(z4) @ params["weights"]["fc4"].T * sigmoid_derivative(z3)) / m,
        "b3": np.mean(output_delta @ params["weights"]["fc5"].T * sigmoid_derivative(z4) @ params["weights"]["fc4"].T * sigmoid_derivative(z3), axis=0),
        "w2": a1.T @ (output_delta @ params["weights"]["fc5"].T * sigmoid_derivative(z4) @ params["weights"]["fc4"].T * sigmoid_derivative(z3) @ params["weights"]["fc3"].T * sigmoid_derivative(z2)) / m,
        "b2": np.mean(output_delta @ params["weights"]["fc5"].T * sigmoid_derivative(z4) @ params["weights"]["fc4"].T * sigmoid_derivative(z3) @ params["weights"]["fc3"].T * sigmoid_derivative(z2), axis=0),
        "w1": X.T @  (output_delta @ params["weights"]["fc5"].T * sigmoid_derivative(z4) @ params["weights"]["fc4"].T * sigmoid_derivative(z3) @ params["weights"]["fc3"].T * sigmoid_derivative(z2) @ params["weights"]["fc2"].T * sigmoid_derivative(z1)) / m,
        "b1": np.mean(output_delta @ params["weights"]["fc5"].T * sigmoid_derivative(z4) @ params["weights"]["fc4"].T * sigmoid_derivative(z3) @ params["weights"]["fc3"].T * sigmoid_derivative(z2) @ params["weights"]["fc2"].T * sigmoid_derivative(z1), axis=0),
    }

    if optimizer == "adam":
        beta1, beta2 = 0.9, 0.999
        for layer in ["fc5", "fc4", "fc3", "fc2", "fc1"]:
            params["weights"][layer], momentums[layer], rms_cache[layer] = update_adam(grads[f"w{layer[-1]}"], params["weights"][layer], momentums[layer], rms_cache[layer], t, lr, beta1, beta2)
            params["biases"][f"b{layer[-1]}"], momentums[f"b{layer[-1]}"], rms_cache[f"b{layer[-1]}"] = update_adam(grads[f"b{layer[-1]}"], params["biases"][f"b{layer[-1]}"], momentums[f"b{layer[-1]}"], rms_cache[f"b{layer[-1]}"], t, lr, beta1, beta2)
    return params

def cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1.0)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def train(epochs, train_loader, valid_loader, lr, path, optimizer, max_time_minutes):
    params = init_params()
    velocities, momentums, rms_cache = init_optimizers(params)
    max_time_seconds = max_time_minutes * 60
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        for X_train, Y_train in train_loader:
            Y_train = np.eye(8)[Y_train]  # One-hot encoding
            z1, a1, z2, a2, z3, a3, z4, a4, z5, a5 = forward_prop(X_train, params)
            params = back_prop(z1, a1, z2, a2, z3, a3, z4, a4, z5, a5, X_train, Y_train, params, lr, optimizer, velocities, momentums, rms_cache, t=epoch+1)
            epoch_loss += cross_entropy_loss(Y_train, a5)

        print(f"Epoch {epoch + 1}: Loss {epoch_loss / len(train_loader):.4f}")
        
        elapsed_time = time.time() - start_time
        if elapsed_time >= max_time_seconds:
            print(f"Training stopped after {epoch + 1} epochs due to time limit")
            break

    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a neural network for classification.')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of the dataset.')
    parser.add_argument('--save_weights_path', type=str, required=True, help='Path to save the weights.')

    args = parser.parse_args()
    train_loader, valid_loader = load_data(args.dataset_root)
    train(epochs=1000, train_loader=train_loader, valid_loader=valid_loader, lr=0.001, path=args.save_weights_path, optimizer="adam", max_time_minutes=15)
