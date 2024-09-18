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

    train_loader = DataLoader(dataset=train_dataset, batch_size=256)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)

    return train_loader, val_loader

def init_params() :
    weights = {
            "fc1": np.random.randn(625, 512) * np.sqrt(2/(625)),
            "fc2": np.random.randn(512, 256) * np.sqrt(2/(512)),
            "fc3": np.random.randn(256, 128) * np.sqrt(2/(256)),
            "fc4": np.random.randn(128, 64) * np.sqrt(2/(128)),
            "fc5": np.random.randn(64, 32) * np.sqrt(2/64),
            "fc6": np.random.randn(32, 8) * np.sqrt(2/(32))
        }
    bias = {
            "b1": np.zeros((512,), dtype=np.float64),
            "b2": np.zeros((256,), dtype=np.float64),
            "b3": np.zeros((128,), dtype=np.float64),
            "b4": np.zeros((64,), dtype=np.float64),
            "b5": np.zeros((32,), dtype=np.float64)
            "b6": np.zeros((8,), dtype=np.float64)
        }
    
    params = {
            "weights" : weights,
            "bias" : bias
            }
    
    return params

def init_optimizers(params):
   
    velocities = {}
    momentums = {}
    rms_cache = {}

    # Initialize velocities, momentums, and rms_cache for each fully connected (fc) layer
    for layer in params["weights"]:
        velocities[layer] = np.zeros_like(params["weights"][layer])  # Momentum storage
        momentums[layer] = np.zeros_like(params["weights"][layer])   # Adam first moment
        rms_cache[layer] = np.zeros_like(params["weights"][layer])   # RMSProp/Adam second moment

    return velocities, momentums, rms_cache


def softmax(X):
    m=np.max(X,axis=1).reshape(-1,1)        
    e=np.exp(X-m)
    s=np.sum(e,axis=1).reshape(-1,1)
    return e/s

def sigmoid(z) :
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return 1 / (2 + np.exp(-z) + 1/np.exp(-z))

def forward_prop(X, params) :
    z1 = (X @ params["weights"]["fc1"]) + params["bias"]["b1"]    
    a1 = sigmoid(z1)

    z2 = (a1 @ params["weights"]["fc2"]) + params["bias"]["b2"]
    a2 = sigmoid(z2)

    z3 = (a2 @ params["weights"]["fc3"]) + params["bias"]["b3"]
    a3 = sigmoid(z3)

    z4 = (a3 @ params["weights"]["fc4"]) + params["bias"]["b4"]
    a4 = sigmoid(z4)

    z5 = (a4 @ params["weights"]["fc5"]) + params["bias"]["b5"]
    a5 = softmax(z5)

    z6 = (a5 @ params["weights"]["fc6"]) + params["bias"]["b6"]
    a6 = softmax(z6)

    out_dict = {
        "z1": z1
        "z2": z2
        "z3": z3
        "z4": z4
        "z5": z5
        "z6": z6
        "a1": a1
        "a2": a2
        "a3": a3
        "a4": a4
        "a5": a5
        "a6": a6}

    return out_dict, a6

def back_prop(out_dict, a6, X, Y, params, lr, optimizer, velocities, momentums, rms_cache, t) :
    m = Y.shape[0]  # Number of samples
    
    output_delta = a6 - Y

    # Gradients for weights and biases
    d_w6 = (out_dict["a5"].T @ output_delta) / m
    d_b6 = np.mean(output_delta, axis=0)

    he_5 = output_delta @ params["weights"]["fc6"].T
    hd_5 = he_5 * sigmoid_derivative(out_dict["z5"]) 

    d_w5 = (a4.T @ hd_5) / m
    d_b5 = np.mean(hd_5, axis=0)

    he_4 = hd_4 @ params["weights"]["fc5"].T
    hd_4 = he_4 * sigmoid_derivative(z4)

    d_w4 = (a3.T @ hd_4) / m
    d_b4 = np.mean(hd_4, axis=0) 

    he_3 = hd_4 @ params["weights"]["fc4"].T
    hd_3 = he_3 * sigmoid_derivative(z3)

    d_w3 = (a2.T @ hd_3) / m
    d_b3 = np.mean(hd_3, axis=0) 

    he_2 = hd_3 @ params["weights"]["fc3"].T
    hd_2 = he_2 * sigmoid_derivative(z2)

    d_w2 = (a1.T @ hd_2) / m
    d_b2 = np.mean(hd_2, axis=0) 

    he_1 = hd_2 @ params["weights"]["fc2"].T
    hd_1 = he_1 * sigmoid_derivative(z1)

    d_w1 = (X.T @ hd_1) / m
    d_b1 = np.mean(hd_1, axis=0) 

    beta1 = 0.6  # Decay factor for the first moment
    beta2 = 0.95 # Decay factor for the second moment
    epsilon = 1e-8

    #######################################################################################
    # Update momentums and rms_cache for each layer
    momentums["fc5"] = beta1 * momentums["fc5"] + (1 - beta1) * d_w5
    rms_cache["fc5"] = beta2 * rms_cache["fc5"] + (1 - beta2) * d_w5 ** 2
    m_hat_5 = momentums["fc5"] / (1 - beta1 ** t)
    v_hat_5 = rms_cache["fc5"] / (1 - beta2 ** t)
    params["weights"]["fc5"] -= lr * m_hat_5 / (np.sqrt(v_hat_5) + epsilon)
    params["bias"]["b5"] -= lr * d_b5

    momentums["fc4"] = beta1 * momentums["fc4"] + (1 - beta1) * d_w4
    rms_cache["fc4"] = beta2 * rms_cache["fc4"] + (1 - beta2) * d_w4 ** 2
    m_hat_4 = momentums["fc4"] / (1 - beta1 ** t)
    v_hat_4 = rms_cache["fc4"] / (1 - beta2 ** t)
    params["weights"]["fc4"] -= lr * m_hat_4 / (np.sqrt(v_hat_4) + epsilon)
    params["bias"]["b4"] -= lr * d_b4 

    momentums["fc3"] = beta1 * momentums["fc3"] + (1 - beta1) * d_w3
    rms_cache["fc3"] = beta2 * rms_cache["fc3"] + (1 - beta2) * d_w3 ** 2
    m_hat_3 = momentums["fc3"] / (1 - beta1 ** t)
    v_hat_3 = rms_cache["fc3"] / (1 - beta2 ** t)
    params["weights"]["fc3"] -= lr * m_hat_3 / (np.sqrt(v_hat_3) + epsilon)
    params["bias"]["b3"] -= lr * d_b3 

    momentums["fc2"] = beta1 * momentums["fc2"] + (1 - beta1) * d_w2
    rms_cache["fc2"] = beta2 * rms_cache["fc2"] + (1 - beta2) * d_w2 ** 2
    m_hat_2 = momentums["fc2"] / (1 - beta1 ** t)
    v_hat_2 = rms_cache["fc2"] / (1 - beta2 ** t)
    params["weights"]["fc2"] -= lr * m_hat_2 / (np.sqrt(v_hat_2) + epsilon)
    params["bias"]["b2"] -= lr * d_b2 

    momentums["fc1"] = beta1 * momentums["fc1"] + (1 - beta1) * d_w1
    rms_cache["fc1"] = beta2 * rms_cache["fc1"] + (1 - beta2) * d_w1 ** 2
    m_hat_1 = momentums["fc1"] / (1 - beta1 ** t)
    v_hat_1 = rms_cache["fc1"] / (1 - beta2 ** t)
    params["weights"]["fc1"] -= lr * m_hat_1 / (np.sqrt(v_hat_1) + epsilon)
    params["bias"]["b1"] -= lr * d_b1 
    ################################################################################
    
    return params


def cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1.0)
    
    n_samples = y_true.shape[0]
    cross_entropy = -np.sum(y_true * np.log(y_pred)) / n_samples
    return cross_entropy

def save_weights(params, path, i):
    with open(f'{path}_{i}.pkl', 'wb') as f:
        pickle.dump(params, f)

def one_hot(Y) :
    n_classes = np.max(Y) + 1  
    Y_one_hot = np.eye(n_classes)[Y]
    return Y_one_hot

def train(epochs, train_loader, valid_loader, lr, path, optimizer):
    params = init_params()
    velocities, momentums, rms_cache = init_optimizers(params)
    save_weights(params, path, 0)

    for i in range(epochs):
        epoch_loss = 0.
        for X_train, Y_train in train_loader:
            Y_train = one_hot(Y_train)
            z1, a1, z2, a2, z3, a3, z4, a4, z5, a5 = forward_prop(X_train, params)
            params = back_prop(z1, a1, z2, a2, z3, a3, z4, a4, z5, a5, X_train, Y_train, params, lr, optimizer, velocities, momentums, rms_cache, t=i+1)
            epoch_loss += cross_entropy_loss(Y_train, a5)
        
        print(f'epoch: {i} loss: {epoch_loss}')
        # save_weights(params, path, i+1)

    valid_loss = 0.
    for X_val, Y_val in valid_loader:
        z1, a1, z2, a2, z3, a3, z4, a4, z5, a5 = forward_prop(X_val, params)
        valid_loss = cross_entropy_loss(Y_val, a5)
    print(f'validation loss: {valid_loss}')

    # Save predictions on the training set
    all_predictions = []
    for X_train, _ in train_loader:
        _, _, _, _, _, _, _, _, _, a5 = forward_prop(X_train, params)
        predictions = np.argmax(a5, axis=1)  # Assuming your output is one-hot encoded
        all_predictions.extend(predictions)
    
    all_predictions = np.array(all_predictions)
    
    # Save predictions to a .pkl file
    with open('predictions.pkl', 'wb') as f:
        pickle.dump(all_predictions, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a neural network for binary classification.')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of the dataset.')
    parser.add_argument('--save_weights_path', type=str, required=True, help='Path to save the weights.')

    args = parser.parse_args()

    train_loader, valid_loader = load_data(args.dataset_root)
    start_time = time.time()
    
    params = train(epochs = 15, train_loader=train_loader, valid_loader=valid_loader, lr = 0.001, path=args.save_weights_path, optimizer="adam")
    
    end_time = time.time()
    running_time = end_time - start_time
    print(f'Training completed in {running_time:.2f} seconds')
    
