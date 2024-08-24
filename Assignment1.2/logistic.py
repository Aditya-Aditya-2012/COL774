import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.sparse import csr_matrix
import math
import sys

def preprocess(train):
    y_train = np.array(train['Race'])
    train = train.drop(columns=['Race'])

    X_train = train.to_numpy()
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]  
    
    y_encod = pd.get_dummies(y_train).to_numpy()
    
    return X_train, y_encod

# Computes the modified log-likelihood loss for weighted logistic regression.
def compute_loss(X, y, W, freq):
    n = X.shape[0]
    loss = 0
    for i in range(n):
        sumj = np.sum(np.exp(X[i].dot(W)))
        for j in range(W.shape[1]):
            if y[i, j] == 1:
                loss += (X[i].dot(W[:, j]) - max_val - np.log(sumj)) / freq[j]
    return -loss / (2 * n)

# Computes the gradient of the loss with respect to the weights.
def compute_gradient(X, y, W, freq):
    n, m = X.shape
    k = W.shape[1]
    gradient = np.zeros(W.shape)

    for i in range(n):
        exp_values = np.exp(X[i].dot(W))
        sumj = np.sum(exp_values)
        for j in range(k):
            indicator = y[i, j]
            probs = exp_values[j] / sumj
            gradient[:, j] += (indicator - probs) * X[i] / freq[j]
    
    return -gradient / (2 * n)

# Performs exact line search using ternary search to find the optimal learning rate.
def ternary_search(X, y, W, freq, g, eta0):
    eta_l = 0
    eta_h = eta0

    while compute_loss(X, y, W - eta_h * g, freq) >= compute_loss(X, y, W, freq):
        eta_h *= 2

    for _ in range(20):
        eta1 = (2 * eta_l + eta_h) / 3
        eta2 = (eta_l + 2 * eta_h) / 3

        loss1 = compute_loss(X, y, W - eta1 * g, freq)
        loss2 = compute_loss(X, y, W - eta2 * g, freq)

        if loss1 > loss2:
            eta_l = eta1
        elif loss1 < loss2:
            eta_h = eta2
        else:
            eta_l = eta1
            eta_h = eta2

    return (eta_l + eta_h) / 2


# Executes mini-batch gradient descent with exact line search to update weights.
def mini_batch_gradient_descent(X, y, W, freq, batch_size, epochs, eta0, train_strat, k):
    m, n = X.shape
    for epoch in range(epochs):
        for i in range(0, m, batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            
            g = compute_gradient(X_batch, y_batch, W, freq)
            if train_strat == 1:
                W -= eta0 * g
            elif train_strat == 2:
                eta = eta0 / (1 + k * i)
                W -= eta * g
            else:
                eta = ternary_search(X_batch, y_batch, W, freq, g, eta0)
                W -= eta * g
    return W

def main(task, train_file, params_file, output_file):
    train_data = pd.read_csv(train_file)

    X_train, y_train = preprocess(train_data)

    if task == 'a':
        train_strat = 1
        k = 0
        eta0 = 0
        epochs = 0
        batch_size = 0
        
        with open(params_file, 'r') as f:
            params = f.read().splitlines()
            train_strat = int(params[0])
            if train_strat == 1:
                eta0 = float(params[1])
            elif train_strat == 2:
                eta0, k = map(float, params[1].strip().split(','))
            else:
                eta0 = float(params[1])
            epochs = int(params[2])
            batch_size = int(params[3])

        m = X_train.shape[1]
        n = y_train.shape[1]
        W = np.zeros((m, n), dtype=np.float64)
        
        freq = np.array([np.sum(y_train[:, j]) for j in range(n)], dtype=np.float64)
        W = mini_batch_gradient_descent(X_train, y_train, W, freq, batch_size, epochs, eta0, train_strat, k)
        np.savetxt(output_file, W)

if __name__ == "__main__":
    task = sys.argv[1]
    train_file = sys.argv[2]
    params_file = sys.argv[3]
    output_file = sys.argv[4]
    main(task, train_file, params_file, output_file)
