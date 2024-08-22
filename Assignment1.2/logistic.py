import numpy as np
import pandas as pd
import sys

# Computes the softmax probabilities for each class given the logits.
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Computes the modified log-likelihood loss for weighted logistic regression.
def compute_loss(X, y, W, freq):
    m = X.shape[0]
    logits = X @ W
    probs = softmax(logits)
    loss = 0
    for i in range(m):
        for j in range(W.shape[1]):
            if y[i] == j + 1:
                loss += np.log(probs[i, j]) / freq[j]
    return -loss / (2 * m)

# Computes the gradient of the loss with respect to the weights.
def compute_gradient(X, y, W, freq):
    m, n = X.shape
    k = W.shape[1]
    logits = X @ W
    probs = softmax(logits)
    gradient = np.zeros(W.shape)

    for i in range(m):
        for j in range(k):
            indicator = (y[i] == j + 1)
            gradient[:, j] += (indicator - probs[i, j]) * X[i] / freq[j]
    
    return -gradient / (2 * m)

# Performs exact line search using ternary search to find the optimal learning rate.
def ternary_search(X, y, W, freq, g, eta0):
    eta_l = 0
    eta_h = eta0
    
    while compute_loss(X, y, W - eta_h * g, freq) < compute_loss(X, y, W, freq):
        eta_h *= 2

    for _ in range(20):
        eta1 = (2 * eta_l + eta_h) / 3
        eta2 = (eta_l + 2 * eta_h) / 3

        loss1 = compute_loss(X, y, W - eta1 * g, freq)
        loss2 = compute_loss(X, y, W - eta2 * g, freq)

        if loss1 > loss2:
            eta_l = eta1
        else:
            eta_h = eta2 #editted this in algo

    return (eta_l + eta_h) / 2

# Executes mini-batch gradient descent with exact line search to update weights.
def mini_batch_gradient_descent(X, y, W, freq, batch_size, epochs, eta0):
    m, n = X.shape
    for epoch in range(epochs):
        for i in range(0, m, batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            
            g = compute_gradient(X_batch, y_batch, W, freq)
            eta = ternary_search(X_batch, y_batch, W, freq, g, eta0)
            W -= eta * g

    return W

def main(task, train_file, params_file, output_file):
    train_data = pd.read_csv(train_file)
    X = train_data.iloc[:, :-1].values
    y = train_data.iloc[:, -1].values
    
    if task == 'a':
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        with open(params_file, 'r') as f:
            params = f.read().splitlines()
            batch_size = int(params[0])
            eta0 = float(params[1])
            epochs = int(params[2])
            seed = int(params[3])

        np.random.seed(seed)

        n = X.shape[1]
        k = len(np.unique(y))
        W = np.zeros((n, k), dtype=np.float64)
        
        freq = np.array([np.sum(y == j + 1) for j in range(k)], dtype=np.float64)
        
        W = mini_batch_gradient_descent(X, y, W, freq, batch_size, epochs, eta0)
        np.savetxt(output_file, W)

if __name__ == "__main__":
    task = sys.argv[1]
    train_file = sys.argv[2]
    params_file = sys.argv[3]
    output_file = sys.argv[4]
    main(task, train_file, params_file, output_file)
