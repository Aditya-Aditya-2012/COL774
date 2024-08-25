# This file contains all the three training algos to be implemented
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.sparse import csr_matrix
import sys

# loss = -1/2n*(sumi(sumj({yi=j}log(g_wj(xi)))))
# g_wj(x)=exp(wj.T.x)/sumk(exp(wi.T.x))

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return np.float64(exp_logits / np.sum(exp_logits, axis=1, keepdims=True))


def loss_fn(X, y, w, freq):
    n, d = X.shape
    k = w.shape[1]

    # Compute linear outputs
    logits = np.dot(X, w)  # Shape (n, k)

    # Apply softmax to get the predicted probabilities
    g_wj_x = softmax(logits)  # Shape (n, k)

    # Initialize the loss
    loss = 0

    # Compute the loss for each class
    for j in range(k):
        # Select the corresponding softmax output for class j where y is 1 for class j
        selected_probs = g_wj_x[:, j][y[:, j] == 1]  # Shape (number of samples where y == j,)
        
        # Compute the weighted sum of log probabilities
        loss += -np.sum(np.log(selected_probs) / freq[j])
    
    # Average the loss and return
    loss = loss / (2 * n)
    
    return loss
    

def gradient(X, Y, W, freq) :
    #grad = 1/2n(X.T.(U-Y))
    U=softmax(np.dot(X,W))
    G = np.zeros_like(U, dtype=np.float64)
    n = np.float64(Y.shape[0])

    for i in range(Y.shape[0]) :
        index = np.where( Y[i] == 1 )[0][0]
        fact = np.float64(2)*n*freq[index]
        G[i] = (U[i] - Y[i]) / fact

    grad = np.transpose(X)@G

    return grad

def constant_lr(X, Y, W, lr, epochs, batch_size, freq) :
    n_batches = Y.shape[0] // batch_size

    for epoch in range(epochs) :
        loss = 0
        for i in range(n_batches) :
            X_batch = np.float64(X[i*batch_size : (i+1)*batch_size])
            Y_batch = np.float64(Y[i*batch_size : (i+1)*batch_size])

            W -=  np.float64(lr)*gradient(X_batch, Y_batch, W, freq)
            loss += loss_fn(X_batch, Y_batch, W, freq)

        print(f"epoch : {epoch} loss : {np.mean(loss)}")
    return W

def adaptive_lr(X, Y, W, lr, k, epochs, batch_size, freq) :
    n_batches = Y.shape[0] // batch_size

    for epoch in range(epochs) :
        loss = 0
        for i in range(n_batches) :
            X_batch = np.float64(X[i*batch_size : (i+1)*batch_size])
            Y_batch = np.float64(Y[i*batch_size : (i+1)*batch_size])

            W -=  np.float64(lr/(1 + k * (epoch+1)))*gradient(X_batch, Y_batch, W, freq)
            loss += loss_fn(X_batch, Y_batch, W, freq)
        
        print(f"epoch : {epoch} loss : {np.mean(loss)}")

    
    return W

def ternary_search(X, y, W, freq, g, eta0) :
    eta_l = np.float64(0.0)
    eta_h = np.float64(eta0)

    while loss_fn(X, y, W - eta_h * g, freq) < loss_fn(X, y, W, freq):
        eta_h *= 2

    for _ in range(20):
        eta1 = eta_l - ((eta_l - eta_h) / 3)
        eta2 = eta_h - ((eta_h- eta_l) / 3)

        loss1 = loss_fn(X, y, W - eta1 * g, freq)
        loss2 = loss_fn(X, y, W - eta2 * g, freq)

        if loss1 > loss2:
            eta_l = eta1
        else :
            eta_h = eta2

    return (eta_l + eta_h) / 2

def ternary_lr(X, Y, W, lr, epochs, batch_size, freq) :
    n_batches = Y.shape[0] // batch_size

    for epoch in range(epochs) :
        loss = 0
        for i in range(n_batches) :
            X_batch = np.float64(X[i*batch_size : (i+1)*batch_size])
            Y_batch = np.float64(Y[i*batch_size : (i+1)*batch_size])
            
            g = np.float64(gradient(X_batch, Y_batch, W, freq))

            lr = np.float64(ternary_search(X_batch, Y_batch, W, freq, g, np.float64(lr)))

            W -=  lr*g
            loss += loss_fn(X_batch, Y_batch, W, freq)

        print(f"epoch : {epoch} loss : {np.mean(loss)}")
    
    return W

def dataloader(train_path, params_path):
    train = pd.read_csv(train_path)
    x = train.iloc[:, :-1]
    y = train.iloc[:, -1]

    X = x.to_numpy(dtype=np.float64)
    ones_column = np.ones((X.shape[0], 1), dtype=np.float64)
    
    X = np.hstack((ones_column, X))
    #shape X=(n,m+1)

    # one hot encode y as given in the writeup
    y = y.to_numpy(dtype=np.float64)
    Y = pd.get_dummies(y).to_numpy(dtype=np.float64)
    # Y = Y.astype(float64)
    #done, y of shape (n,4)

    #now, load params
    k=0
    lr=0
    with open(params_path, 'r') as f:
            params = f.read().splitlines()
    train_strat = int(params[0])
    if train_strat == 2 :
        lr, k = map(float, params[1].strip().split(','))
    else:
        lr = float(params[1])
    epochs = int(params[2])
    batch_size = int(params[3])

    return X, Y, train_strat, lr, k, epochs, batch_size

##################################################
##################### part a #####################
##################################################

task = sys.argv[1]
train_path = sys.argv[2]
param_file_or_test_file = sys.argv[3]
output_file = sys.argv[4]
modelpredictions_file = sys.argv[5] if len(sys.argv) > 5 else None


if task == 'a' :
    params_path = sys.argv[3]
    X, Y, train_strat, lr, k, epochs, batch_size = dataloader(train_path, params_path)

    m = X.shape[1]
    classes = Y.shape[1]

    W = np.zeros((m, classes), dtype=np.float64)
    freq = np.array([np.sum(Y[:, j]) for j in range(classes)], dtype=np.float64)
    print(freq)
        
    if train_strat == 1 :
        W = constant_lr(X, Y, W, lr, epochs, batch_size, freq)
    
    elif train_strat == 2 :
        W = adaptive_lr(X, Y, W, lr, k, epochs, batch_size, freq)
    
    else :
        W = ternary_lr(X, Y, W, lr, epochs, batch_size, freq)

    np.savetxt(output_file, W)

##################################################
##################### part b #####################
##################################################

from sklearn import preprocessing
import time

def preprocess(train_path, test_path):
    # Load the training data
    train = pd.read_csv(train_path)
    x_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]

    # Convert to numpy array
    X_train = x_train.to_numpy(dtype=np.float64)
    
    # Feature scaling (Normalization)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    
    # Insert bias column (column of ones)
    ones_column_train = np.ones((X_train.shape[0], 1), dtype=np.float64)
    X_train = np.hstack((ones_column_train, X_train))
    
    # One-hot encode the target variable (y)
    Y_train = pd.get_dummies(y_train).to_numpy(dtype=np.float64)

    classes = Y_train.shape[1]
    freq = np.array([np.sum(Y_train[:, j]) for j in range(classes)], dtype=np.float64)
    print(freq)
    
    # Load the test data
    test = pd.read_csv(test_path)
    x_test = test.to_numpy(dtype=np.float64)
    
    # Apply the same scaling to the test data
    X_test = scaler.transform(x_test)
    
    # Insert bias column (column of ones)
    ones_column_test = np.ones((X_test.shape[0], 1), dtype=np.float64)
    X_test = np.hstack((ones_column_test, X_test))
    
    return X_train, Y_train, X_test, freq


def initialize_weights(n_features, n_classes, method='zeros', mean=0.0, std=0.01):
    
    if method == 'zeros':
        W = np.zeros((n_features, n_classes))
    elif method == 'random':
        W = np.random.uniform(-0.01, 0.01, (n_features, n_classes))
    elif method == 'normal':
        W = np.random.normal(mean, std, (n_features, n_classes))
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    
    return W

def hyperparameter_tuning(X_train, Y_train, X_test, y_freq):
    batch_sizes = [16, 32, 64]
    learning_rates = [0.1, 0.5]
    strategies = [1, 2]
    init_methods = ['zeros', 'normal']
    epochs = 10
    best_loss = float('inf')
    best_W = None
    best_params = None
    start_time = time.time()
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for strategy in strategies:
                for init_method in init_methods:
                    start_iter_time = time.time()
                    m = X_train.shape[1]
                    n = Y_train.shape[1]
                    W = initialize_weights(m, n, method=init_method)
                    print(f"Batch Size: {batch_size}, Learning Rate: {learning_rate}, Strategy: {strategy}, Initialization Method: {init_method}")

                    if strategy == 1:
                        W = constant_lr(X_train, Y_train, W, learning_rate, epochs, batch_size, y_freq)
                    elif strategy == 2:
                        k = 10  # Example value, adjust as needed
                        W = adaptive_lr(X_train, Y_train, W, learning_rate, k, epochs, batch_size, y_freq)
                    else:
                        W = ternary_lr(X_train, Y_train, W, learning_rate, epochs, batch_size, y_freq)
                    loss = loss_fn(X_train, Y_train, W, y_freq)
                    if np.isinf(loss):
                        continue
                    elapsed_time = time.time() - start_iter_time
                    print(elapsed_time)
                    if loss < best_loss:
                        best_loss = loss
                        best_W = W
                        best_params = (batch_size, learning_rate, strategy, init_method)
                    # if time.time() - start_time > 600:
                    #     return best_W, None, best_loss
                    
    elapsed_time = time.time() - start_time
    print(best_params)
    print(elapsed_time)
    
    if best_W is None:
        raise ValueError("No valid weights found within the time limit.")
    softmax_probs = softmax(X_test.dot(best_W))
    return best_W, softmax_probs, best_loss


if task == 'b':
    test_path = param_file_or_test_file
    X_train, Y_train, X_test, y_freq = preprocess(train_path, test_path)
    best_W, softmax_probs, best_loss = hyperparameter_tuning(X_train, Y_train, X_test, y_freq)
    np.savetxt(output_file, best_W.flatten(), fmt='%.6f')
    np.savetxt(modelpredictions_file, softmax_probs, delimiter=',', fmt='%.6f')
