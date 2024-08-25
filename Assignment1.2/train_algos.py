# This file contains all the three training algos to be implemented

import numpy as np
import scipy
import math
from scipy.special import softmax

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
        for i in range(n_batches) :
            X_batch = np.float64(X[i*batch_size : (i+1)*batch_size])
            Y_batch = np.float64(Y[i*batch_size : (i+1)*batch_size])

            W -=  np.float64(lr/(1 + k * (epoch+1)))*gradient(X_batch, Y_batch, W, freq)
    
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