# This file contains all the three training algos to be implemented
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.sparse import csr_matrix
import sys
from train_algos import *

##################################################
##################### Input ######################
##################################################

task = sys.argv[1]
train_path = sys.argv[2]
params_path_or_test_path = sys.argv[3]
output_file = sys.argv[4]
modelpredictions_file = sys.argv[5] if len(sys.argv) > 5 else None

##################################################
################# Preprocess A ###################
##################################################

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
params_path = sys.argv[3]
output_file = sys.argv[4]

X, Y, train_strat, lr, k, epochs, batch_size = dataloader(train_path, params_path)

if task == 'a' :
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

    W_flattened = W.flatten()
    np.savetxt(output_file, W_flattened)

##################################################
################# Preprocess B ###################
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

##################################################
##################### part b #####################
##################################################
        


    