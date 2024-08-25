#Assignment 1.2 part a

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.sparse import csr_matrix
import math
import sys
from train_algos import *

##################################################
##################### DATA #######################
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
    k=0.
    lr=0.
    with open(params_path, 'r') as f:
            params = f.read().splitlines()
    train_strat = int(params[0])
    if train_strat == 2 :
        lr, k = map(np.float64, params[1].strip().split(','))
    else:
        lr = np.float64(params[1])
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
        


    