# Codes for part a and b

############################# IMPORTS #############################
import sys
import os
import numpy as np
import scipy 
import pandas as pd

part = sys.argv[1]            #a or b

def comment(s):
    print('Comment :=>> ' + s)

##################################################################
############################# PART A #############################
##################################################################
def weighted_regression(Y,X,U):   #outputs weight
        # eqn L(w)=||y-xw-b||^2
        # matrix form (Y_XW).T.U.(Y-XW)
        # x=x+column of all 1s of n entries if n samples
        # differentiate the U term
        # final expression W=(X.T.U.X)-1(X.T.U.Y)
        ux=np.zeros_like(X)
        uy=np.zeros_like(Y)
        for i in range(len(X)):
            ux[i]=U[i]*X[i]
        for i in range(len(X)):
            uy[i]=U[i]*Y[i]
         
        xt=np.transpose(X)
        t1=np.linalg.inv(xt@ux)
        t2=(xt@uy)
        w=t1@t2
        return w

if part=='a':
    train_data = sys.argv[2]      #train.csv
    test_data = sys.argv[3]       #test.csv 
    u_file = sys.argv[4]          #sample_weights1.txt or sample_weights2.txt
    pred_file = sys.argv[5]       #modelpredictions.txt (should be created)
    pred_wt_file = sys.argv[6]    #modelweights.txt (should be created)

###################### PREPROCESSING TRAIN #########################

    train=pd.read_csv(train_data)
    # to be split into y and x
    u_inp=open(u_file, "r")
    u_lines=u_inp.read().split('\n')
    u_lines=u_lines[:-1]
    y=train.iloc[:, -1]
    x=train.iloc[:, :-1]
    Y=y.to_numpy(dtype=np.float64)
    X=x.to_numpy(dtype=np.float64)
    ones_column = np.ones((X.shape[0], 1), dtype=np.float64)
    X_b = np.hstack((ones_column, X))
    U=np.asarray(u_lines, dtype=np.float64)

 ###################### PREPROCESSING TEST ########################

    test=pd.read_csv(test_data)
    x_tst=test
    X_tst=x_tst.to_numpy(dtype=np.float64)
    ones_column = np.ones((X_tst.shape[0], 1), dtype=np.float64)
    X_b_tst = np.hstack((ones_column, X_tst))

############################ OUTPUT ###############################

    weights_parta=weighted_regression(Y,X_b,U)
    Y_pr=X_b_tst@weights_parta
    np.savetxt(pred_file, Y_pr, delimiter='\n', fmt='%f')
    np.savetxt(pred_wt_file, weights_parta, delimiter='\n', fmt='%f')

##################################################################
############################# PART B #############################
##################################################################

def ridge_regression(Y,X,λ):
    #eqn for regression with regularization
    #W=(X.T.X+λ.I)^-1Y
    Xt=np.transpose(X)
    XtX=Xt@X
    indices = np.diag_indices_from(XtX)
    XtX[indices]+=λ
    w=np.linalg.inv(XtX)@Xt@Y
    return w

def k_fold_cross_validation(X, Y, k, λ):
    n_samples = X.shape[0]
    fold_size = n_samples // k
    
    indices = np.arange(n_samples)
    
    fold_errors = []
    
    #set k=10
    for i in range(k):
        start = i * fold_size
        end = start + fold_size 
        
        test_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        
        #split
        X_train, X_test = X[train_indices], X[test_indices]
        Y_train, Y_test = Y[train_indices], Y[test_indices]
        
        w = ridge_regression(Y_train, X_train, λ)
        
        Y_pred = X_test @ w 
        
        error = np.mean((Y_test - Y_pred) ** 2)
        fold_errors.append(error)
    
    # Return the mean error across all folds
    return np.sum(fold_errors)

def find_best_λ(X, Y, λ_values, k=10):
    best_λ = None
    best_score = float('inf')
    
    for λ in λ_values:
        score = k_fold_cross_validation(X, Y, k, λ)
        
        if score < best_score:
            best_score = score
            best_λ = λ
            
    return best_λ, best_score
    

if part=='b':
    train_data = sys.argv[2]      #train.csv
    test_data = sys.argv[3]       #test.csv 
    regu_file = sys.argv[4]       #regularization.txt 
    pred_file = sys.argv[5]       #modelpredictions.txt (should be created)
    pred_wt_file = sys.argv[6]    #modelweights.txt (should be created)
    best_λ_file = sys.argv[7]     #bestλ.txt  

###################### PREPROCESSING TRAIN #########################

    train=pd.read_csv(train_data)
    # to be split into y and x
    λ_inp=open(regu_file, "r")
    λ_lines=λ_inp.read().split('\n')
    λ_lines=λ_lines[:-1]

    y=train.iloc[:, -1]
    x=train.iloc[:, :-1]
    Y=y.to_numpy(dtype=np.float64)
    X=x.to_numpy(dtype=np.float64)
    ones_column = np.ones((X.shape[0], 1), dtype=np.float64)
    X_b = np.hstack((ones_column, X))
    λ_values=np.asarray(λ_lines, dtype=np.float64)

    X_b_new = X_b[:-4]
    Y_new = Y[:-4]

 ###################### PREPROCESSING TEST ########################

    test=pd.read_csv(test_data)
    x_tst=test
    X_tst=x_tst.to_numpy(dtype=np.float64)
    ones_column = np.ones((X_tst.shape[0], 1), dtype=np.float64)
    X_b_tst = np.hstack((ones_column, X_tst))

############################ OUTPUT ###############################
    
    best_λ, best_score=find_best_λ(X_b_new, Y_new, λ_values, 10)
    weights_partb=ridge_regression(Y_new, X_b_new, best_λ)
    Y_pr=X_b_tst@weights_partb

    np.savetxt(pred_file, Y_pr, delimiter='\n', fmt='%f')
    np.savetxt(pred_wt_file, weights_partb, delimiter='\n', fmt='%f')
    
    with open(best_λ_file, 'w') as file:
        file.write(f'{best_λ}\n')
    
    
