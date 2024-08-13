# Codes for part a and b

############################# IMPORTS #############################
import sys
import os
import numpy as np
import scipy 
import pandas as pd

part = sys.argv[1]            #a or b
train_data = sys.argv[2]      #train.csv
test_data = sys.argv[3]       #test.csv 
u_file = sys.argv[4]           #sample_weights1.txt or sample_weights2.txt
pred_file = sys.argv[5]       #modelpredictions.txt (should be created)
pred_wt_file = sys.argv[6]    #modelweights.txt (should be created)

def comment(s):
    print('Comment :=>> ' + s)

############################# PREPROCESSING #############################
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

# @harshul code is done but weight is not coming right, please look at the code and debug
############################# PART A #############################
def part_a(Y,X,U):   #outputs weight
        # eqn L(w)=||y-xw-b||^2
        # matrix form (Y_XW).T.U.(Y-XW)
        # x=x+column of all 1s of n entries if n samples
        # differentiate the U term
        # final expression W=(X.T.U.X)-1(X.T.U.Y)
        ux=X
        uy=Y
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
    if os.path.exists(pred_file) == False:
        comment("Prediction file not created for part a")
        exit()
    if os.path.exists(pred_wt_file) == False:
        comment("Weight file not created for part a")
        exit()
    print(part_a(Y,X_b,U))

