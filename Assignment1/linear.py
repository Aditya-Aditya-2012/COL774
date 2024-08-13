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
u_inp = sys.argv[4]           #sample_weights1.txt or sample_weights2.txt
pred_file = sys.argv[5]       #modelpredictions.txt (should be created)
pred_wt_file = sys.argv[6]    #modelweights.txt (should be created)

def comment(s):
    print('Comment :=>> ' + s)

train=pd.read_csv(train_data)
# to be split into y and x
u_lines=u_inp.read().split('\n')
u_lines=u_lines[:-1]
u=np.asarray(u_lines, dtype=np.float128) #TO DIAGNOLIZE
############################# PART A #############################

if part=='a':
    if os.path.exists(pred_file) == False:
        comment("Prediction file not created for part a")
        exit()
    if os.path.exists(pred_wt_file) == False:
        comment("Weight file not created for part a")
        exit()
    def part_a(y,x,u):   #outputs weight
        # eqn L(w)=||y-xw-b||^2
        # matrix form (Y_XW).T.U.(Y-XW)
        # x=x+column of all 1s of n entries if n samples
        # differentiate the U term
        # final expression W=(X.T.U.X)-1(X.T.U.Y) 
        xt=np.transpose(x)
        t1=np.linalg.inv(xt@u@x)
        t2=(xt@u@y)
        w=t1@t2
        return w

