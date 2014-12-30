'''
Created on Feb 26, 2014

@author: noampeled
'''
import numpy as np

from src.commons.utils import MLUtils

'''
X dimentions:
N,T,C = X.shape
N: number of samples
T: time
C: channels
'''

def RMS(X):
    return np.sqrt(np.mean(X*X,2))

def ALL(X):
    N,T,C = X.shape
    newx = np.zeros((N,C,T))
    for i in range(N):
        newx[i,:,:] = X[i,:,:].T
    newx = X.reshape((N,T*C))
    return newx

def cor(X):
    N,_,C = X.shape
    newx = np.zeros((N,C*(C-1)/2))
    for i in range(N):
        coef = np.corrcoef(X[i,:,:].T)
        newx[i,:] = MLUtils.halfMat(coef, includeDiagonal=False)
    return newx    

def cov(X):
    N,_,C = X.shape
    newx = np.zeros((N,C*(C+1)/2))
    for i in range(N):
        coef = np.cov(X[i,:,:].T)
        newx[i,:] = MLUtils.halfMat(coef, includeDiagonal=True)
    return newx    
          
