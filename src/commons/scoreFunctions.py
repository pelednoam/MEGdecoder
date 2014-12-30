'''
Created on Feb 26, 2014

@author: noampeled
'''
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

def accuracyScore(ytrue, ypred):
    return sum(ytrue==ypred)/float(len(ytrue))

def gmeanScore(ytrue, ypred):
    rates = calcRates(ytrue, ypred)
    return gmeanScoreFromRates(rates)

def gmeanScoreFromRates(rates):
    return np.sqrt(rates[0]*rates[1])

def calcRates(ytrue, ypred):
    conMat = confusion_matrix(ytrue, ypred, [0, 1])
    r1 = (conMat[0, 0] / float(np.sum(conMat[0, :])))
    r2 = (conMat[1, 1] / float(np.sum(conMat[1, :])))
    return r1,r2

def AUCScore(ytrue, probs):
    return roc_auc_score(ytrue, probs[:,1])