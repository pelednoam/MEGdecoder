'''
Created on Nov 28, 2011

@author: noam
'''

import numpy as np
import numpy.linalg as lin
import math
from collections import Counter
import pandas as pd
import scipy.fftpack
from scipy import interp  
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.base import NeighborsWarning
from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import KernelDensity
import pylab as pl
import random
from random import randrange, choice
from src.commons.utils import utils
from math import factorial

import warnings
warnings.simplefilter("ignore", NeighborsWarning)

def crossvalidation(x, y, K=10):
    for k in range(K):
        training_idx = [i for i in range(len(y)) if i % K != k]
        test_idx = [i for i in range(len(y)) if i % K == k]
        yield x[training_idx, :], x[test_idx, :], y[training_idx], y[test_idx]

def normalizeData(x, isNP=True):
    xmean = np.mean(x, 0)
    xstd = np.std(x, 0)          
    x = (x - xmean) / xstd
    x[np.where(np.isnan(x))] = 0
#    for i in range(x.shape[1]):
#        if (xstd[i] != 0):
#            x[:, i] = (x[:, i] - xmean[i]) / xstd[i]
#        else:
#            x[:, i] = xmean[i]      
    if (not isNP):
        xl = []
        for i in range(x.shape[0]):  
            xl.append(list(x[i, :]))
        return (xl, xmean, xstd)
    else:
        return (x, xmean, xstd)

def calcCova(x, elmsNum=50):
    elmsNum = min(elmsNum, x.shape[1])
    newx = x[:, -elmsNum:]
    xmean = np.mean(newx, 1)
    xmean = np.mean(xmean)
    newx = newx - xmean             
    cov = np.dot(newx, newx.T) 
    cov = cov / elmsNum 
    return cov

def covMatrix(x):
    N = x.shape[0]
    cova = np.zeros((N * (N - 1) / 2))
    cov = np.cov(x)
    k = 0
    for i in range(N):
        for j in range(N):
            if (j > i):
                cova[k] = cov[i, j]
                k += 1
    return cova
    
#
# def calcCorr(x):
#    cov = calcCov(x)
#    var = np.var(x, 0)
#    return cov / var

def removeNone(x, y, z=None):
    notNoneInd = (~np.isnan(y))
    x = x[notNoneInd]
    y = y[notNoneInd]
    if (z is not None):
        z = z[notNoneInd]
    return (x, y, z)

def checkModulo(lr, modulo):
    if (modulo == 10):
        lr.classes = np.zeros((len(lr.model.label_), 2))
        for i, val in enumerate(lr.model.label_):
            h1 = int(val / 10)
            h2 = val % 10
            lr.classes[i][0] = h1
            lr.classes[i][1] = h2
    elif (modulo == 100):
        lr.classes = np.zeros((len(lr.model.label_), 3))
        for i, val in enumerate(lr.model.label_):
            h1 = int(val / 100)
            h2 = (val - h1 * 100) / 10
            h3 = val % 10
            lr.classes[i][0] = h1
            lr.classes[i][1] = h2        
            lr.classes[i][2] = h3
    elif (modulo == 1000):
        lr.classes = np.zeros((len(lr.model.label_), 4))
        for i, val in enumerate(lr.model.label_):
            h1 = int(val / 1000)
            h2 = (val - h1 * 1000) / 100
            h3 = (val - h1 * 1000 - h2 * 100) / 10 
            h4 = val % 10
            lr.classes[i][0] = h1
            lr.classes[i][1] = h2        
            lr.classes[i][2] = h3
            lr.classes[i][3] = h4
    else: 
        lr.classes = lr.model.label_
    print ('unique y values: %d' % len(lr.classes))

def buildSamplesWeights(y, samplesWeights=None):
    if not samplesWeights: samplesWeights = {}
    weights = []
    if (samplesWeights.keys() != []):
        for i in range(len(y)):
            weights.append(samplesWeights[y[i]])
    return weights

def calcConfusionMatrix(ytrue, ypred, labels=None, doPrint=True):
    if not labels: labels = ['label1', 'label2']
    s=''
    conMat = confusion_matrix(ytrue, ypred, [0, 1])
    totalAccuracy = (conMat[0, 0] + conMat[1, 1]) / float(np.sum(conMat))
    label1Accuracy = (conMat[0, 0] / float(np.sum(conMat[0, :])))
    label2Accuracy = (conMat[1, 1] / float(np.sum(conMat[1, :])))
    s+=conMat.__str__()
    s+=('\nTotal accuracy: %.5f' % totalAccuracy)
    s+=('\n%s accuracy: %.5f' %(labels[0],label1Accuracy))
    s+=('\n%s accuracy: %.5f' %(labels[1],label2Accuracy))
    s+=('\nG-Mean accuracy: %.5f' % math.sqrt((label1Accuracy * label2Accuracy)))
    if (doPrint):
        print(s)
#        print('Size: %d' % np.sum(conMat))
    return (conMat, totalAccuracy, label1Accuracy, label2Accuracy,s)

def calcPredition(probs, threshold=0.5):
    ypred = []
    for p in probs:
        ypred.append(0 if p[0] > threshold else 1)
    return ypred


def boost(x, y, pickRand=True):
    actionsCount = Counter(y)
    if (len(actionsCount) < 2):
        return x, y

    minorityLabel, majorityLabel = (0, 1) \
        if actionsCount[0] < actionsCount[1] else (1, 0)
    boostFactor = (float(actionsCount[majorityLabel]) /
                   float(actionsCount[minorityLabel])) - 1
    if (boostFactor > 0):
        inds = np.where(y == minorityLabel)[0]
        targetx = np.array(x[inds, :])
        targety = y[inds]
        boostx = np.array(x)
        boosty = y[:]
        for _ in range(int(boostFactor)):
            boostx = np.concatenate((boostx, targetx), 0)
            boosty = np.concatenate((boosty, targety))
        more = int(len(inds) * (boostFactor - int(boostFactor)))
        if (more > 0):
            moreInds = random.sample(inds, more) if (pickRand) else inds[:more]
            boostx = np.concatenate((boostx, np.array(x[moreInds, :])), 0)
            boosty = np.concatenate((boosty, y[moreInds]))
        return (boostx, boosty)
    else:
        return x, y


def boostSmote(x, y, k=5, similarity=0.5, pickRand=True):
    actionsCount = Counter(y)
    if (len(actionsCount) < 2):
        return x, y
    minorityLabel, majorityLabel = (0, 1) \
        if actionsCount[0] < actionsCount[1] else (1, 0)
    boostFactor = actionsCount[majorityLabel] / float(
        actionsCount[minorityLabel]) - 1
    if (boostFactor > 0):
        inds = np.where(y == minorityLabel)[0]
        targetx = np.array(x[inds, :])
        targety = y[inds]
        boosty = y[:]
        intBoostFactor = int(boostFactor)
        smotex, neigh = smote(targetx, intBoostFactor * 100, k, similarity)
        boostx = np.concatenate((np.array(x), smotex), 0)
        for _ in range(intBoostFactor):
            boosty = np.concatenate((boosty, targety))
        more = int(len(inds) * (boostFactor - intBoostFactor))
        if (more > 0):
            moreInds = random.sample(inds, more) if (pickRand) else inds[:more]
            moresmotex = generateSyntheticRecords(targetx, moreInds, k,
                similarity, neigh)
            boostx = np.concatenate((boostx, moresmotex), 0)
            boosty = np.concatenate((boosty, y[moreInds]))
        return (boostx, boosty)
    else:
        return x, y


def boostAllData(x, y, boostFactor= -1, pickRand=True, useSmote=False, knn=2, similarity=1):
    if (boostFactor == 0): 
        return (x, y)
    
    if (useSmote):
        inds0 = np.where(y == 0)[0]
        inds1 = np.where(y == 1)[0]
        x1, y1 = boostSmoteAllData(x[inds0], y[inds0], boostFactor, k=knn, similarity=similarity)
        x2, y2 = boostSmoteAllData(x[inds1], y[inds1], boostFactor, k=knn, similarity=similarity)
        boostx = np.concatenate((x1, x2), 0)
        boosty = np.concatenate((y1, y2))
        return boostx,boosty
    
    x, y = utils.underSampling(x, y)    
    boostx, boosty = x, y
    for _ in range(int(boostFactor)):
        boostx = np.vstack((boostx, x))
        boosty = np.hstack((boosty, y))
    
    more = int(len(y) * (boostFactor - int(boostFactor)))
    if (more > 0):
#        inds = range(len(y))
        inds0 = np.where(y == 0)[0]
        inds1 = np.where(y == 1)[0]
        more0 = int(more / 2)
        more1 = more - more0
        moreInds0 = random.sample(inds0, more0) if (pickRand) else inds0[:more0]
        moreInds1 = random.sample(inds1, more1) if (pickRand) else inds1[:more1]
        moreInds = np.concatenate((moreInds0, moreInds1))
#        else:
#            moreInds = random.sample(inds, more) if (pickRand) else inds[:more]
        moreInds = moreInds.astype(np.int64)            
        boostx = np.concatenate((boostx, np.array(x[moreInds, :])), 0)
        boosty = np.concatenate((boosty, y[moreInds]))            
    return (boostx, boosty)

def boostSmoteAllData(x, y, boostFactor, k, similarity=0.5, pickRand=True):
    IntBoostFactor = int(boostFactor)
    if (IntBoostFactor == 0): 
        return (x, y)
    inds = range(len(y))
    boosty = y
    smotex, neigh = smote(x, IntBoostFactor * 100, k, similarity)
    boostx = np.concatenate((x, smotex), 0)
    for _ in range(IntBoostFactor):
        boosty = np.hstack((boosty, y))
    more = int(len(inds) * (boostFactor - IntBoostFactor))
    if (more > 0):
        moreInds = random.sample(inds, more) if (pickRand) else inds[:more]
        moresmotex = generateSyntheticRecords(x, moreInds, k, similarity, neigh)
        boostx = np.concatenate((boostx, moresmotex), 0)
        boosty = np.concatenate((boosty, y[moreInds]))  
    return (boostx, boosty)

def shuffle(x):  # seed=13
    idx = np.arange(x.shape[0])
    np.random.seed(13)
    np.random.shuffle(idx)
    xshuf = x[idx]
    return (xshuf, idx)

def firstKElements(N, k, seed=13):
    idx = np.arange(N)
    np.random.seed(seed)
    np.random.shuffle(idx)
    return (idx[:k], idx[k:])

def halfMat(x, includeDiagonal):
    k = 0 if includeDiagonal else -1
    tril = np.tril(x, k)
    ret = []
    for i in range(tril.shape[0]):
        for j in range(i + 1 + k):
            ret.append((tril[i, j]))
    ret = np.array(ret)
    return ret

def scaleBoostShuffle(x, y, train, test, boostFactor=0, boostLabel=1, doScale=True, doShuffle=True):
    xtrain = x[train]
    xtest = x[test]
    ytrain = y[train]
    if (doScale):
        scaler = preprocessing.StandardScaler().fit(xtest)
        xtrain = scaler.transform(xtrain)
        xtest = scaler.transform(xtest)
    (xtrainb, ytrainb) = boost(xtrain, ytrain, boostLabel, boostFactor)
    if (doShuffle):
        (xtrainb, idx) = shuffle(xtrainb)
        ytrainb = ytrainb[idx]
    return (xtrainb, ytrainb, xtest, y[test])

def scale(x):
    return preprocessing.StandardScaler().fit(x).transform(x)

def scaleTrainTest(xtrain, xtest):
    scaler = preprocessing.StandardScaler().fit(xtest)
    xtrain = scaler.transform(xtrain)
    xtest = scaler.transform(xtest)
    return xtrain, xtest
        
def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s = np.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:  
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len:-window_len + 1]     

def kalman1d(x, R=0.1 ** 2):
    # R: estimate of measurement variance, change to see effect
    n_iter = len(x)
    sz = (n_iter,)  # size of array
    v = np.var(x)
        
    # allocate space for arrays
    xhat = np.zeros(sz)  # a posteri estimate of x
    P = np.zeros(sz)  # a posteri error estimate
    xhatminus = np.zeros(sz)  # a priori estimate of x
    Pminus = np.zeros(sz)  # a priori error estimate
    K = np.zeros(sz)  # gain or blending factor
    
    # intial guesses
    xhat[0] = x[0]
    P[0] = 1.0
    
    for k in range(1, n_iter):
        # time update
        xhatminus[k] = xhat[k - 1]
        Pminus[k] = P[k - 1] + v
    
        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (x[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
        
    return xhat


def fft(x):
    FFT = abs(scipy.fft(x))
    freqs = scipy.fftpack.fftfreq(x.size, len(x))  # t[1]-t[0])
    FFT[freqs == 0] = 0

def movingAvg(x, windowSize):
    return pd.rolling_mean(x, windowSize)[windowSize - 1:]

def movingAvgNoNone(x, windowSize, minPeriods=1):
    func = lambda z: z[pd.notnull(z)].mean()
#    newx = pd.Series(x)    
    return pd.rolling_apply(x, windowSize, func, minPeriods)

def movingFunc(x, windowSize, func, minPeriods=None):
    return pd.rolling_apply(x, windowSize, func, minPeriods)

def ewMovingAvg(x, span):
    return pd.ewma(x, span=span)

def kalman2d(x, P, measurement, R,
              motion=np.matrix('0. 0. 0. 0.').T,
              Q=np.matrix(np.eye(4))):
    """
    Parameters:    
    x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
    P: initial uncertainty convariance matrix
    measurement: observed position
    R: measurement noise 
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    """
    return _kalman2d(x, P, measurement, R, motion, Q,
                  F=np.matrix('''
                      1. 0. 1. 0.;
                      0. 1. 0. 1.;
                      0. 0. 1. 0.;
                      0. 0. 0. 1.
                      '''),
                  H=np.matrix('''
                      1. 0. 0. 0.;
                      0. 1. 0. 0.'''))

def _kalman2d(x, P, measurement, R, motion, Q, F, H):
    '''
    Parameters:
    x: initial state
    P: initial uncertainty convariance matrix
    measurement: observed position (same shape as H*x)
    R: measurement noise (same shape as H)
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    F: next state function: x_prime = F*x
    H: measurement function: position = H*x

    Return: the updated and predicted new values for (x, P)

    See also http://en.wikipedia.org/wiki/Kalman_filter

    This version of kalman can be applied to many different situations by
    appropriately defining F and H 
    '''
    # UPDATE x, P based on measurement m    
    # distance between measured and current position-belief
    y = np.matrix(measurement).T - H * x
    S = H * P * H.T + R  # residual convariance
    K = P * H.T * S.I  # Kalman gain
    x = x + K * y
    I = np.matrix(np.eye(F.shape[0]))  # identity matrix
    P = (I - K * H) * P

    # PREDICT x, P based on motion
    x = F * x + motion
    P = F * P * F.T + Q

    return x, P

def linearRegression(x, y, doPlot=False):
    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y)[0]
    if (doPlot):
        import matplotlib.pyplot as plt
        plt.plot(x, y, 'o', label='Original data', markersize=10)
        plt.plot(x, a * x + b, 'r', label='Fitted line')
        plt.legend()
        plt.show()    
    return (a, b)


def smote(T, N, k, similarity):
    """
    Returns (N/100) * n_minority_samples synthetic minority samples.

    Parameters
    ----------
    T : array-like, shape = [n_minority_samples, n_features]
        Holds the minority samples
    N : percetange of new synthetic samples: 
        n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
    k : int. Number of nearest neighbours. 

    Returns
    -------
    S : array, shape = [(N/100) * n_minority_samples, n_features]
    """
    n_minority_samples, n_features = T.shape
    k = min(k, round(n_minority_samples - 1))
    k = max(k, 2)
    if N < 100:
        # create synthetic samples only for a subset of T.
        # TODO: select random minortiy samples
        N = 100
        return []

    if (N % 100) != 0:
        raise ValueError("N must be < 100 or multiple of 100")

    N = int(N) / 100
    n_synthetic_samples = N * n_minority_samples
    S = np.zeros(shape=(n_synthetic_samples, n_features))

    # Learn nearest neighbours
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(T)

    # Calculate synthetic samples
    for i in xrange(n_minority_samples):
        dists, nn = neigh.kneighbors(T[i], return_distance=True)
        dists, nn = dists[0], list(nn[0])
        # if some records are duplicated, i sometimes won't be in nn
        if (i in nn): 
            nn.remove(i)   
            dists = dists[1:]
        for n in xrange(N):
            if (len(nn) > 0): 
                nn_index = choice(nn)
#                probs = utils.distsToAccProbs(dists)
#                nn_index = nn[utils.findCutof(probs)]
                if (nn_index == i): raise Exception('erorr in smote!')
            else:
                nn_index = i  # duplicate i                
            dif = T[nn_index] - T[i]
            gap = np.random.random() * similarity
            S[n + i * N, :] = T[i, :] + gap * dif[:]
    return S, neigh

def generateSyntheticRecords(T, inds, k, similarity, neigh=None):
    n_minority_samples, n_features = T.shape
    k = min(k, round(n_minority_samples - 1))
    k = max(k, 2)

    n_synthetic_samples = len(inds)
    S = np.zeros(shape=(n_synthetic_samples, n_features))
    
    # Learn nearest neighbours
    if (neigh is None): 
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(T)

    # Calculate synthetic samples
    for synInd in range(n_synthetic_samples):
        i = choice(range(n_minority_samples))
        dists, nn = neigh.kneighbors(T[i], return_distance=True)
        dists, nn = dists[0], list(nn[0])
        # if some records are duplicated, i sometimes won't be in nn
        if (i in nn): 
            nn.remove(i)   
            dists = dists[1:]
        if (len(nn) > 0): 
            nn_index = choice(nn)
            if (nn_index == i): raise Exception('erorr in smote!')
        else:
            nn_index = i  # duplicate i                
        dif = T[nn_index] - T[i]
        gap = np.random.random() * similarity
        S[synInd, :] = T[i, :] + gap * dif[:]
    return S
    

def knn(x, i, k, exIndices):
    k += len(exIndices)
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(x)
    nn = list(neigh.kneighbors(x[i], return_distance=False)[0])
    nn = list(set(nn) - set(exIndices))
    return nn
        
def fitNeighbors(X):
    neigh = NearestNeighbors(n_neighbors=X.shape[0])
    neigh.fit(X)
    return neigh 

def calcKNeighborsProbs(X, neigh, ind, fitIndices, exIndices):
    dists, nn = neigh.kneighbors(X[ind], return_distance=True)
    dists, nn = dists[0], nn[0]
    inds = [x for x in nn if x in fitIndices and x not in exIndices and x != ind]
    dists = dists[1:]  # Remove the first item - it's ind
    probs = utils.distsToAccProbs(dists[inds])
    return probs, inds

def pickkNN(X, neigh, fitIndices, k, exIndices): 
    nnRet = set()
    exIndicesSet = set(exIndices)
    # Calc the kneighbors for every exIndices
    if (len(exIndices) > k): print ('len(exIndices)>k! %d>%d' % (len(exIndices), k)) 
    probsInds = {}
    for ind in exIndices:
        dists, nn = neigh.kneighbors(X[ind], return_distance=True)
        dists, nn = dists[0], nn[0]
        if (nn[0] != ind):
            raise Exception('ind is not the first!')
        nn = nn[1:]  # Remove the current item, ind        
        inds = [x for x in nn if x in fitIndices and x not in exIndicesSet]
        dists = dists[1:]  
        cumsumProbs = utils.distsToAccProbs(dists[inds])
        probsInds[ind] = (cumsumProbs, inds)
    for _ in range(k):
        ind = choice(exIndices)
        cumsumProbs, inds = probsInds[ind]
        cutofInd = utils.findCutofWithEx(cumsumProbs, nnRet, inds)
        if (inds[cutofInd] in nnRet):
            raise Exception('already in nnRet!')
        nnRet.add(inds[cutofInd])
    if (len(nnRet) != k): 
        raise Exception('nnRet (%d) not in like k length %d' % (len(nnRet), k))
    return nnRet

def pickkNNFromLast(X, neigh, fitIndices, k, exIndices): 
    exIndices = [exIndices[-1]]
    ind = 0
    nnRet, nnEx = set(), set()
    # Calc the kneighbors for every exIndices
    if (len(exIndices) > k): print ('len(exIndices)>k! %d>%d' % (len(exIndices), k)) 
    exNN = {}
    dists, nn = neigh.kneighbors(X[ind], return_distance=True)
    dists, nn = dists[0], nn[0]
    inds = [x for x in nn if x in fitIndices and x not in exIndices and x != ind]
    dists = dists[1:]  # Remove the first item - it's ind
    probs = utils.distsToProbs(dists[inds])
    exNN[ind] = (probs, inds)
    for _ in range(k):
        ind = exIndices[0]  # choice(exIndices)
#        probs, inds = exNN[ind]
        cutofInd = utils.findCutof(probs, nnRet, inds)
        probs[cutofInd] = 0
        if (inds[cutofInd] in nnRet):
            raise Exception('already in nnRet!')
        nnRet.add(inds[cutofInd])
    if (len(nnRet) != k): 
        raise Exception('nnRet (%d) not in like k length %d' % (len(nnRet), k))
    return nnRet


def calcGmean(r1, r2):
    return math.sqrt(r1 * r2)

def calcGmeans(r1, r2):
    gmean = []
    for rr1, rr2 in zip(r1, r2):
        gmean.append(calcGmean(rr1, rr2))
    return gmean
    
def cosineDistance(u, v):
    """
    Returns the cosine of the angle between vectors v and u. This is equal to
    u.v / |u||v|.
    """
    return np.dot(u, v) / (math.sqrt(np.dot(u, u)) * math.sqrt(np.dot(v, v)))

def cosineDistanceFlatMatrix(x):
    P = x.shape[1]
    ret = np.zeros((P * (P - 1) / 2))
    ind = 0
    for i in range(P):
        for j in range(i + 1, P):
            ret[ind] = cosineDistance(x[:, i], x[:, j])
            ind += 1
    return ret

def calcFoldACU(y, probs, mean_tpr, mean_fpr):
    fpr, tpr, thresholds = roc_curve(y, probs[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    return roc_auc, mean_tpr, mean_fpr, fpr, tpr

def plotFoldROC(fpr, tpr, key, auc):
    pl.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (key, auc))    

def calcAUC(probs, y, doPlot=True,ROCFigName=''):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    fpr, tpr, thresholds = roc_curve(y, probs[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    return plotROC(mean_tpr, mean_fpr, doPlot, fileName=ROCFigName)

def calcAUCs(probsArr, ys, labels=None, doPlot=True):
    if not labels: labels = []
    tpr_fpr,aucs = [],[]
    for (probs, y) in zip(probsArr, ys):
        mean_auc , mean_tpr, mean_fpr = calcAUC(probs, y, False)
        aucs.append(mean_auc)
        tpr_fpr.append((mean_tpr, mean_fpr))
    if (doPlot):
        plotNROC(tpr_fpr, labels, doPlot)
    return aucs

def plotROC(mean_tpr, mean_fpr, doPlot, lenCV=1, fileName=''):
    if (doPlot):    
        pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Chance')
    
    mean_tpr /= float(lenCV)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    if (doPlot):
        pl.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
        pl.xlim([-0.05, 1.05])
        pl.ylim([-0.05, 1.05])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('ROC curve')
        pl.legend(loc="lower right")
        if (fileName!=''):
            pl.savefig(fileName)
        pl.show()    
    return mean_auc, mean_tpr, mean_fpr

def plotNROC(tpr_fpr_tuple, labels, doPlot, lenCV=1, linewidth = 3):
    if (doPlot):    
        pl.plot([0, 1], [0, 1], 'k--', color=(0.6, 0.6, 0.6), label='Chance')

    for (mean_tpr,mean_fpr), label in zip(tpr_fpr_tuple,labels):
        print(label)
        mean_tpr /= float(lenCV)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        if (doPlot):
            print('%s (%0.2f)' % (label, mean_auc))
            pl.plot(mean_fpr, mean_tpr, label='%s (%0.2f)' % (label, mean_auc))
#             pl.plot(mean_fpr, mean_tpr, next(lines), label='%s (%0.2f)' % (label, mean_auc), lw=linewidth)
    pl.xlim([-0.05, 1.05])
    pl.ylim([-0.05, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
#    pl.title('ROC curve')
    pl.legend(loc="lower right")
    pl.show()    
    return mean_auc

def distanceToProb(dist):
    if (dist.ndim==1):
        return np.array([np.exp(d)/(np.exp(d)+np.exp(-d)) for d in dist])
    else:
        return np.array([np.exp(d[0])/(np.exp(d[0])+np.exp(-d[0])) for d in dist])

def probToClass(probs):
    return np.array([0 if p<0.5 else 1 for p in probs])

def probsToPreds(probs):
    return np.array([0 if p[0]>0.5 else 1 for p in probs]) 

def printAccuracyResults(true, prediction, playerID='', labels=None):
    if not labels: labels = ['label1', 'label2']
    (conMat, totalAccuracy, label1Accuracy, label2Accuracy) = calcConfusionMatrix(true, prediction, doPrint=False)
    gmean = calcGmean(label1Accuracy,label2Accuracy)
    if (playerID==''):
        print('%s %.5f (%d), %s %.5f (%d), gmean %.5f' % (labels[0], label1Accuracy, sum(conMat[0]), labels[1], label2Accuracy, sum(conMat[1]),gmean))
    else:
        # noinspection PyStringFormat
        print('Player %d, %s %.5f (%d), %s %.5f (%d), gmean %.5f' % (playerID, labels[0], label1Accuracy, sum(conMat[0]), labels[0], label2Accuracy, sum(conMat[1]),gmean))
    return label1Accuracy, label2Accuracy, gmean

def rms(x):
    return np.sqrt(np.mean(x**2,0))

def kde(xAxis,data,density=1000,kernel='gaussian', bandwidth=0.75):
    xSamples = np.linspace(0, len(xAxis), density)[:, np.newaxis]
    xAxis = np.linspace(xAxis[0],xAxis[-1],density)
    if (data.shape[0]>0):
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(data)
        log_dens = kde.score_samples(xSamples)
        return np.exp(log_dens),xAxis
    else:
        return np.zeros(xAxis.shape),xAxis
    
def calcPeaks(scores,threshold,xAxis):
    peakWindows = utils.whereInMatrix(scores.T,threshold)#[:, np.newaxis] 
    peakWindowsNorm = np.concatenate((np.ones(scores.size-len(peakWindows))*(-10),peakWindows))[:, np.newaxis]
    peaksRatio = len(peakWindows)/float(scores.size)
    dens,densX = kde(xAxis, peakWindowsNorm)#, kernel='tophat')
    return dens,densX,peaksRatio


def savitzkyGolaySmooth(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')