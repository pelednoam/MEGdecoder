'''
Created on May 28, 2014

@author: noampeled
'''
import numpy as np
import os
import scipy.fftpack

from src.commons.utils import plots
from src.commons.utils import MLUtils
import sectionsUtils as su

import seaborn as sns


def calcPS(X, timeStep, minFreq, maxFreq):
    ''' X if for a specific channel (X[:,:,c] '''
    ps = np.abs(np.fft.fft(X)) ** 2
    freqs = scipy.fftpack.fftfreq(X[0, :].size, timeStep)
    return cutPS(ps, freqs, minFreq, maxFreq)


def cutPS(ps, freqs, minFreq, maxFreq):
    idx = np.argsort(freqs)
    ps = ps[:, idx]
    freqs = freqs[idx]
    idx2 = np.where((freqs > minFreq) & (freqs < maxFreq))[0]
    ps = ps[:, idx2]
    freqs = freqs[idx2]
    return freqs, ps


def calcAllAvgPS(X, y, timeStep, sensors=None, minFreq=0, maxFreq=50):
    C = X.shape[2]
    if (sensors is None):
        sensors = range(C)
    freqs, ps = calcPS(X[:, :, 0], timeStep, minFreq, maxFreq)
    pss = np.empty((C, ps.shape[1], 2))
    pss.fill(np.NaN)
    for c in sensors:
        freqs, ps = calcPS(X[:, :, c], timeStep, minFreq, maxFreq)
        pss[c, :, 0] = np.mean(ps[y == 0, :], 0)
        pss[c, :, 1] = np.mean(ps[y == 1, :], 0)
    return freqs, pss


def calcAllPS(X, timeStep, minFreq=0, maxFreq=50, sensors=None,
              cvIndices=None, timeIndices=None, weights=None):
    C = X.shape[2] if weights is None else weights.shape[0]
    if (sensors is None):
        sensors = range(C)
    for k, c in enumerate(sensors):
        xc = su.calcSlice(X, c, cvIndices, timeIndices, weights)
        freqs, ps = calcPS(xc, timeStep, minFreq, maxFreq)
        if (k == 0):
            pss = np.empty((ps.shape[0], ps.shape[1], C))
            pss.fill(np.NaN)
        pss[:, :, k] = ps
    return freqs, pss


def plotScatterPlot(X,y,c,sigFreqs=[12,25],lims=[10e-21,1e-21],labels=['0','1']):
    freqs, ps = calcPS(X[:, :, c])
    x1_0 = ps[y==0,np.where(freqs>sigFreqs[0])[0][0]]
    x2_0 = ps[y==0,np.where(freqs>sigFreqs[1])[0][0]]
    x1_1 = ps[y==1,np.where(freqs>sigFreqs[0])[0][0]]
    x2_1 = ps[y==1,np.where(freqs>sigFreqs[1])[0][0]]

    plots.plt.figure()
    plots.plt.scatter(x1_0, x2_0, color='tomato',label=labels[0]);
    plots.plt.scatter(x1_1, x2_1, color='b',label=labels[1]);
    plots.plt.legend()
    plots.plt.xlim([0,lims[0]])
    plots.plt.ylim([0,lims[1]])
    plots.plt.xlabel('{}Hz'.format(str(sigFreqs[0])))
    plots.plt.ylabel('{}Hz'.format(str(sigFreqs[1])))
    plots.plt.show()

def plotPSAverage(X,y):
    freqs,pss = calcAllAvgPS(X, y)
    plots.graph2(freqs,np.mean(pss[:,:,0],0),np.mean(pss[:,:,1],0),['0','1'],xlim=[0,40],xlabel='Hz')

        
def calcCM1D(X,y,c,thresholf_freq = 12,threshold_ps = 0.5e-20):
    freqs, ps = calcPS(X[:, :, c])
    ypred = np.ones(y.shape)
    idx0 = np.where(ps[:,np.where(freqs>thresholf_freq)[0][0]]>threshold_ps)[0]
    ypred[idx0]=0
    MLUtils.calcConfusionMatrix(y, ypred, doPrint=True)        

def calcCM2D(X,y,c,svc,sigFreqs=[12,25]):
    freqs, ps = calcPS(X[:, :, c])
    ind1 = np.where(freqs>sigFreqs[0])[0][0]
    ind2 = np.where(freqs>sigFreqs[1])[0][0]
    data = np.vstack((ps[:,ind1],ps[:,ind2])).T
    datab,yb = MLUtils.boost(data, y)
    svc.fit(datab,yb)
    ypred = svc.predict(data,False)
    MLUtils.calcConfusionMatrix(y, ypred, doPrint=True)        

def plotPSDiff(X,y,timeStep,folder,sensors=None,labels=['0','1']):
    N,T,C = X.shape 
    if (sensors is None): sensors = range(C)
    freqs,pss = calcAllAvgPS(X, y, timeStep, sensors)
    diff = pss[:,:,0] - pss[:,:,1]
#     plots.matShow(diff)
#     plots.matShow(diff[:,np.where((freqs>10) & (freqs<30))[0]])
#     plots.graph2(freqs,np.mean(pss[:,:,0],0),np.mean(pss[:,:,1],0),labels,xlim=[0,40],xlabel='Frequency (Hz)',ylabel='Power',fileName=os.path.join(folder,'freqs'))
    for c in sensors:
        plots.graph2(freqs,pss[c,:,0],pss[c,:,1],labels,xlim=[0,40],xlabel='Frequency (Hz)',ylabel='Power',fileName=os.path.join(folder,'freqs_{}'.format(c)))        

def plotPSCI(X,y,timeStep,folder,sensors=None,labels=['0','1']):
    freqs,pss = calcAllPS(X,timeStep,sensors=sensors)
    colors = sns.color_palette(None, 2)
    for c in sensors:
        sns.tsplot(pss[y==0, :, c], label=labels[0],color=colors[0])
        sns.tsplot(pss[y==1, :, c], label=labels[1],color=colors[1])
        plots.plt.show()            
    
    

def plotPS(X,y,fileName):
    N,T,C = X.shape 
    freqs,pss = calcAllAvgPS(X, y)
    diffs = pss[:,:,0]-pss[:,:,1]
    maxDiff, minDiff = np.max(diffs),np.min(diffs)
    for c in range(C):
        plots.graph(freqs,diffs[c,:],ylim=[minDiff,maxDiff],xlim=[0,40],xlabel='Hz',fileName='{}_{}'.format(fileName, c))

def plotSensorsPValues(X,y,selector,timeStep, minFreq, maxFreq,fileName):
    N,T,C = X.shape 
    freqs,pss = calcAllPS(X, timeStep, minFreq, maxFreq)
    for c in range(C):
        model = selector.fit(pss[:, :, c],y)
        plots.graph(freqs, model.pvalues_, fileName='{}_{}'.format(fileName, c))


def calcSensorsScores(X,y,selector):
    N,T,C = X.shape 
    freqs,pss = calcAllPS(X)
    scores = np.zeros((C,len(freqs)))
    pvalues = np.zeros((C,len(freqs)))
    for c in range(C):
        model = selector.fit(pss[:,:, c],y)
        scores[c,:] = model.scores_
        pvalues[c,:] = model.pvalues_
#         plots.matShow(scores, 'freqs scores','sensors','freqs')
    plots.barPlot(np.sum(pvalues,1))
    plots.matShow(pvalues, 'freqs scores','sensors','freqs')
