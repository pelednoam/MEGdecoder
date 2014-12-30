'''
Created on Mar 20, 2014

@author: noampeled
'''

from matplotlib import animation as animation
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pickle

BEST_ESTIMATOR_FILE = '/Users/noampeled/Copy/Data/MEG/InbalData/Data5_35Hz/svmFiles/data_5_35_ForML_8.mat_inbal_3OrRN_features_windowCutter_sub8_bestEstimator.pkl'
PRED_PARAMS_FILE = '/Users/noampeled/Copy/Data/MEG/InbalData/Data5_35Hz/svmFiles/data_5_35_ForML_8.mat_inbal_3OrRN_features_windowCutter_sub8_predictionsParamters.pkl'
TIME_AXIS_FILE = '/Users/noampeled/Copy/Data/MEG/InbalData/Data5_35Hz/svmFiles/data_5_35_ForML_8.mat_inbal_timeAxis.npy'
DATA_FILE = '/Users/noampeled/Dropbox/postDocMoshe/MEG/AnalyzeMEG/estimatorHeldinProbsPerThreshold.pkl'

power = 0
thresholdIndex = 7

def animateResults():
    bestEstimators = load(BEST_ESTIMATOR_FILE)
#     predParams = load(PRED_PARAMS_FILE)
#     timeAxis = np.load(TIME_AXIS_FILE)    
    data = load(DATA_FILE)
    colors = sns.color_palette(None, len(bestEstimators))
    thresholds = np.linspace(0, 1, 11)
    
    plotProbs(data,bestEstimators,colors,thresholds[thresholdIndex])
    plt.gcf().canvas.mpl_connect('key_press_event', lambda event: updateThreshold(event,data,bestEstimators,thresholds,colors))
    plt.show()    

def updateThreshold(event,data,bestEstimators,thresholds,colors):
    global thresholdIndex
    if event.key == 'right' and thresholdIndex<len(thresholds):
        thresholdIndex += 1
    elif event.key == 'left':
        thresholdIndex -= 1 and thresholdIndex>0
            
    plt.clf()
    plotProbs(data, bestEstimators, colors, thresholds[thresholdIndex])
    plt.draw()
    
def plotProbs(data,bestEstimators,colors,threshold):
    for (featureExtractorName,bestEstimator),color in zip(bestEstimators.iteritems(),colors):
        d = data[threshold][featureExtractorName]
        sns.tsplot(d.probs,d.xAxis,label=featureExtractorName, color=color)
        plt.title('probs>{}'.format(threshold))


def calculatEstimatorHeldinProbsPerThreshold(self,bestEstimators,predParams,timeAxis):
    data={}
    for threshold in np.linspace(0, 1, 11):
        data[threshold] = self.calcEstimatorHeldinProbs(bestEstimators,predParams,timeAxis, threshold)
    utils.save(data, '/Users/noampeled/Dropbox/postDocMoshe/MEG/AnalyzeMEG/estimatorHeldinProbsPerThreshold.pkl')


def calcEstimatorHeldinProbs(self, bestEstimators,predParams,timeAxis, threshold):
    data={}
    colors = sns.color_palette(None, len(bestEstimators))
    print('threshold {}'.format(threshold))
    for (featureExtractorName,bestEstimator),color in zip(bestEstimators.iteritems(),colors):
        bep = bestEstimator.parameters
        timeSelector = self.timeSelector(0, bep.windowSize,predParams.windowsNum,predParams.T)
        startIndices = np.array(timeSelector.windowsGenerator())
        xAxis = timeAxis[startIndices+bep.windowSize/2]
        print('Best results for features extractor: {}, channels: {}, windowSize: {}, kernel: {}, c: {}, gamma: {}'.format(
            featureExtractorName,bep.channelsNum,bep.windowSize,bep.kernel,bep.C,bep.gamma))

        W = bestEstimator.probsScores.shape[0]
        probsScores = np.reshape(bestEstimator.probsScores,(W,-1))
        probsScores=probsScores[:,np.max(probsScores,0)>threshold]
        data[featureExtractorName] = Bunch(probs=probsScores.T, xAxis=xAxis, label=featureExtractorName, color=color)
    return data


def test():
    data = np.linspace(1, 100)
    plt.plot(data**power)
    plt.gcf().canvas.mpl_connect('key_press_event', lambda event: on_keyboard(event,data))
    plt.show()    


def on_keyboard(event,data):
    global power
    if event.key == 'right':
        power += 1
    elif event.key == 'left':
        power -= 1

    plt.clf()
    plt.plot(data**power)
    plt.draw()


def load(fileName):
    with open(fileName, 'r') as pklFile:
        obj = pickle.load(pklFile)
    return obj    

if __name__ == '__main__':
    animateResults()
#     animateResults()