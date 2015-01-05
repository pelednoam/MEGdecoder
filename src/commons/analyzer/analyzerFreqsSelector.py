'''
Created on Jun 1, 2014

@author: noampeled
'''

from src.commons.analyzer.analyzer import Analyzer
from src.commons.analyzer.analyzerSelector import AnalyzerSelector
from src.commons.selectors.frequenciesSelector import FrequenciesSelector
from src.commons.utils import mpHelper
from src.commons.utils import utils
from src.commons.utils import freqsUtils
from src.commons.utils import tablesUtils as tabu
from src.commons.utils import sectionsUtils as su

import numpy as np
from collections import namedtuple
import itertools
import os
from sklearn.datasets.base import Bunch


class AnalyzerFreqsSelector(AnalyzerSelector):

    def _prepareCVParams(self, p):
        p = p.merge(p.kwargs)
        params = []
        paramsNum = len(list(p.cv))
        T = p.x.shape[1]
        print('T is {}'.format(T))
        timeStep = self.calcTimeStep(T)
        pss, freqs = su.preCalcPS(p.x, min(p.minFreqs), max(p.maxFreqs),
            timeStep)
        index = 0
        x = None if tabu.DEF_TABLES else mpHelper.ForkedData(p.x)
        trialsInfo = None if tabu.DEF_TABLES else p.trialsInfo
        for fold, (trainIndex, testIndex) in enumerate(p.cv):
            pssTrain = pss[:, trainIndex, :]
            pssTest = pss[:, testIndex, :]
            shuffleIndices = None
            if (self.shuffleLabels):
                p.y, shuffleIndices = self.permutateTheLabels(p.y, trainIndex,
                    self.useSmote)
            params.append(Bunch(
                x=x, y=p.y, trainIndex=trainIndex, testIndex=testIndex,
                trialsInfo=trialsInfo, fold=fold, paramsNum=paramsNum,
                pssTrain=mpHelper.ForkedData(pssTrain),
                pssTest=mpHelper.ForkedData(pssTest), freqs=freqs,
                sigSectionMinLengths=p.sigSectionMinLengths,
                sigSectionAlphas=p.sigSectionAlphas, minFreqs=p.minFreqs,
                maxFreqs=p.maxFreqs, index=index,
                onlyMidValueOptions=p.onlyMidValueOptions,
                kernels=p.kernels, Cs=p.Cs, gammas=p.gammas,
                shuffleIndices=shuffleIndices))
            index += 1
        return params

    def parametersGenerator(self, p):
        for hp in itertools.product(*(p.minFreqs, p.maxFreqs,
            p.onlyMidValueOptions, p.sigSectionMinLengths,
            p.sigSectionAlphas)):
                yield self.createParamsObj(hp)

    def createParamsObj(self, paramsTuple):
        (minFreq, maxFreq, onlyMidValue, sigSectionMinLength,
         sigSectionAlpha) = paramsTuple
        return Bunch(minFreq=minFreq, maxFreq=maxFreq,
            onlyMidValue=onlyMidValue, sigSectionMinLength=sigSectionMinLength,
            sigSectionAlpha=sigSectionAlpha)

    def selectorFactory(self, timeStep, p, params=None, maxSurpriseVal=20,
                        doPlotSections=False):
        if (params is None):
            params = p
        return FrequenciesSelector(timeStep, p.sigSectionAlpha,
            p.sigSectionMinLength, params.minFreq, params.maxFreq,
            params.onlyMidValue, maxSurpriseVal,
            self.LABELS[self.procID], doPlotSections)

    def resultItem(self, selector, p, res, params, ytest):
        return Bunch(predResults=res, fold=p.fold, ytest=ytest,
            featureExtractorName=Analyzer.FE_ALL,
            sections=selector.sections, minFreq=params.minFreq,
            maxFreq=params.maxFreq, onlyMidValue=params.onlyMidValue,
            sigSectionMinLength=p.sigSectionMinLength,
            sigSectionAlpha=p.sigSectionAlpha)

    def _predictorParamtersKeyClass(self):
        return namedtuple('predictorParamters', ['sigSectionMinLength',
            'sigSectionAlpha', 'minFreq', 'maxFreq', 'onlyMidValue', 'kernel',
            'C', 'gamma'])

    def predictorParamtersKeyItem(self, res, predRes):
        PredictorParamtersKey = self._predictorParamtersKeyClass()
        return PredictorParamtersKey(
            sigSectionMinLength=res.sigSectionMinLength,
            sigSectionAlpha=res.sigSectionAlpha,
            minFreq=res.minFreq, maxFreq=res.maxFreq,
            onlyMidValue=res.onlyMidValue,
            kernel=predRes.kernel, C=predRes.C, gamma=predRes.gamma)

    def printBestPredictorResults(self, bestEstimator, printSections=True):
        bep = bestEstimator.parameters
        print('Best results: ' + \
            'sigSectionMinLength: {}, '.format(bep.sigSectionMinLength) + \
            'sigSectionAlpha:{}, '.format(bep.sigSectionAlpha,) + \
            'minFreqs: {}, maxFreqs: {}, '.format(bep.minFreq, bep.maxFreq) + \
            'onlyMidValueOptions:{}, '.format(bep.onlyMidValue) + \
            'kernel: {}, c: {}, gamma: {}'.format(
            bep.kernel, bep.C, bep.gamma))
#         threshold = 0.6
#         bestInds = [k for k,r in enumerate(bestEstimator.rates) if (r[0]>threshold and r[1]>threshold)]
#         gmeans = np.array([MLUtils.calcGmean(r[0],r[1]) for r in bestEstimator.rates])
#         scoreArgSortInds = np.argsort(gmeans)[::-1]
#         channels = set(itertools.chain(*[sec.keys() for k,sec in enumerate(bestEstimator.sections) if k in bestInds]))        
#         utils.saveToMatlab({'channels':list(channels)}, '/home/noam/Dropbox/postDocMoshe/MEG/centipede/matlabSource/dor/channels')
#         for sections in bestEstimator.sections:
#         if (printSections):
#             print('Sections:')
#             selector = utils.load(self.selectorName)
#             self.printBestSections(bestEstimator.sections, selector.freqs, bep.onlyMidValue)

    def calcFeaturesSpace(self, X, channels, bep):
        T = X.shape[1]
        timeStep = self.calcTimeStep(T)
        _, pss = freqsUtils.calcAllPS(X,
            timeStep, bep.minFreq, bep.maxFreq, channels)
        return pss

    def featuresAxis(self, selector=None):
        return selector.freqs

    @property
    def featuresAxisLabel(self):
        return 'Frequency (Hz)'

    def plotFreqsPerSensor(self,sensors=None,multipleByWeights=False):
        folder = os.path.join(self.folder,self.subject,'activityFreqsPower' if multipleByWeights else 'sensorsFreqsPower')
        utils.createDirectory(folder)
        X,y,_ = self.getXY(self.STEP_SPLIT_DATA)
        T = X.shape[1]
        timeStep = self.calcTimeStep(T)        
        freqsUtils.plotPSDiff(X, y, timeStep, folder, sensors, self.LABELS[self.procID])
                    
#     def heldOutFeaturesExtraction(self,x,y,trialsInfo,bep,normalizationField='subject'):
#         T = x.shape[1]
#         timeStep = self.calcTimeStep(T)        
#         print('Freqs Selector')                
#         freqsSelector = GSS.FrequenciesSelector(timeStep, bep.sigSectionAlpha, bep.sigSectionMinLength, bep.minFreq, bep.maxFreq, bep.onlyMidValue)
#         xFeatures = freqsSelector.fit_transform(x, y)
#         utils.save(freqsSelector.freqs, self.freqsFileName)
#         utils.save(freqsSelector.sections, self.sectionsFileName)
#         self.printBestSections(freqsSelector.sections, freqsSelector.freqs, bep.onlyMidValue)
#         if (normalizationField in trialsInfo):
#             xFeatures = self.normalizeFeatures(xFeatures, trialsInfo, normalizationField) 
#         return xFeatures, freqsSelector

#     def heldOutFeaturesTransform(self, p, x_heldout, featureExtractorName):
#         xHeldoutFeaturesTimed = p.selector.transform(x_heldout)  
#         xHeldoutFeaturesTimedNormalized,_,_ = MLUtils.normalizeData(xHeldoutFeaturesTimed)    
#         return xHeldoutFeaturesTimedNormalized      

#     def calcSensorsImporatances(self, foldsNum, doCalc=True, doPlot=True):
#         if (doCalc):
#             bestEstimators = utils.load(self.bestEstimatorsFileName)
#             bep = bestEstimators[bestEstimators.keys()[0]].parameters
#             print('load all the data')
#             x,y,trialsInfo = self.getXY(self.STEP_SPLIT_DATA)
#             _,T,C = x.shape
#             print('sections length: {}'.format(C))
#             timeStep = self.calcTimeStep(T)
#             print('fit the selector')
#             selector = self.selectorFactory(timeStep, bep, bep)   
#             # split into folds, in each fold calculates the sensors importance over the test 
#             cv = self.featuresCV(y,trialsInfo,foldsNum)
#             importance = np.zeros((foldsNum,C))
#             for fold, (train_index, test_index) in enumerate(cv):  
#                 xtrain, ytrain = x[train_index], y[train_index]
#                 xtest, ytest = x[test_index], y[test_index]
#                 # fit the selector using the train data
#                 selector.fit(xtrain, ytrain)
#                 # calc the PS of the train and test data
#                 selectorInitObjTrain = selector.initTransform(xtrain)
#                 selectorInitObjTest = selector.initTransform(xtest)
#                 # calc the auc of the test data
#                 totalAUC = self.tranformPredict(xtrain, ytrain, xtest, ytest, selector, bep, selectorInitObjTrain, selectorInitObjTest, doPrint=True)
#                 print('fold {} auc: {}'.format(fold,totalAUC))
#                 # remove each sensor and predict again the test data
#                 sections = selector.sections.copy()
#                 for c in sections.keys():
#                     # Remove the sensorSections from the selector.sections
#                     selector.sections = sections.copy()
#                     selector.sections.pop(c,None)
#                     # transform and predict using the other sections
#                     auc = self.tranformPredict(xtrain, ytrain, xtest, ytest, selector, bep, selectorInitObjTrain, selectorInitObjTest)
# #                         print('sensor {}, sections len: {}, auc: {}'.format(c,len(sensorSections),auc))
#                     importance[fold,c] = 1 - auc/float(totalAUC)
#             utils.save(importance, self.sensorsImportanceFileName)
#         else:
#             importance = utils.load(self.sensorsImportanceFileName)
#         plots.barPlot(np.mean(importance,0), 'Sensors Importance', doShow=doPlot)#, fileName=self.figureFileName('SensorsImportance'))
#         return importance
# 
#     def tranformPredict(self, xtrain, ytrain, xtest, ytest, selector, bep, selectorInitObjTrain, selectorInitObjTest, doPrint=False):
#         xtrainFeatures = selector.transform(xtrain, selectorInitObjTrain)
#         xtestFeatures = selector.transform(xtest, selectorInitObjTrain)
#         xtrainFeaturesBoost, ytrainBoost = MLUtils.boost(xtrainFeatures, ytrain)
#         svc = self.predictor(bep.C, bep.kernel, bep.gamma)
#         svc.fit(xtrainFeaturesBoost,ytrainBoost)
#         xtestProbs = svc.predict(xtestFeatures) 
#         if (doPrint):
#             ytestPred = MLUtils.probsToPreds(xtestProbs)
#             MLUtils.calcConfusionMatrix(ytest, ytestPred, self.LABELS[self.procID],True) 
#             print(sf.calcRates(ytest, ytestPred))
#         return sf.AUCScore(ytest, xtestProbs)
    
    @property
    def selectorName(self):
        return 'FrequenciesSelector.pkl'
