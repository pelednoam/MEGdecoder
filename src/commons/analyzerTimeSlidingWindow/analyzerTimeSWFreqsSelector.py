'''
Created on Nov 12, 2014

@author: noampeled
'''

from src.commons.analyzer.analyzer import Analyzer
from src.commons.analyzerTimeSlidingWindow.analyzerTimeSWSelector import \
    AnalyzerTimeSWSelector
from src.commons.sliders.timeSliders import TimeWindowSlider
from src.commons.selectors.frequenciesSelector import FrequenciesSelector
from src.commons.utils import mpHelper
from src.commons.utils import tablesUtils as tabu
from src.commons.utils import sectionsUtils as su
from src.commons.utils import freqsUtils as fu

from sklearn.datasets.base import Bunch
import itertools
from collections import namedtuple


class AnalyzerTimeSWFreqsSelector(AnalyzerTimeSWSelector):

    def _prepareCVParams(self, p):
        p = p.merge(p.kwargs)
        params = []
        windowsGenerator = list(itertools.product(*(
            p.windowSizes, p.windowsNums)))
        totalWindowsNum = sum([wn for (_, wn) in windowsGenerator])
        index = 0
        T = p.x.shape[1]
        print('T is {}'.format(T))
        timeStep = self.calcTimeStep(T)
        cv = list(p.cv)
        paramsNum = len(cv) * totalWindowsNum
        for windowSize, windowsNum in windowsGenerator:
            timeSlider = TimeWindowSlider(0, windowSize, windowsNum, T)
            for timeSlider.startIndex in timeSlider.windowsGenerator():
                timeIndices = timeSlider.transform(p.x, returnIndices=True)
                pss, freqs = fu.calcPSX(p.x, min(p.minFreqs),
                    max(p.maxFreqs), timeStep, timeIndices=timeIndices)
                for fold, (trainIndex, testIndex) in enumerate(cv):
                    pssTrain = pss[:, trainIndex, :]
                    pssTest = pss[:, testIndex, :]
                    y, shuffleIndices = self.permutateTheLabels(p.y, trainIndex,
                        self.useSmote) if self.shuffleLabels else (p.y, None)
                    params.append(Bunch(
                        y=y, trainIndex=trainIndex, testIndex=testIndex,
                        fold=fold, paramsNum=paramsNum,
                        pssTrain=mpHelper.ForkedData(pssTrain),
                        pssTest=mpHelper.ForkedData(pssTest), freqs=freqs,
                        windowSize=windowSize, windowsNum=windowsNum,
                        startIndex=timeSlider.startIndex, index=index,
                        sigSectionMinLengths=p.sigSectionMinLengths,
                        sigSectionAlphas=p.sigSectionAlphas,
                        minFreqs=p.minFreqs, maxFreqs=p.maxFreqs,
                        onlyMidValueOptions=p.onlyMidValueOptions,
                        windowSizes=p.windowSizes, windowsNums=p.windowsNums,
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

    def resultItem(self, selector, p, res, hp, ytest):
        return Bunch(predResults=res, fold=p.fold, ytest=ytest,
            featureExtractorName=Analyzer.FE_ALL,
            windowSize=hp.windowSize,
            windowsNum=hp.windowsNum,
            sections=selector.sections,
            startIndex=hp.startIndex,
            minFreq=hp.minFreq, maxFreq=hp.maxFreq,
            onlyMidValue=hp.onlyMidValue,
            sigSectionMinLength=p.sigSectionMinLength,
            sigSectionAlpha=p.sigSectionAlpha)

    def printBestPredictorResults(self, bestEstimator):
        bep = bestEstimator.parameters
        print('Best results: ' +
            'sigSectionMinLength: {}, '.format(bep.sigSectionMinLength) +
            'sigSectionAlpha:{}, '.format(bep.sigSectionAlpha) +
            'minFreqs: {}, maxFreqs: {}, '.format(bep.minFreq, bep.maxFreq) +
            'onlyMidValueOptions:{}, '.format(bep.onlyMidValue) +
            'windowSize:{}, '.format(bep.windowSize) +
            'windowsNum:{}, '.format(bep.windowsNum) +
            'kernel: {}, c: {}, gamma: {}'.format(
            bep.kernel, bep.C, bep.gamma))

    def _predictorParamtersKeyClass(self):
        predictorParamters = namedtuple('predictorParamters',
            ['sigSectionMinLength', 'sigSectionAlpha', 'minFreq', 'maxFreq',
             'onlyMidValue', 'windowSize', 'windowsNum',
             'kernel', 'C', 'gamma'])
        # Otherwise the pickle doesn't work
        # http://stackoverflow.com/questions/16377215/how-to-pickle-a-namedtuple-instance-correctly
        globals()[predictorParamters.__name__] = predictorParamters
        return predictorParamters

    def predictorParamtersKeyItem(self, res, predRes):
        PredictorParamtersKey = self.predictorParamtersKeyClass
        return PredictorParamtersKey(
            sigSectionMinLength=res.sigSectionMinLength,
            sigSectionAlpha=res.sigSectionAlpha,
            minFreq=res.minFreq, maxFreq=res.maxFreq,
            onlyMidValue=res.onlyMidValue,
            windowSize=res.windowSize, windowsNum=res.windowsNum,
            kernel=predRes.kernel, C=predRes.C, gamma=predRes.gamma)
