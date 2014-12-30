'''
Created on Dec 11, 2014

@author: noam
'''
from src.commons.analyzerTimeSlidingWindow.analyzerTimeSWFreqsSelector import \
    AnalyzerTimeSWFreqsSelector
from src.commons.sliders.freqsSlider import \
    FreqsWindowSlider
from src.commons.utils import MLUtils
from src.commons.utils import utils
from src.commons.utils import plots
from src.commons.utils import mpHelper
from src.commons.utils import tablesUtils as tabu
from src.commons.utils import sectionsUtils as su

import numpy as np
import scipy.integrate as integrate
from sklearn.datasets.base import Bunch
import operator
from collections import namedtuple
import itertools
import random


class AnalyzerFreqsSWSelector(AnalyzerTimeSWFreqsSelector):

    def _prepareCVParams(self, p):
        p = p.merge(p.kwargs)
        params = []
        windowsGenerator = list(itertools.product(*(
            p.minFreqs, p.maxFreqs, p.windowSizes, p.windowsNums)))
        totalWindowsNum = sum([wn for (_, wn, _, _) in windowsGenerator])
        index = 0
        T = p.x.shape[1]
        print('T is {}'.format(T))
        timeStep = self.calcTimeStep(T)
        cv = list(p.cv)
        paramsNum = len(cv) * totalWindowsNum
        for fold, (trainIndex, testIndex) in enumerate(cv):
            pssTrain, freqs = su.preCalcPS(p.x, min(p.minFreqs),
                max(p.maxFreqs), timeStep, trainIndex)
            pssTest, _ = su.preCalcPS(p.x, min(p.minFreqs),
                max(p.maxFreqs), timeStep, testIndex)
            x = None if tabu.DEF_TABLES else mpHelper.ForkedData(p.x)
            trialsInfo = None if tabu.DEF_TABLES else p.trialsInfo
            if (self.shuffleLabels):
                print('Shuffling the labels')
                random.shuffle(p.y)
            for minFreq, maxFreq, windowSize, windowsNum in windowsGenerator:
                freqsSlider = FreqsWindowSlider(minFreq, maxFreq,
                    windowSize, windowsNum)
                for windowMinFreq, windowMaxFreq in \
                        freqsSlider.windowsGenerator():
                    params.append(Bunch(
                        x=x, y=p.y, trainIndex=trainIndex, testIndex=testIndex,
                        trialsInfo=trialsInfo, fold=fold, paramsNum=paramsNum,
                        pssTrain=mpHelper.ForkedData(pssTrain),
                        pssTest=mpHelper.ForkedData(pssTest), freqs=freqs,
                        windowSize=windowSize, windowsNum=windowsNum,
                        minFreq=windowMinFreq, maxFreq=windowMaxFreq,
                        sigSectionMinLengths=p.sigSectionMinLengths,
                        sigSectionAlphas=p.sigSectionAlphas, index=index,
                        onlyMidValueOptions=p.onlyMidValueOptions,
                        windowSizes=p.windowSizes, windowsNums=p.windowsNums,
                        kernels=p.kernels, Cs=p.Cs, gammas=p.gammas))
                    index += 1
        return params

    def _preparePredictionsParameters(self, ps):
        resultsFileNames = []
        for p in ps:
            t = utils.ticToc()
            resultsFileName, doCalc = self.checkExistingResultsFile(p)
            resultsFileNames.append(resultsFileName)
            if (not doCalc):
                return resultsFileName
            print('{} out of {}'.format(p.index, p.paramsNum))
            x, ytrain, ytest, p.trialsInfo, _ = self._preparePPInit(p)
            pssTrain = p.pssTrain.value
            pssTest = p.pssTest.value
            T = x.shape[1]
            timeStep = self.calcTimeStep(T)
            bestScore = Bunch(auc=0.5, gmean=0.5)
            bestParams = Bunch(auc=None, gmean=None)
            externalParams = Bunch(fold=p.fold, windowSize=p.windowSize,
                windowsNum=p.windowsNum, minFreq=p.minFreq)
            for hp in self.parametersGenerator(p):
                selector = self.selectorFactory(timeStep, hp)
                xtrainFeatures = selector.fit_transform(
                    x, ytrain, p.trainIndex,
                    preCalcPSS=pssTrain, preCalcFreqs=p.freqs)
                xtestFeatures = selector.transform(x, p.testIndex,
                    preCalcPSS=pssTest, preCalcFreqs=p.freqs)
                if (xtrainFeatures.shape[0] > 0 and
                        xtestFeatures.shape[0] > 0):
                    xtrainFeaturesBoost, ytrainBoost = MLUtils.boost(
                        xtrainFeatures, ytrain)
                    self._predict(Bunch(
                        xtrainFeatures=xtrainFeaturesBoost,
                        xtestFeatures=xtestFeatures,
                        ytrain=ytrainBoost, ytest=ytest,
                        kernels=p.kernels, Cs=p.Cs, gammas=p.gammas),
                        bestScore, bestParams, hp)

            utils.save((externalParams, bestScore, bestParams), resultsFileName)
            howMuchTime = utils.howMuchTimeFromTic(t)
            print('finish {}, {}'.format(externalParams, bestScore, howMuchTime))
        return resultsFileNames

    def parametersGenerator(self, p):
        for hp in itertools.product(*([p.minFreq], [p.maxFreq],
                p.onlyMidValueOptions, p.sigSectionMinLengths,
                p.sigSectionAlphas)):
            yield self.createParamsObj(hp)

    def createParamsObj(self, paramsTuple):
        (minFreq, maxFreq, onlyMidValue, sigSectionMinLength,
            sigSectionAlpha) = paramsTuple
        return Bunch(minFreq=minFreq, maxFreq=maxFreq,
            onlyMidValue=onlyMidValue, sigSectionMinLength=sigSectionMinLength,
            sigSectionAlpha=sigSectionAlpha)


    def resultItem(self, selector, p, res, hp, ytest):
        return Bunch(predResults=res, fold=p.fold, ytest=ytest,
            windowSize=hp.windowSize,
            windowsNum=hp.windowsNum,
            sections=selector.sections,
            minFreq=hp.minFreq, maxFreq=hp.maxFreq,
            onlyMidValue=hp.onlyMidValue,
            sigSectionMinLength=p.sigSectionMinLength,
            sigSectionAlpha=p.sigSectionAlpha)

    def _predictorParamtersKeyClass(self):
        return namedtuple('predictorParamters',
            ['sigSectionMinLength', 'sigSectionAlpha',
             'onlyMidValue', 'windowSize', 'windowsNum',
             'kernel', 'C', 'gamma'])

    def predictorParamtersKeyItem(self, res, predRes):
        PredictorParamtersKey = self._predictorParamtersKeyClass()
        return PredictorParamtersKey(
            sigSectionMinLength=res.sigSectionMinLength,
            sigSectionAlpha=res.sigSectionAlpha,
            onlyMidValue=res.onlyMidValue,
            windowSize=res.windowSize, windowsNum=res.windowsNum,
            kernel=predRes.kernel, C=predRes.C, gamma=predRes.gamma)

    def scorerFoldsResultsKey(self, res):
        return res.windowSize

    def scorerFoldsResultsItem(self, score, probsScore, rates, res,
                               predRes, auc, gmean):
        probs = predRes.probs if predRes is not None else []
        return Bunch(score=score, probsScore=probsScore, fold=res.fold,
            rates=rates, sections=res.sections, minFreq=res.minFreq,
            auc=auc, gmean=gmean, y=res.ytest, probs=probs)

    def _bestEstimatorsSortKey(self):
        return operator.itemgetter('minFreq', 'fold')

    def _bestEstimatorsGroupByKey(self):
        return operator.itemgetter('minFreq')

    def analyzeResults(self, freqsSliderRange, doPlot=True,
            printResults=False, plotPerAccFunc=True, doShow=True,
            doSmooth=True, smoothWindowSize=21, smoothOrder=3):
#         self.timeAxis = self.loadTimeAxis()
        allScores = {}
        bestEstimatorsPerWindowAccs = utils.load(
            self.bestEstimatorsPerWindowFileName)
        if (not plotPerAccFunc and doPlot):
            plots.plt.figure()
            labels = []
        for accFunc, bestEstimatorsPerWindow in \
                bestEstimatorsPerWindowAccs.iteritems():
            bestEstimatorsPerWindow = utils.sortDictionaryByKey(
                bestEstimatorsPerWindow)
            scoresGenerator = self.scoresGeneratorPerWindow(
                bestEstimatorsPerWindow, printResults)
            if (plotPerAccFunc and doPlot):
                plots.plt.figure()
                labels = []
            for (scores, scoresStd, _, _, _, windowSize, bep) in scoresGenerator:
                freqsSlider = FreqsWindowSlider(freqsSliderRange[0],
                    freqsSliderRange[1], bep.windowSize, bep.windowsNum)
                xAxis = np.array([np.mean(f) for f in freqsSlider.windowsGenerator()])
                if (doSmooth):
                    scores = MLUtils.savitzkyGolaySmooth(scores,
                        smoothWindowSize, smoothOrder)
                allScores[accFunc] = (scores, scoresStd)
                if (doPlot):
                    ylabel = 'Accuracy ({})'.format(accFunc) \
                        if plotPerAccFunc else 'Accuracy'
                    plots.graph(xAxis, scores, xlabel='Freqs (Hz)',
                        ylabel=ylabel, yerr=scoresStd, doShow=False)
                    acc = integrate.simps(scores, xAxis)
                    acc = acc * self.T / (max(xAxis) - min(xAxis))
                    if (plotPerAccFunc):
                        labels.append('{} Hz ({:.2f})'.format(bep.windowSize, acc))
            if (doPlot):
                if (not plotPerAccFunc):
                    labels.append(accFunc)
                else:
                    self._analyzeResultsPlot(labels, accFunc, windowSize,
                        freqsSliderRange, doShow, plotPerAccFunc)

        if (not plotPerAccFunc and doPlot):
            self._analyzeResultsPlot(labels, accFunc, windowSize,
                freqsSliderRange, doShow,  plotPerAccFunc)

        return allScores, xAxis

    def _analyzeResultsPlot(self, labels, accFunc, windowSize,
            freqsSliderRange, doShow, plotPerAccFunc):
        plots.plt.xlim(freqsSliderRange)
        if (len(labels) > 1):
            legend = plots.plt.legend(labels, bbox_to_anchor=(1.02, 1.03),
                frameon=True, fancybox=True)
            frame = legend.get_frame()
            frame.set_lw(1.5)
        fileName = 'accuracyOverFreq_{}_{}_{}.jpg'.format(self.subject,
            accFunc if plotPerAccFunc else '', windowSize)
        plots.plt.savefig(self.figureFileName(fileName))
        if (doShow):
            plots.plt.show()

    @property
    def selectorName(self):
        return 'FreqsSWSelector'
