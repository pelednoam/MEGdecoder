'''
Created on Dec 15, 2014

@author: noam
'''

from src.commons.analyzerTimeSlidingWindow.analyzerTimeSWFreqsSelector import \
    AnalyzerTimeSWFreqsSelector
from src.commons.sliders.freqsSlider import FreqsWindowSlider
from src.commons.sliders.timeSliders import TimeWindowSlider
from src.commons.utils import MLUtils
from src.commons.utils import utils
from src.commons.utils import plots

import numpy as np
import scipy.integrate as integrate
from sklearn.datasets.base import Bunch
import operator
from collections import namedtuple
import itertools


class AnalyzerFreqsTimeSWSelector(AnalyzerTimeSWFreqsSelector):

    def _preparePredictionsParameters(self, p, overwriteResultsFile=True):
        resultsFileName, doCalc = self.checkExistingResultsFile(p)
        if (not doCalc):
            return resultsFileName
        x, ytrain, ytest, p.trialsInfo, _ = self._preparePPInit(p)
        results = []
        print('{} out of {}'.format(p.index, p.paramsNum))
        T = x.shape[1]
        timeStep = self.calcTimeStep(T)

        for hp in self.parametersGenerator(p):
            hp = self.createParamsObj(hp)
            timeSlider = TimeWindowSlider(0, hp.timeWindowSize,
                hp.timeWindowsNum, T)
            for timeSlider.startIndex in timeSlider.windowsGenerator():
                hp.startIndex = timeSlider.startIndex
                xtrainTimedIndices = timeSlider.fit_transform(
                    x, p.trainIndex, returnIndices=True)
                xtestTimedIndices = timeSlider.transform(
                    x, p.testIndex, returnIndices=True)
                freqsSlider = FreqsWindowSlider(hp.minFreq,
                    hp.maxFreq, hp.freqsWindowSize, hp.freqsWindowsNum)
                for hp.minFreq, hp.maxFreq in \
                        freqsSlider.windowsGenerator():
                    selector = self.selectorFactory(timeStep, p, hp)
                    xtrainTimedFeatures = selector.fit_transform(
                        x, ytrain, p.trainIndex, xtrainTimedIndices)
                    xtestTimedFeatures = selector.transform(x, p.testIndex,
                        xtestTimedIndices)
                    if (xtrainTimedFeatures.shape[0] == 0 or
                        xtestTimedFeatures.shape[0] == 0):
                        res = None
                    else:
                        xtrainTimedFeaturesBoost, ytrainBoost = MLUtils.boost(
                            xtrainTimedFeatures, ytrain)
                        res = self._predict(Bunch(
                            xtrainFeatures=xtrainTimedFeaturesBoost,
                            xtestFeatures=xtestTimedFeatures,
                            ytrain=ytrainBoost, kernels=p.kernels,
                            Cs=p.Cs, gammas=p.gammas))
                    results.append(self.resultItem(
                        selector, p, res, hp, ytest))

        utils.save(results, resultsFileName)
        return resultsFileName

    def parametersGenerator(self, p):
        return itertools.product(*(p.minFreqs, p.maxFreqs,
            p.onlyMidValueOptions, p.freqsWindowSizes, p.freqsWindowsNums,
            p.timeWindowSizes, p.timeWindowsNums))

    def createParamsObj(self, paramsTuple):
        (minFreq, maxFreq, onlyMidValue, freqsWindowSizes, freqsWindowsNums,
            timeWindowSizes, timeWindowsNums) = paramsTuple
        return Bunch(minFreq=minFreq, maxFreq=maxFreq,
            onlyMidValue=onlyMidValue, freqsWindowSizes=freqsWindowSizes,
            freqsWindowsNums=freqsWindowsNums, timeWindowSizes=timeWindowSizes,
            timeWindowsNums=timeWindowsNums)

    def resultItem(self, selector, p, res, hp, ytest):
        return Bunch(predResults=res, fold=p.fold, ytest=ytest,
            freqsWindowSize=hp.freqsWindowSize,
            freqsWindowsNum=hp.freqsWindowsNum,
            timeWindowSize=hp.timeWindowSize,
            timeWindowsNum=hp.timeWindowsNum,
            sections=selector.sections,
            startIndex=hp.startIndex,
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
        return (res.windowSize, res.minFreq)

    def scorerFoldsResultsItem(self, score, probsScore, rates, res,
                               predRes, auc, gmean):
        probs = predRes.probs if predRes is not None else []
        return Bunch(score=score, probsScore=probsScore, fold=res.fold,
            rates=rates, sections=res.sections, minFreq=res.minFreq,
            startIndex=res.startIndex,
            auc=auc, gmean=gmean, y=res.ytest, probs=probs)

    # todo: add start_index
    def _bestEstimatorsSortKey(self):
        return operator.itemgetter('minFreq', 'fold')

    def _bestEstimatorsGroupByKey(self):
        return operator.itemgetter('minFreq')

    def analyzeResults(self, freqsSliderRange, doPlot=True,
            printResults=False, plotPerAccFunc=True,
            doSmooth=True, smoothWindowSize=21, smoothOrder=3):
#         self.timeAxis = self.loadTimeAxis()
        bestEstimatorsPerWindowAccs = utils.load(
            self.bestEstimatorsPerWindowFileName)
        if (not plotPerAccFunc):
            plots.plt.figure()
            labels = []
        for accFunc, bestEstimatorsPerWindow in \
                bestEstimatorsPerWindowAccs.iteritems():
            bestEstimatorsPerWindow = utils.sortDictionaryByKey(
                bestEstimatorsPerWindow)
            scoresGenerator = self.scoresGeneratorPerWindow(
                bestEstimatorsPerWindow, printResults)
            if (plotPerAccFunc):
                plots.plt.figure()
                labels = []
            for (scores, scoresStd, _, _, _, windowSize, bep) in scoresGenerator:
                if (windowSize == 500):
                    print('time! not freqs!')
                    break
                freqsSlider = FreqsWindowSlider(freqsSliderRange[0],
                    freqsSliderRange[1], bep.windowSize, bep.windowsNum)
                xAxis = np.array([np.mean(f) for f in freqsSlider.windowsGenerator()])
#                 scores = MLUtils.savitzkyGolaySmooth(scores, smoothWindowSize,
#                     smoothOrder)
                ylabel = 'Accuracy ({})'.format(accFunc) \
                    if plotPerAccFunc else 'Accuracy'
                plots.graph(xAxis, scores, xlabel='Freqs (Hz)',
                    ylabel=ylabel, yerr=scoresStd, doShow=False)
                acc = integrate.simps(scores, xAxis)
                acc = acc * self.T / (max(xAxis) - min(xAxis))
                if (plotPerAccFunc):
                    labels.append('{} Hz ({:.2f})'.format(bep.windowSize, acc))
            if (not plotPerAccFunc):
                labels.append(accFunc)
            else:
                self._analyzeResultsPlot(labels, accFunc, windowSize,
                    freqsSliderRange, doPlot, plotPerAccFunc)

        if (not plotPerAccFunc):
            self._analyzeResultsPlot(labels, accFunc, windowSize,
                freqsSliderRange, doPlot,  plotPerAccFunc)

    def _analyzeResultsPlot(self, labels, accFunc, windowSize,
            freqsSliderRange, doPlot, plotPerAccFunc):
        plots.plt.xlim(freqsSliderRange)
        if (len(labels) > 1):
            legend = plots.plt.legend(labels, bbox_to_anchor=(1.02, 1.03),
                frameon=True, fancybox=True)
            frame = legend.get_frame()
            frame.set_lw(1.5)
        fileName = 'accuracyOverFreq_{}_{}_{}.jpg'.format(self.subject,
            accFunc if plotPerAccFunc else '', windowSize)
        plots.plt.savefig(self.figureFileName(fileName))
        if (doPlot):
            plots.plt.show()

    @property
    def selectorName(self):
        return 'FreqsTimeSWSelector'
