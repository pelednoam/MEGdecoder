'''
Created on Aug 17, 2014

@author: noampeled
'''
from src.commons.analyzer.analyzerSelector import AnalyzerSelector
from src.commons.sliders.timeSliders import TimeWindowSlider
from src.commons.utils import utils
from src.commons.utils import MLUtils
from src.commons.utils import plots

from collections import defaultdict
from sklearn.datasets.base import Bunch
import operator
import numpy as np
import scipy.integrate as integrate
import traceback


class AnalyzerTimeSWSelector(AnalyzerSelector):

    def _preparePredictionsParameters(self, ps):
        resultsFileNames = []
        for p in ps:
            t = utils.ticToc()
            resultsFileName, doCalc = self.checkExistingResultsFile(p)
            resultsFileNames.append(resultsFileName)
            if (not doCalc):
                continue
            x, ytrain, ytest, p.trialsInfo, _ = self._preparePPInit(p)
            print('{} out of {}'.format(p.index, p.paramsNum))
            T = x.shape[1]
            timeStep = self.calcTimeStep(T)
            bestScore = Bunch(auc=0.5, gmean=0.5)
            bestParams = Bunch(auc=None, gmean=None)
            timeSlider = TimeWindowSlider(0, p.windowSize, p.windowsNum, T)
            timeSlider.startIndex = p.startIndex
            externalParams = Bunch(fold=p.fold, windowSize=p.windowSize,
                windowsNum=p.windowsNum, startIndex=timeSlider.startIndex)
            for hp in self.parametersGenerator(p):
                xtrainTimedIndices = timeSlider.fit_transform(
                    x, p.trainIndex, returnIndices=True)
                xtestTimedIndices = timeSlider.transform(
                    x, p.testIndex, returnIndices=True)
                selector = self.selectorFactory(timeStep, hp)
                xtrainTimedFeatures = selector.fit_transform(
                    x, ytrain, p.trainIndex, xtrainTimedIndices)
                xtestTimedFeatures = selector.transform(x, p.testIndex,
                    xtestTimedIndices)
                if (xtrainTimedFeatures.shape[0] > 0 and
                        xtestTimedFeatures.shape[0] > 0):
                    if (self.useSmote):
                        xtrainTimedFeaturesBoost, ytrainBoost = \
                            MLUtils.boostSmote(xtrainTimedFeatures, ytrain)
                        if (self.shuffleLabels):
                            ytrainBoost = ytrainBoost[p.shuffleIndices]
                    else:
                        xtrainTimedFeaturesBoost, ytrainBoost = \
                            MLUtils.boost(xtrainTimedFeatures, ytrain)
                    self._predict(Bunch(
                        xtrainFeatures=xtrainTimedFeaturesBoost,
                        xtestFeatures=xtestTimedFeatures,
                        ytrain=ytrainBoost, ytest=ytest,
                        kernels=p.kernels, Cs=p.Cs, gammas=p.gammas),
                        bestScore, bestParams, hp)

            utils.save((externalParams, bestScore, bestParams), resultsFileName)
            howMuchTime = utils.howMuchTimeFromTic(t)
            print('finish {}, {}'.format(externalParams, bestScore, howMuchTime))
        return resultsFileNames

    def getBestEstimators(self, getRemoteFiles=False):
        print('loading all the results files')
        results = {}
        results['gmean'] = defaultdict(dict)
        results['auc'] = defaultdict(dict)
        print('calculate prediction scores')
        for fileName in self.predictorResultsGenerator(getRemoteFiles):
            externalParams, bestScore, bestParams = utils.load(
                fileName, overwrite=getRemoteFiles)
            key1, key2 = self.resultsKeys(externalParams)
            if (key2 not in results['gmean'][key1]):
                results['gmean'][key1][key2] = {'scores': [], 'params': []}
                results['auc'][key1][key2] = {'scores': [], 'params': []}
            # Fix: If the best score wasn't initialized to 0, and there were
            # no results, set it to chance level
            if (bestParams.gmean is None):
                bestScore.auc = 0.5
                bestScore.gmean = 0.5
            results['gmean'][key1][key2]['scores'].append(bestScore.gmean)
            results['gmean'][key1][key2]['params'].append(bestParams.gmean)
            results['auc'][key1][key2]['scores'].append(bestScore.auc)
            results['auc'][key1][key2]['params'].append(bestParams.auc)
        print('results for {}'.format(self.defaultFileNameBase))
        for acc in ['auc', 'gmean']:
            for key1, res in results[acc].iteritems():
                print('{} key1: {}, {} keys'.format(acc, key1, len(res)))
                lengths = [len(val['scores']) for key2, val in res.iteritems()]
                print('#results: {}'.format(lengths))
        utils.save(results, self.bestEstimatorsPerWindowFileName)

    def scorerFoldsResultsKey(self, res):
        return res.windowSize

    def resultsKeys(self, p):
        return p.windowSize, p.startIndex

    def scorerFoldsResultsItem(self, score, probsScore, rates, res,
                               predRes, auc, gmean):
        probs = predRes.probs if predRes is not None else []
        return Bunch(score=score, probsScore=probsScore, fold=res.fold,
            rates=rates, sections=res.sections, startIndex=res.startIndex,
            auc=auc, gmean=gmean, y=res.ytest, probs=probs)

#     def _findBestEstimators(self, scorerFoldsResults=None):
#         if (scorerFoldsResults is None):
#             scorerFoldsResults = utils.load(self.scorerFoldsResultsFileName)
#         bestEstimatorsPerWindow = Bunch(auc=defaultdict(dict),
#             gmean=defaultdict(dict))
#         # loop over the different window sizes
#         for resultsKey, scorerFoldsResultsDic in \
#                 scorerFoldsResults.iteritems():
#             bestScorePerWindow = Bunch(auc=defaultdict(int),
#                 gmean=defaultdict(int))
#             for predictorParamters, predictorResults in \
#                     scorerFoldsResultsDic.iteritems():
#                 # sort according to _bestEstimatorsSortKey
#                 results = sorted(predictorResults,
#                     key=self._bestEstimatorsSortKey())
#                 # group by groupByKey
#                 resultsByFold = itertools.groupby(results,
#                     key=self._bestEstimatorsGroupByKey())
#                 for groupByKey, resultsPerStartIndex in resultsByFold:
#                     scoresPerWindow = Bunch(auc=[], gmean=[])
#                     probsScoresPerWindow, sectionsPerWindow = [], []
#                     sectionsDicsPerWindow = []
#                     for res in resultsPerStartIndex:
#                         sectionsPerWindow.append(res.sections.keys())
#                         if ('sectionsDic' in res):
#                             sectionsDicsPerWindow.append(res.sectionsDic)
#                         scoresPerWindow.auc.append(res.auc)
#                         scoresPerWindow.gmean.append(res.gmean)
# #                     print(groupByKey, len(scoresPerWindow.auc))
#                     # Find the best estimator per window
#                     for acc in ['auc', 'gmean']:
# #                         scoresPerWindow[acc] = utils.removeNone(scoresPerWindow[acc])
# #                         scoresPerWindow[acc] = utils.replaceNone(scoresPerWindow[acc], 0.5)
#                         if (len(scoresPerWindow[acc]) > 0 and np.mean(scoresPerWindow[acc]) >
#                                 bestScorePerWindow[acc][groupByKey]):
#                             bestEstimatorsPerWindow[acc][resultsKey][groupByKey] = \
#                                 Bunch(parameters=predictorParamters,
#                                 probsScoresPerWindow=probsScoresPerWindow,
#                                 scoresPerWindow=scoresPerWindow[acc],
#                                 sectionsPerWindow=sectionsPerWindow,
#                                 sectionsDicsPerWindow=sectionsDicsPerWindow)
#                             bestScorePerWindow[acc][groupByKey] = \
#                                 np.mean(scoresPerWindow[acc])
#
# #         for resultsKey in scorerFoldsResults.keys():
# #             # The namedtuple PredictorParamters isn't pickleable,
# #             # so convert it to Bunch
# #             for acc in ['auc', 'gmean']:
# #                 bestEstimatorsPerWindow[acc][resultsKey] = \
# #                     utils.sortDictionaryByKey(bestEstimatorsPerWindow[acc][resultsKey])
# #                 for groupByKey in bestEstimatorsPerWindow[acc][resultsKey].keys():
# #                     bestEstimatorsPerWindow[acc][resultsKey][groupByKey].parameters = \
# #                         utils.namedtupleToBunch(
# #                         bestEstimatorsPerWindow[acc][resultsKey][groupByKey].parameters)
#         utils.save(bestEstimatorsPerWindow,
#                    self.bestEstimatorsPerWindowFileName)

    def _bestEstimatorsSortKey(self):
        return operator.itemgetter('startIndex', 'fold')

    def _bestEstimatorsGroupByKey(self):
        return operator.itemgetter('startIndex')

    def analyzeResults(self, calcProbs=False, probsThreshold=0.5, doShow=True,
            doPlot=True, printResults=False, windowSizes=None, plotPerAccFunc=True,
            doSmooth=False, smoothWindowSize=21, smoothOrder=3):
        self.timeAxis = self.loadTimeAxis()
        bestEstimatorsPerWindowAccs = utils.load(
            self.bestEstimatorsPerWindowFileName)
        allScores, allStds = {}, {}
        labels = []
        for accFunc, bestEstimatorsPerWindow in bestEstimatorsPerWindowAccs.iteritems():
            scoresGenerator = self.scoresGeneratorPerWindow(
                bestEstimatorsPerWindow, printResults=printResults)
            for (scores, scoresStd, _, _, windowSize, bep) in scoresGenerator:
                if (windowSizes is not None and windowSize not in windowSizes):
                    continue
                windowsNum = scores.shape[0]
                timeSelector = TimeWindowSlider(0, windowSize, windowsNum,
                    len(self.timeAxis))
                startIndices = np.array(timeSelector.windowsGenerator())
                xAxis = self.timeAxis[startIndices + windowSize / 2]
                allScores[accFunc] = scores
                allStds[accFunc] = scoresStd
                if (doPlot):
                    acc = integrate.simps(scores, xAxis)
                    acc = acc * self.T / (max(xAxis) - min(xAxis))
                    labels.append('{}, {} ms ({:.2f})'.format(
                        accFunc, windowSize, acc))

        if (doPlot):
            plots.graphN(xAxis, [s for s in allScores.values()], labels,
                [s for s in allStds.values()], 'Time (ms)',
                'Accuracy', doSmooth=doSmooth, title=self.defaultFileNameBase,
                smoothWindowSize=smoothWindowSize, smoothOrder=smoothOrder)

        return allScores, xAxis

    def _analyzeResultsPlot(self, labels, accFunc, windowSize, doShow,
            plotPerAccFunc):
        plots.plt.xlim([0, self.T])
        if (len(labels) > 1):
            legend = plots.plt.legend(labels, bbox_to_anchor=(1.02, 1.03),
                frameon=True, fancybox=True)
            frame = legend.get_frame()
            frame.set_lw(1.5)
        fileName = 'accuracyOverTime_{}_{}_{}.jpg'.format(self.subject,
            accFunc if plotPerAccFunc else '', windowSize)
        plots.plt.savefig(self.figureFileName(fileName))
        if (doShow):
            plots.plt.show()

    def findSignificantResults(self, foldsNum=0, doShow=True, windowSizes=None,
            doPlot=True, printResults=False, overwrite=False):
        self.timeAxis = self.loadTimeAxis()
        self.shuffleLabels = True
        bestEstimatorsShuffle = utils.load(
            self.bestEstimatorsPerWindowFileName, overwrite=overwrite)
        self.shuffleLabels = False
        bestEstimators = utils.load(
            self.bestEstimatorsPerWindowFileName, overwrite=overwrite)
        allScores, allPs = {}, {}
        for (accFunc, bestEstimatorPerWindow), \
                (_, bestEstimatorShufflePerWindow) in \
                zip(bestEstimators.iteritems(),
                    bestEstimatorsShuffle.iteritems()):
            scoresGenerator = self.scoresGeneratorPerWindow(
                bestEstimatorPerWindow, printResults=printResults, meanScores=False,
                estimatorKeys=windowSizes)
            scoresGeneratorShuffle = self.scoresGeneratorPerWindow(
                bestEstimatorShufflePerWindow, printResults=printResults, meanScores=False,
                estimatorKeys=windowSizes)

            for (scores, _, _, _, windowSize, bep), \
                (scoresShuffle, _, _, _, _, _) in \
                    zip(scoresGenerator, scoresGeneratorShuffle):
                print('window size {}'.format(windowSize))
                windowsNum = scores.shape[0] 
                timeSelector = TimeWindowSlider(0, windowSize, windowsNum,
                    len(self.timeAxis))
                startIndices = np.array(timeSelector.windowsGenerator())
                xAxis = self.timeAxis[startIndices + windowSize / 2]
                ps = []
                print(scores.shape, scoresShuffle.shape)
                for windowScores, windowScoresShuffle in zip(scores, scoresShuffle):
                    ps.append(utils.ttestGreaterThan(windowScores, windowScoresShuffle))

#                 scores = MLUtils.savitzkyGolaySmooth(scores, smoothWindowSize,
#                     smoothOrder)
                ps = np.array(ps)
                allScores[accFunc] = scores
                allPs[accFunc] = ps
                if (not np.all([len(s) for s in scores] == foldsNum) and foldsNum != 0):
                    print('Not all the result from all the folds were collected!')
                # In case we don't have the results from all the folds
                scoresMean, scoresStd = [np.mean(s) for s in scores], \
                    [np.std(s) for s in scores]
                scoresShuffleMean, scoresShuffleStd = [np.mean(s) for s in
                    scoresShuffle], [np.std(s) for s in scoresShuffle]

                if (doPlot):
                    plots.graph2(xAxis, scoresMean, scoresShuffleMean,
                        yerrs=[scoresStd, scoresShuffleStd],
                        xlabel='Time (sec)', ylabel='Accuracy', title=accFunc,
                        labels=['scores', 'shuffle'], doPlot=True)
                    plots.graph2(xAxis, ps, np.ones(ps.shape) * 0.05, xlabel='Time (sec)',
                        ylabel='Significance', markers=('b-', 'r--'), doPlot=True)

#         if (doPlot):
#             self._analyzeResultsPlot(labels, accFunc,
#                 windowSize, doShow, False)

        return allPs, xAxis

    def scoresGeneratorPerWindow(self, bestEstimatorsPerWindow, estimatorKeys=None,
            scoresAsDict=False, printResults=False, meanScores=True):
        for estimatorKey, bestEstimatorPerWindow in \
                bestEstimatorsPerWindow.iteritems():
            if (estimatorKeys is not None and estimatorKey not in estimatorKeys):
                continue
            parameters = []
            scores, scoresStd, sections, sectionsDics = {}, {}, {}, {}
            for groupByKey, bestEstimator in \
                    bestEstimatorPerWindow.iteritems():
#                 if ('probsScores' in bestEstimator.keys()):
#                     probsScores = utils.arrAppend(
#                         probsScores, bestEstimator.probsScores)
                scores[groupByKey] = np.mean(bestEstimator['scores']) \
                    if meanScores else bestEstimator['scores']
                scoresStd[groupByKey] = np.std(bestEstimator['scores'])
                parameters.append(bestEstimator['params'])
                # insert the sections union from all folds
#                 groupBySections, groupBySectionsDics = set(), {}
#                 for sectionsPerWindow, sectionDicsPerWindow in zip(
#                         bestEstimator.sectionsPerWindow,
#                         bestEstimator.sectionsDicsPerWindow):
#                     groupBySections |= set(sectionsPerWindow)
#                     utils.mergeDics(groupBySectionsDics, sectionDicsPerWindow)
#                 sections[groupByKey] = groupBySections
#                 sectionsDics[groupByKey] = groupBySectionsDics
#                 if (printResults):
#                     self.printBestPredictorResults(bestEstimator)

            if (not scoresAsDict):
                scores = utils.sortDictionaryByKey(scores)
                scoresStd = utils.sortDictionaryByKey(scoresStd)
                scores = np.array([s for s in scores.values()])
                scoresStd = np.array([s for s in scoresStd.values()])
            yield (scores, scoresStd, sections, sectionsDics,
                   estimatorKey, parameters)

    def scoresGenerator(self, bestEstimators, predParams, timeAxis,calcProbs=False,probsThreshold=0.5,printResults=True):
        for windowSize,bestEstimator in bestEstimators.iteritems():
            print('windowSize: {}'.format(windowSize))
            bep = bestEstimator.parameters
            if (printResults):
                self.printBestPredictorResults(bestEstimator)

            timeSelector = TimeWindowSlider(0, bep.windowSize,bep.windowsNum,len(timeAxis))
            startIndices = np.array(timeSelector.windowsGenerator())
            startIndices = np.array(np.linspace(0,timeSelector.T-windowSize,len(bestEstimator.scores)).astype(int))
            xAxis = timeAxis[startIndices+bep.windowSize/2]
            if (calcProbs):
                W = bestEstimator.probsScores.shape[0]
                probsScores = np.reshape(bestEstimator.probsScores,(W,-1))
                probsScores=probsScores[:,np.max(probsScores,0)>probsThreshold]
                yield (xAxis,bestEstimator.scores,probsScores)
            else:
                print(bestEstimator.parameters.windowSize,len(bestEstimator.scores),len(startIndices))
                yield (xAxis,bestEstimator.scores,None)

    def calcImporatances(self, foldsNum, windowSize, windowsNum, testSize=None,
                         ratesThreshold=None, doCalc=True, doShow=False,
                         removeSections=True, permutationNum=2000,
                         permutationLen=10, normalizeUsingWinAcc=False,
                         doSmooth=True, smoothWindowSize=21, smoothOrder=3):
        if (doCalc):
            bestEstimatorsPerWindow = utils.load(
                self.bestEstimatorsPerWindowFileName)
            bestEstimatorPerWindow = bestEstimatorsPerWindow[windowSize]
            print('load all the data')
            x, y, trialsInfo = self.getXY(self.STEP_SPLIT_DATA)
            _, T, C = x.shape
            timeStep = self.calcTimeStep(T)
            importanceAUC, importanceGmean = [], []
            windowsACUs, windowsGmeans = [], []
            for windowNum, (startIndex, bestEstimator) in enumerate(
                bestEstimatorPerWindow.iteritems()):
                try:
                    bep = bestEstimator.parameters
                    timeSelector = TimeWindowSlider(startIndex, windowSize,
                        windowsNum, T)
                    print('Calculating windowImportance for ' + \
                        'windowNum: {} '.format(windowNum + 1) + \
                        'out of {} using the estimator:'.format(
                        timeSelector.windowsNum))
                    self.printBestPredictorResults(bestEstimator)
                    continue
                    xTimed = timeSelector.transform(x)
                    (windowImportanceAUC, windowImportanceGmean, windowAUCs,
                        windowGmeans) = super(AnalyzerTimeSWSelector,
                        self).calcImporatances(
                        foldsNum, testSize, xTimed, y, trialsInfo, bep,
                        timeStep, removeSections, ratesThreshold,
                        permutationLen, permutationNum, doShow=False,
                        doSave=False, doCalc=doCalc, doPlotSections=False)
                    importanceAUC.append(windowImportanceAUC)
                    importanceGmean.append(windowImportanceGmean)
                    windowsACUs.append(windowAUCs)
                    windowsGmeans.append(windowGmeans)
                except:
                    importanceAUC.append(np.zeros((C)))
                    importanceGmean.append(np.zeros((C)))
                    windowsACUs.append(None)
                    windowsGmeans.append(None)
                    print('error with window {}!'.format(windowNum))
                    print traceback.format_exc()
                    utils.dump((windowNum, (startIndex, bestEstimator)),
                        'SWcalcImporatances')

            utils.save((importanceAUC, importanceGmean, windowsACUs,
                windowsGmeans, T), self.sensorsImportanceFileName)
        else:
            doSmooth = False
            importanceAUC, importanceGmean, windowsACUs, windowsGmeans, T = \
                utils.load(self.sensorsImportanceFileName)
            importanceAUC = self.normalizeImportance(importanceAUC,
                windowsACUs, foldsNum, normalizeUsingWinAcc,
                doSmooth, smoothWindowSize, smoothOrder)
            importanceGmean = self.normalizeImportance(importanceGmean,
                windowsGmeans, foldsNum, normalizeUsingWinAcc,
                doSmooth, smoothWindowSize, smoothOrder)
            timeSelector = TimeWindowSlider(0, windowSize, windowsNum, T)
            startIndices = np.array(timeSelector.windowsGenerator())
            timeAxis = self.timeAxis[startIndices + windowSize / 2]

            winImpsAUC = [np.mean(windAcc) for windAcc in windowsACUs]
            winImpsGmean = [np.mean(windAcc) for windAcc in windowsGmeans]
            winImpsAUCStd = [np.std(windAcc) for windAcc in windowsACUs]
            winImpsGmeanStd = [np.std(windAcc) for windAcc in windowsGmeans]
            winImpsAUC = MLUtils.savitzkyGolaySmooth(winImpsAUC,
                smoothWindowSize, smoothOrder)
            winImpsGmean = MLUtils.savitzkyGolaySmooth(winImpsGmean,
                smoothWindowSize, smoothOrder)
            plots.graph2(timeAxis, winImpsAUC, winImpsGmean, ['AUC', 'Gmean'],
                yerrs=[winImpsAUCStd, winImpsGmeanStd], xlabel='Time (ms)',
                ylabel='Accuracy', ylim=[0, 1])
            plots.graphN(timeAxis, importanceAUC.T)

            utils.saveToMatlab(timeAxis, self.timeAxisFileName[:-4],
                'timeAxis')
            utils.saveDictToMatlab(self.sensorsImportanceFileName[:-4],
                {'importanceAUC': importanceAUC,
                 'importanceGmean': importanceGmean})

            self.animateImportance(importanceAUC, timeAxis)

    def normalizeImportance(self, importance, accuracy, foldsNum,
                            normalizeUsingWinAcc=False, doSmooth=True,
                            smoothWindowSize=21, smoothOrder=3):
        C = len(importance[0][0])  # number of channels
        W = len(importance)  # number of windows
        nomalizedImp = np.zeros((W, C))
        # loop over all the windows importance and accuracy
        for w, (windImp, windAcc) in enumerate(zip(importance, accuracy)):
            # Append the mean for each fold: accuracy - channel importance
            if (normalizeUsingWinAcc):
                # tile the window accuracy to fit the window
                # importance shape: Folds x Channels
                windAcc = np.tile(np.array(windAcc), (C, 1)).T
                windImp = np.array(windImp) * windAcc
                windImp = np.vstack((windImp, np.zeros((foldsNum -
                    windImp.shape[0], C))))
                nomalizedImp[w, :] = windImp.mean(0)
            else:
                nomalizedImp[w, :] = np.mean(np.array(windImp), 0)
        # loop over each channel to smooth the data
        if (doSmooth):
            for c in range(C):
                nomalizedImp[:, c] = MLUtils.savitzkyGolaySmooth(
                    nomalizedImp[:, c], smoothWindowSize, smoothOrder)
        return nomalizedImp

    def animateImportance(self, importance, timeAxis):
        dataGenerator = self._animateImportanceDataGenerator(importance, timeAxis)
        maxImp = np.max(importance)
        plots.runAnimatedBar(dataGenerator, dtTimer=100, title='', ylim=[0,maxImp], startsWithZeroZero=True)        

    def _animateImportanceDataGenerator(self,importance,timeAxis):
        c = len(importance[0])
        def dataGenerator():
            for windowImportance,t in zip(importance,timeAxis):
                if (windowImportance is None):
                    yield np.zeros((c)),t
                else:
                    yield windowImportance,t
            yield None,None
        return dataGenerator()

    def bestScore(self, scores, isProbs=False):
        return np.max(scores)

    @property
    def selectorName(self):
        return 'TimeSWSelector'

    @property
    def sensorsImportanceFileName(self):
        return '{}_sensorsImportanceSlidingWindows.pkl'.format(self.dataFileName(self.STEP_FEATURES)[:-4])
#         try:
#             if (isProbs):
#                 return np.max(np.mean(scores,(1,2)))
#             else:
#                 return np.max(np.mean(scores,1))
#         except:
#             return 0