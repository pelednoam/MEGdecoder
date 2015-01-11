'''
Created on Sep 1, 2014

@author: noam
'''

from src.commons.analyzer.analyzer import Analyzer
from src.commons.utils import utils
from src.commons.utils import MLUtils
from src.commons.utils import plots
from src.commons import scoreFunctions as sf
from src.commons.utils import mpHelper
from src.commons.utils import tablesUtils as tabu
from src.commons.selectors.timeSelector import TimeSelector

import os
import numpy as np
from sklearn.datasets.base import Bunch
from abc import abstractmethod


class AnalyzerSelector(Analyzer):

    def process(self, **kwargs):
        ''' Step 3) Processing the data '''
        print('Proccessing the data. First load it')
        x, y, trialsInfo = self.getXY(self.STEP_PRE_PROCCESS, kwargs)
        print(x.shape)
        verbose = kwargs.get('verbose', True)
        utils.log(utils.count(np.array(y)), verbose)
        utils.save(Bunch(foldsNum=kwargs['foldsNum']),
                   self.predictionsParamtersFileName)  # T=T,
        cv = self.featuresCV(y, trialsInfo, kwargs['foldsNum'],
                             kwargs.get('testSize', None))
        utils.log('prepare CV params', verbose)
        params = self._prepareCVParams(utils.BunchDic(locals()))
        for p in params:
            p.overwriteResultsFile = kwargs.get('overwriteResultsFile', True)
        params = utils.splitList(params, self.jobsNum)
        func = self._preparePredictionsParameters
        if (self.jobsNum == 1):
            mapResults = [func(p) for p in params]  # For debugging
        else:
            mapResults = utils.parmap(func, params, self.jobsNum)
        utils.save(mapResults, self.paramsFileName)

    def _preparePredictionsParameters(self, ps):
        resultsFileNames = []
        for p in ps:
            t = utils.ticToc()
            resultsFileName, doCalc = self.checkExistingResultsFile(p)
            if (not doCalc):
                return resultsFileName
            x, ytrain, ytest, _, _ = self._preparePPInit(p)
            print('{} out of {}'.format(p.index, p.paramsNum))
            bestScore = Bunch(auc=0.5, gmean=0.5)
            bestParams = Bunch(auc=None, gmean=None)
            externalParams = Bunch(fold=p.fold)
            for hp in self.parametersGenerator(p):
                selector = self.selectorFactory(hp)
                xtrainFeatures = selector.fit_transform(x, ytrain, p.trainIndex)
                xtestFeatures = selector.transform(x, p.testIndex)
                if (xtrainFeatures.shape[0] > 0 and xtestFeatures.shape[0] > 0):
                    if (self.useSmote):
                        xtrainFeaturesTimedBoost, ytrainBoost = \
                            MLUtils.boostSmote(xtrainFeatures, ytrain)
                        if (self.shuffleLabels):
                            ytrainBoost = ytrainBoost[p.shuffleIndices]
                    else:
                        xtrainFeaturesTimedBoost, ytrainBoost = \
                            MLUtils.boost(xtrainFeatures, ytrain)

                    self._predict(Bunch(ytest=ytest,
                        xtrainFeatures=xtrainFeaturesTimedBoost,
                        ytrain=ytrainBoost, xtestFeatures=xtestFeatures,
                        kernels=p.kernels, Cs=p.Cs, gammas=p.gammas),
                        bestScore, bestParams, hp)

            utils.save((externalParams, bestScore, bestParams), resultsFileName)
            howMuchTime = utils.howMuchTimeFromTic(t)
            print('finish {}, {}'.format(externalParams, bestScore, howMuchTime))
        return resultsFileNames

    def checkExistingResultsFile(self, p):
        overwriteResultsFile = p.get('overwriteResultsFile', True)
        resultsFileName = '{}_{}_{}_results.pkl'.format(
            self.dataFileName(self.STEP_FEATURES, os.path.join(
            self.tempFolder))[:-4], p.fold, p.index)
        if (not overwriteResultsFile and utils.fileExists(resultsFileName)):
            print('{} already exist'.format(resultsFileName))
            return resultsFileName, False
        else:
            return resultsFileName, True

    def _preparePPInit(self, p, getX=True, getWeights=False, doCV=True):
        x, weights, trialsInfo = None, None, None
        if (tabu.DEF_TABLES):
            groupName = self.defaultGroup
            if (getX):
                x = tabu.findTable(self.hdfFile, 'x', groupName)
            # Add code to deal with the shuffling
            y = tabu.findTable(self.hdfFile, 'y', groupName)
            if (getWeights):
                weights = tabu.findTable(self.hdfFile, 'weights', groupName)
            trialsInfo = tabu.findTable(self.hdfFile, 'trialsInfo', groupName)
        else:
            if (getX):
                x = p.x.value
            y = p.y
            if ('trialsInfo' in p):
                trialsInfo = p.trialsInfo
            if (getWeights):
                weights = p.get('weights', None)
        if (doCV):
            ytrain = y[p.trainIndex]
            ytest = y[p.testIndex]
            return x, ytrain, ytest, trialsInfo, weights
        else:
            return x, y, trialsInfo, weights

    def selectorFactory(self, p, maxSurpriseVal=20, doPlotSections=False):
        verbose = p.get('verbose', True)
        utils.log('ss alpha: {}, ss len: {}'.format(
            p.sigSectionAlpha, p.sigSectionMinLength), verbose)
        return TimeSelector(p.sigSectionAlpha, p.sigSectionMinLength,
            p.onlyMidValue, self.xAxis, maxSurpriseVal,
            self.LABELS[self.procID], doPlotSections)

    def permutateTheLabels(self, y, trainIndex, useSmote):
        if (useSmote):
            # Create a shuffle indices. The length is twice the length
            # of the majority class trials number
            # Should do it here, because the shuffling can be done only
            # after the boosting, and we want to use the same shuffling
            # for every fold
            cnt = utils.count(y[trainIndex])
            majority = max([cnt[0], cnt[1]])
            shuffleIndices = np.random.permutation(majority * 2)
        else:
            shuffleIndices = np.random.permutation(len(y))
            y = y[shuffleIndices]
        return y, shuffleIndices

    def scorerFoldsResultsItem(self, score, probsScore, rates, res,
                               predRes, auc, gmean):
        return Bunch(score=score, probsScore=probsScore, fold=res.fold,
             rates=rates, sections=res.sections, auc=auc, gmean=gmean,
             y=res.ytest, probs=predRes.probs)

    def bestEstimatorItem(self, predictorParamters, probsScores, scores,
                          results):
        ys, probs, sections = [], [], []
        for res in results:
            ys.append(res.y)
            probs.append(res.probs)
            sections.append(res.sections)
        return Bunch(parameters=predictorParamters, probsScores=probsScores,
            scores=scores, rates=[res.rates for res in results],
            sections=sections, y=ys, probs=probs)

    def printBestSections(self, selector, onlyMidValue,
                          doPlot=True, doPrint=False):
        bestFeatures = []
        featuresAxis = self.featuresAxis(selector)
        for c, sensorSections in selector.sections.iteritems():
            for sec in sensorSections:
                if (onlyMidValue):
                    if (doPrint):
                        print('sensor {}: {}'.format(
                            c, featuresAxis[sec[2]]))
                    bestFeatures.append(featuresAxis[int(sec[2])])
                else:
                    if (doPrint):
                        print('sensor {}: {}-{}'.format(
                            c, selector.freqs[sec[0]],
                            selector.freqs[sec[1]]))
                    bestFeatures.extend(featuresAxis[range(sec[0],
                            sec[1] + 1)])
        if (doPlot and len(bestFeatures) > 1):
            plots.histCalcAndPlot(bestFeatures, binsNum=len(featuresAxis),
                xlabel=self.featuresAxisLabel,
                fileName=self.figureFileName('bestSections.jpg'))
        return bestFeatures

    def heldOutFeaturesExtraction(self, x, y, trialsInfo, bep,
            normalizationField='subject', maxSurpriseVal=20,
            doPlotSections=False):
        T = x.shape[1]
        timeStep = self.calcTimeStep(T)
        selector = self.selectorFactory(timeStep, bep, None,
            maxSurpriseVal, doPlotSections)
        xFeatures = selector.fit_transform(x, y)
        if (normalizationField in trialsInfo):
            xFeatures = self.normalizeFeatures(xFeatures, trialsInfo,
                normalizationField)
        return xFeatures, selector

    def fullDataAnlysis(self, dataRanges=[None], ratesThreshold=0.5,
                        permutationNum=2000, permutationLen=20,
                        maxSurpriseVal=20, doPlotSections=False,
                        doPrintBestSections=True, plotDataForGivenRange=True):
        self.xAxis = self.loadTimeAxis()
        bestEstimators = utils.load(self.bestEstimatorsFileName)
        featureExtractorName, bestEstimator = bestEstimators.iteritems().next()
        x, y, trialsInfo = self.getXY(self.STEP_SPLIT_DATA)
        xb, yb = MLUtils.boost(x, y)
        bep = bestEstimator.parameters
        bep.featureExtractorName = featureExtractorName
        _, selector = self.heldOutFeaturesExtraction(xb, yb,
            trialsInfo, bep, self.normalizationField,
            maxSurpriseVal=maxSurpriseVal, doPlotSections=doPlotSections)
        utils.save(selector, self.selectorName)

        if (doPrintBestSections):
            self.printBestSections(selector, bep.onlyMidValue, doPlot=True)
        if (plotDataForGivenRange):
            for dataRange in dataRanges:
                self.plotDataForGivenRange(x, y, selector, bep,
                    dataRange, doPlot=True)
        if (permutationNum > 0):
            foldImportanceAUC, foldImportanceGmean, foldAUC, foldGmean = \
                self.calcFoldImporatances(selector, x, y, x, y, bep, None,
                    ratesThreshold, permutationNum, permutationLen)
            utils.save((foldImportanceAUC, foldImportanceGmean,
                foldAUC, foldGmean), self.sensorsImportanceTrainFileName)
            plots.barPlot(foldImportanceGmean, 'Sensors Importance (gmean)',
                doShow=True, startsWithZeroZero=False,
                fileName=self.figureFileName('SensorsImportanceTrainGmean'))
            plots.barPlot(foldImportanceAUC, 'Sensors Importance (AUC)',
                doShow=True, startsWithZeroZero=False,
                fileName=self.figureFileName('SensorsImportanceTrainAUC'))

    def plotDataForGivenRange(self, X, y, selector, bep,
                              dataRange=None, doPlot=False):
        channels = []
        if (dataRange is None):
            dataRange = (-np.inf, np.inf)
        featuresAxis = self.featuresAxis(selector)
        for c, sensorSections in selector.sections.iteritems():
            for sec in sensorSections:
                if (bep.onlyMidValue):
                    features = featuresAxis[int(sec[2])]
                    if (features >= dataRange[0] and features <= dataRange[1]):
                        channels.append(c)
                        break
                else:
                    featuresAxis1 = featuresAxis[int(sec[0])]
                    featuresAxis1 = featuresAxis[int(sec[1])]
                    if (featuresAxis1 >= dataRange[0] and
                        featuresAxis1 <= dataRange[1]):
                        channels.append(c)
                        break
        channels = np.array(channels)
        print(channels)
        fs = self.calcFeaturesSpace(X, channels, bep)
        x0 = np.mean(fs[y == 0, :, :], 2).T
        x1 = np.mean(fs[y == 1, :, :], 2).T
        x0std = np.std(x0, 1)
        x1std = np.std(x1, 1)
        datalim = None
        if ('minFreq' in bep.keys()):
            datalim = [bep.minFreq, bep.maxFreq]
        labels = self.LABELS[self.procID][0], self.LABELS[self.procID][1]
        plots.graph2(featuresAxis, np.mean(x0, 1), np.mean(x1, 1), labels,
            xlim=datalim, xlabel=self.featuresAxisLabel,
            fileName=self.figureFileName('dataFor{}_{}.jpg'.format(
            dataRange[0], dataRange[1])), doPlot=doPlot,
            yerrs=[x0std, x1std])

    def calcFeaturesSpace(self, X, channels, bep):
        return X[:, :, channels]

    def heldOutFeaturesTransform(self, p, x_heldout, featureExtractorName):
        xHeldoutFeatures = p.selector.transform(x_heldout)
        xHeldoutFeaturesNormalized, _, _ = MLUtils.normalizeData(
            xHeldoutFeatures)
        return xHeldoutFeaturesNormalized

    def calcImporatances(self, foldsNum, testSize=None, x=None, y=None,
                 trialsInfo=None, bep=None, timeStep=None, removeSections=True,
                 ratesThreshold=0.5, permutationLen=20, permutationNum=2000,
                 doShow=False, doSave=True, doCalc=True, doPlotSections=False):
        if (doCalc):
            if (bep is None):
                bestEstimators = utils.load(self.bestEstimatorsFileName)
                bep = bestEstimators[bestEstimators.keys()[0]].parameters
            if (x is None):
                print('load all the data')
                x, y, trialsInfo = self.getXY(self.STEP_SPLIT_DATA)
                T = x.shape[1]
                timeStep = self.calcTimeStep(T)
            selector = self.selectorFactory(timeStep, bep,
                doPlotSections=doPlotSections)
            # split into folds, in each fold calculates the sensors importance
            # over the test
            cv = self.featuresCV(y, trialsInfo, foldsNum, testSize)
            importanceAUC, importanceGmean = [], []
            foldsAUC, foldsGmean = [], []
            for fold, (train_index, test_index) in enumerate(cv):
#                 xtrain, ytrain = x[train_index], y[train_index]
#                 xtest, ytest = x[test_index], y[test_index]
                # fit the selector using the train data
                selector.fit(x, y, train_index)
                if (len(selector.sections.keys()) == 0):
                    continue
                foldImportanceAUC, foldImportanceGmean, foldAUC, foldGmean = \
                    self.calcFoldImporatances(selector, x, y,
                    train_index, test_index,
                    bep, fold, ratesThreshold, removeSections,
                    permutationLen, permutationNum)
                if (foldImportanceAUC is not None):
                    importanceAUC.append(foldImportanceAUC)
                    importanceGmean.append(foldImportanceGmean)
                    foldsAUC.append(foldAUC)
                    foldsGmean.append(foldGmean)
            # If this function is being called from slidingWindowSelector,
            # there is no points in saving fold's results
#             print('AUCs: {}'.format(importanceAUC))
#             print('gmeans: {}'.format(importanceGmean))
            if (doSave):
                utils.save((importanceAUC, importanceGmean, foldsAUC,
                            foldsGmean), self.sensorsImportanceFileName(
                            removeSections, permutationNum, permutationLen))
            else:
                return (importanceAUC, importanceGmean, foldsAUC, foldsGmean)
        else:
            foldsImportanceAUC, foldsImportanceGmean, foldsAUC, foldsGmean = \
                utils.load(self.sensorsImportanceFileName(
                removeSections, permutationNum, permutationLen))
            print('AUCs: {}'.format(foldsAUC))
            print('gmeans: {}'.format(foldsGmean))
            importanceAUC = np.mean(np.array(foldsImportanceAUC), 0)
            importanceGmean = np.mean(np.array(foldsImportanceGmean), 0)
            utils.saveDictToMatlab(
                self.sensorsImportanceFileName(removeSections,
                permutationNum, permutationLen)[:-4],
                {'importanceAUC': importanceAUC,
                 'importanceGmean': importanceGmean})
            plots.barPlot(importanceGmean, 'Sensors Importance (gmean)',
                doShow=doShow, startsWithZeroZero=False,
                fileName=self.figureFileName('foldsSensorsImportanceTrainGmean'))
            plots.barPlot(importanceAUC, 'Sensors Importance (AUC)',
                doShow=doShow, startsWithZeroZero=False,
                fileName=self.figureFileName('foldsSensorsImportanceTrainAUC'))

    def calcFoldImporatances(self, selector, x, y, trainIndices, testIndices,
            bep, fold=None, ratesThreshold=0.5, removeSections=False,
            permutationLen=10, permutationNum=2000):
        ''' Calc the channels importance of a fold
        Parameters:
            removeSections: Boolenan | False
                If true, remove each permutation from all the sections
                If false, use only the secions in each permutation
        '''
        C = x.shape[2]
        foldImportanceAUC, foldImportanceGmean = np.zeros((C)), np.zeros((C))
        # calc the PS of the train and test data
        selectorInitObjTrain = selector.initTransform(x, trainIndices)
        selectorInitObjTest = selector.initTransform(x, testIndices)
        # calc the auc of the test data
        foldAUC, foldGmean, rates = self.tranformPredict(x, y,
            trainIndices, testIndices, selector, bep, selectorInitObjTrain,
            selectorInitObjTest, trainIndices, testIndices, doPrint=True)
        if (False):  # not self.passRatesThreshold(rates, ratesThreshold)):
            print('fold rates are too low! {}'.format(rates))
            return None, None, None, None
        else:
            print('{}auc: {}, rates: {}'.format('fold {} '.format(
                fold) if fold is not None else '', foldAUC, rates))
            print('#Channles: {}'.format(len(selector.sections.keys())))
            sections = selector.sections.copy()
            if (len(selector.sections.keys()) == 1):
                c = selector.sections.iterkeys().next()
                foldImportanceAUC[c] = foldAUC
                foldImportanceGmean[c] = foldGmean
            else:
                sectionsPerms = np.array(
                    [np.random.permutation(sections.keys()) for _ in
                     xrange(permutationNum)])
                sectionsPerms = sectionsPerms[:, :permutationLen]
                count = utils.count(np.ravel(sectionsPerms))
                # remove each sensor and predict again the test data
#                 fullSectionsComb = combinations(sections.keys(),
#                     combinationsLength)
#                 sectionsCombRC = utils.randomCombination(fullSectionsComb, 20)
                sectionsPerms = utils.splitList(sectionsPerms, self.jobsNum)
                if (not tabu.DEF_TABLES):
                    x = mpHelper.ForkedData(x)
                params = [Bunch(x=x, y=y, trainIndices=trainIndices,
                    testIndices=trainIndices, rates=rates,
                    selector=selector, sections=sections, bep=bep,
                    selectorInitObjTrain=selectorInitObjTrain,
                    selectorInitObjTest=selectorInitObjTest,
                    channlesCombinations=channlesCombination,
                    foldAUC=foldAUC, foldGmean=foldGmean,
                    removeSections=removeSections)
                    for channlesCombination in sectionsPerms]
                if (self.jobsNum == 1):
                    mapResults = [self._calcChannelsCombinationAccuracy(p)
                                  for p in params]  # For debugging
                else:
                    mapResults = utils.parmap(
                        self._calcChannelsCombinationAccuracy, params,
                        self.jobsNum)

                dAUCsAll, dGmeansAll = [], []
                for jobImportanceAUC, jobImportanceGmean, dAUCs, dGmeans in mapResults:
                    foldImportanceAUC += jobImportanceAUC
                    foldImportanceGmean += jobImportanceGmean
                    dAUCsAll.extend(dAUCs)
                    dGmeansAll.extend(dGmeans)
#                 plots.histCalcAndPlot(foldAUC - dAUCsAll)
#                 plots.histCalcAndPlot(foldGmean - dGmeansAll)

                # len(sections)-1 is the number of times each channel appear
                # in all the combinations, so divide to get the average,
                # and than substract it from the fold accuracy to get
                # the importance
                # In case of permutation, count each item
                channels = np.array(sections.keys())
                print('#Channels: {}'.format(len(sections)))
                for channel in channels:
                    foldImportanceAUC[channel] = \
                        foldImportanceAUC[channel] / count[channel]
                    foldImportanceGmean[channel] = \
                        foldImportanceGmean[channel] / count[channel]
                if (removeSections):
                    foldImportanceAUC = foldAUC - foldImportanceAUC
                    foldImportanceGmean = foldGmean - foldImportanceGmean

        return foldImportanceAUC, foldImportanceGmean, foldAUC, foldGmean

    def _calcChannelsCombinationAccuracy(self, p):
        x = p.x if tabu.DEF_TABLES else p.x.value
        C = x.shape[2]
        importanceAUC, importanceGmean = np.zeros((C)), np.zeros((C))
        dGmeans, dAUCs = [], []
        for cs in p.channlesCombinations:
            try:
                p.selector.sections = p.sections.copy()
                if (p.removeSections):
                    # Remove the cs from the selector.sections
                    for c in cs:
                        p.selector.sections.pop(c, None)
                else:
                    # Set the selector.sections as the cs
                    p.selector.sections = dict((c, v) for c, v in
                        p.sections.iteritems() if c in cs)
                # transform and predict using the other sections
                auc, gmean, _ = self.tranformPredict(
                    x, p.y, p.trainIndices, p.testIndices, p.selector, p.bep,
                    p.selectorInitObjTrain, p.selectorInitObjTest)
                dGmeans.append(gmean)
                dAUCs.append(auc)
                for c in cs:
                    importanceAUC[c] += auc
                    importanceGmean[c] += gmean
            except:
                utils.dump((p, cs), '_calcChannelsCombinationAccuracy')

        return (importanceAUC, importanceGmean,
            np.array(dAUCs), np.array(dGmeans))

    def tranformPredict(self, x, y, trainIndices, testIndices, selector, bep,
                        selectorInitObjTrain, selectorInitObjTest,
                        doPrint=False):
        xtrainFeatures = selector.transform(x, trainIndices,
            selectorInitObjTrain)
        if (len(xtrainFeatures) > 0):
            xtestFeatures = selector.transform(x, testIndices,
                selectorInitObjTest)
            xtrainFeaturesBoost, ytrainBoost = MLUtils.boost(
                xtrainFeatures, y[trainIndices])
            svc = self.predictor(bep.C, bep.kernel, bep.gamma)
            svc.fit(xtrainFeaturesBoost, ytrainBoost)
            ytestPredProbs = svc.predict(xtestFeatures)
            ytestPred = MLUtils.probsToPreds(ytestPredProbs)
    #         MLUtils.calcConfusionMatrix(ytest, ytestPred, self.LABELS[self.procID],True) 
            rates = sf.calcRates(y[testIndices], ytestPred)
            gmean = sf.gmeanScoreFromRates(rates)
            acu = sf.AUCScore(y[testIndices], ytestPredProbs)
            return acu, gmean, rates
        else:
            return 0, 0, [0, 0]

    @abstractmethod
    def parametersGenerator(self, p):
        ''' hyper parameter generator for _preparePredictionsParameters '''

#     @abstractmethod
#     def selectorFactory(self, timeStep, p, params, maxSurpriseVal,
#                         doPlotSections):
#         ''' The selector generator '''

    @abstractmethod
    def createParamsObj(self, paramsTuple):
        ''' Gets a tuple and return a Bunch object
        of the selector parameters'''

    @abstractmethod
    def resultItem(self, selector, p, res, params, ytest):
        ''' The results item saved in _preparePredictionsParameters  '''

    @property
    def sectionsFileName(self):
        return '{}_sections.pkl'.format(
            self.dataFileName(self.STEP_FEATURES)[:-4])

    def sensorsImportanceFileName(self, removeSections, permutationNum,
                                  permutationLen):
        return '{}_sensorsImportance_{}_{}_{}.pkl'.format(
            self.dataFileName(self.STEP_FEATURES)[:-4],
            'removeSections' if removeSections else 'addSections',
            permutationNum, permutationLen)

    @property
    def defaultGroup(self):
        return self.STEPS[self.STEP_PRE_PROCCESS]

    @property
    def sensorsImportanceTrainFileName(self):
        return '{}_sensorsImportanceTrain.pkl'.format(self.dataFileName(self.STEP_FEATURES)[:-4])

    @property
    def allSectionsFileName(self):
        return '{}_allSections.pkl'.format(self.dataFileName(self.STEP_FEATURES)[:-4])

    @property
    def allSectionsTrainFileName(self):
        return '{}_allSectionsTrain.pkl'.format(self.dataFileName(self.STEP_FEATURES)[:-4])
    
    @property
    def selectorName(self):
        return 'FrequenciesSelector'
    
    @property
    def moreInfoForResultsFileName(self):
        return ''
        
    