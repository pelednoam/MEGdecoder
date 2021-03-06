'''
Created on Aug 17, 2014

@author: noampeled
'''
from src.commons.analyzerTimeSlidingWindow.analyzerTimeSWSelector import AnalyzerTimeSWSelector
from src.commons.sliders.spacialSliders import SpatialWindowSlider
from src.commons.selectors.timeSelector import TimeSelector
from src.commons.utils import utils
from src.commons.utils import MLUtils
from src.commons.utils import plots
from src.commons.utils import tablesUtils as tabu

from collections import defaultdict
from sklearn.datasets.base import Bunch
import operator
import numpy as np
import traceback


class AnalyzerSpacialSWSelector(AnalyzerTimeSWSelector):

    def _preparePredictionsParameters(self, ps):
        resultsFileNames = []
        for p in ps:
            try:
                t = utils.ticToc()
                resultsFileName, doCalc = self.checkExistingResultsFile(p)
                resultsFileNames.append(resultsFileName)
                if (not doCalc):
                    continue
                cube, y, _, _ = self._preparePPInit(p)
                ytrain, ytest = y[p.trainIndex], y[p.testIndex]
                x = cube.x
                print('{} out of {}'.format(p.index, p.paramsNum))

                bestScore = Bunch(auc=0.5, gmean=0.5)
                bestParams = Bunch(auc=None, gmean=None)
                externalParams = Bunch(fold=p.fold, xCubeSize=p.xCubeSize,
                    yCubeSize=p.yCubeSize, zCubeSize=p.zCubeSize,
                    windowsOverlapped=p.windowsOverlapped,
                    cubeIndex=p.cubeIndex)
                for hp in self.parametersGenerator(p):
                    hp.voxelIndices = cube.voxelIndices
                    selector = self.selectorFactory(hp)
                    xtrainFeatures = selector.fit_transform(
                        x, ytrain, p.trainIndex)
                    xtestFeatures = selector.transform(x, p.testIndex)
                    hp.sections = selector.sections.keys()
                    hp.sectionsDic = selector.sectionsDic
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
            except:
                utils.dump(p)

            utils.howMuchTimeFromTic(t)
            if (bestParams.auc is None):
                print('never calculated the accuracy for {}'.format(externalParams))
            else:
                print('finish!, {}'.format(bestScore))
            utils.save((externalParams, bestScore, bestParams), resultsFileName)

        return resultsFileNames

    def selectorFactory(self, p, maxSurpriseVal=20, doPlotSections=False):
        return TimeSelector(p.sigSectionAlpha, p.sigSectionMinLength,
            p.onlyMidValue, self.xAxis, maxSurpriseVal,
            self.LABELS[self.procID], doPlotSections, p.voxelIndices)

    def resultsKeys(self, p):
        return (p.xCubeSize, p.yCubeSize, p.zCubeSize), p.cubeIndex

    def scorerFoldsResultsKey(self, res):
        return (res.xCubeSize, res.yCubeSize, res.zCubeSize)

    def scorerFoldsResultsItem(self, score, probsScore, rates, res,
                               predRes, auc, gmean):
        probs = predRes.probs if predRes is not None else []
        return Bunch(score=score, probsScore=probsScore, fold=res.fold,
            rates=rates, sections=res.sections, sectionsDic=res.sectionsDic,
            cubeIndex=res.cubeIndex, auc=auc, gmean=gmean, y=res.ytest,
            probs=probs)

    def _bestEstimatorsSortKey(self):
        return operator.itemgetter('cubeIndex', 'fold')

    def _bestEstimatorsGroupByKey(self):
        return operator.itemgetter('cubeIndex')

    def analyzeResults(self, calcProbs=False, threshold=0.5, doPlot=True,
                       printResults=False, cubesSizes=None, scoresAsDict=True,
                       doSmooth=True, smoothWindowSize=21, smoothOrder=3):
        bestEstimatorsPerCubeAccs = utils.load(
            self.bestEstimatorsPerWindowFileName)
        weigths = self.loadWeights()
        metaParams = utils.load(self.metaParametersFileName, True)
        for accFunc, bestEstimatorsPerCube in bestEstimatorsPerCubeAccs.iteritems():
            print('results for {}'.format(accFunc))
            scoresGenerator = self.scoresGeneratorPerWindow(
                bestEstimatorsPerCube, None, scoresAsDict, printResults, True)
            for (scores, scoresStd, _, sectionsIndices, cubesSize, bep) in scoresGenerator:
                if (cubesSizes is not None and cubesSize not in cubesSizes):
                    continue
                firstBep = bep[bep.keys()[0]][0]
                windowsOverlapped = firstBep.windowsOverlapped
                spacialSlider = SpatialWindowSlider(0, metaParams.xlim,
                    metaParams.ylim, metaParams.zlim, metaParams.xstep,
                    cubesSize[0], metaParams.ystep, cubesSize[1],
                    metaParams.zstep, cubesSize[2],
                    windowsOverlapped=windowsOverlapped, calcLocs=True)
                cubesScoresDic = defaultdict(list)
                for cubeIndex in spacialSlider.cubesEnumerator():
                    # Check if the a score was calculated for the current cube
                    # (If all the cube is outside the head, no score is being
                    # calculated)
                    if (cubeIndex in scores):
                        for voxelHullIndex in sectionsIndices[cubeIndex].values():
                            cubesScoresDic[voxelHullIndex].append(
                                scores[cubeIndex])
                cubes = []
                voxelIndices = np.array(cubesScoresDic.keys(), dtype=np.int)
                if (sum(utils.findZerosLines(weigths[voxelIndices])) > 0):
                    utils.throwException('Voxels outside the head!')
                for voxelHullIndex, cubeScores in cubesScoresDic.iteritems():
                    meanScore = np.mean(cubeScores)
                    if (np.mean(cubeScores) > threshold):
                        locs = spacialSlider.cubesLocs[voxelHullIndex]
                        cubes.append(np.hstack((locs, meanScore)))

                cubes = np.array(cubes)
                self.saveToBlender(cubes, metaParams.xstep)
                utils.saveToMatlab(cubes, self.cubesMatlabFileName, 'cubes')
                plots.scatter3d(cubes[:, 0:3], cubes[:, 3])
                plots.histCalcAndPlot(cubes[:, 3], binsNum=40)

    def findSignificantResults(self, foldsNum=0, doShow=True, windowSizes=None,
            doPlot=True, printResults=False, overwrite=False):
        metaParams = utils.load(self.metaParametersFileName, True)
        self.shuffleLabels = True
        bestEstimatorsShuffle = utils.load(
            self.bestEstimatorsPerWindowFileName, overwrite=overwrite)
        self.shuffleLabels = False
        bestEstimators = utils.load(
            self.bestEstimatorsPerWindowFileName, overwrite=overwrite)
        allCubes = defaultdict(list)
        for (accFunc, bestEstimatorPerWindow), \
                (_, bestEstimatorShufflePerWindow) in \
                zip(bestEstimators.iteritems(),
                    bestEstimatorsShuffle.iteritems()):
            print('calculating for {}'.format(accFunc))
            scoresGenerator = self.scoresGeneratorPerWindow(
                bestEstimatorPerWindow, printResults=printResults,
                meanScores=False, estimatorKeys=windowSizes, scoresAsDict=True)
            scoresGeneratorShuffle = self.scoresGeneratorPerWindow(
                bestEstimatorShufflePerWindow, printResults=printResults,
                meanScores=False, estimatorKeys=windowSizes, scoresAsDict=True)

            for (scores, _, _, sectionsIndices, cubesSize, bep), \
                (scoresShuffle, _, _, sectionsShuffleIndices, _, _) in \
                    zip(scoresGenerator, scoresGeneratorShuffle):

                firstBep = bep[bep.keys()[0]][0]
                windowsOverlapped = firstBep.windowsOverlapped
                spacialSlider = SpatialWindowSlider(0, metaParams.xlim,
                    metaParams.ylim, metaParams.zlim, metaParams.xstep,
                    cubesSize[0], metaParams.ystep, cubesSize[1],
                    metaParams.zstep, cubesSize[2],
                    windowsOverlapped=windowsOverlapped, calcLocs=True)
                cubesScoresDic = defaultdict(list)
                cubesScoresShulleDic = defaultdict(list)
                voxels, voxelsShuff = [], []
                for cubeIndex in spacialSlider.cubesEnumerator():
                    if (cubeIndex in scores):
                        for voxelHullIndex in \
                                sectionsIndices[cubeIndex].values():
                            voxels.append(voxelHullIndex)
                            cubesScoresDic[voxelHullIndex].extend(
                                scores[cubeIndex])
                    if (cubeIndex in scoresShuffle):
                        for voxelHullShuffleIndex in \
                                sectionsShuffleIndices[cubeIndex].values():
                            voxelsShuff.append(voxelHullShuffleIndex)
                            cubesScoresShulleDic[voxelHullShuffleIndex].extend(
                                scoresShuffle[cubeIndex])

                cubes = []
                for ((voxelHullIndex, cubeScores),
                    (_, cubeScoresShuffle)) in zip(
                        cubesScoresDic.iteritems(),
                        cubesScoresShulleDic.iteritems()):
                    ps = utils.ttestGreaterThan(cubeScores, cubeScoresShuffle)
                    if (ps < (0.05 / len(voxels))):
                        locs = spacialSlider.cubesLocs[voxelHullIndex]
                        cubes.append(np.hstack((locs, -np.log(ps),
                            np.mean(cubeScores))))

                cubes = np.array(cubes)
                allCubes[accFunc] = cubes
                plots.scatter3d(cubes[:, 0:3], cubes[:, 3])
#                 plots.histCalcAndPlot(cubes[:, 3], binsNum=40)

        utils.saveToMatlab(allCubes, self.cubesMatlabFileName, 'cubes')
#         self.saveToBlender(allCubes, metaParams.xstep)
        return cubes

    def loadWeights(self):
        if (tabu.DEF_TABLES):
            groupName = self.STEPS[self.STEP_PRE_PROCCESS]
            weights = tabu.findTable(self.hdfFile, 'weights', groupName)
        else:
            weightsDict = utils.loadMatlab(
                self.weightsFullFileName(self.weightsFileName))
            weights = weightsDict[self.weightsDicKey]
        return weights

    def saveToBlender(self, cubes, r):
        with open(self.blenderFileName, 'wb') as bFile:
            file_writer = utils.csv.writer(bFile, delimiter=',')
            colors = plots.arrToColors(cubes[:, 3])
            r /= 20.
            for index, (cube, c) in enumerate(zip(cubes, colors)):
                cube[:3] /= 10
                file_writer.writerow([index, 'q', cube[0], cube[1], cube[2],
                    r, c[0], c[1], c[2], '', '', '', '', 0.5])

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
                    timeSelector = GSS.TimeWindowSlider(startIndex, windowSize,
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
            timeSelector = GSS.TimeWindowSlider(0, windowSize, windowsNum, T)
            startIndices = np.array(timeSelector.windowsGenerator())
            timeAxis = self.xAxis[startIndices + windowSize / 2]

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
                'xAxis')
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

    def _animateImportanceDataGenerator(self, importance, timeAxis):
        c = len(importance[0])

        def dataGenerator():
            for windowImportance, t in zip(importance, timeAxis):
                if (windowImportance is None):
                    yield np.zeros((c)), t
                else:
                    yield windowImportance, t
            yield None, None
        return dataGenerator()

    def bestScore(self, scores, isProbs=False):
        return np.max(scores)

    @property
    def selectorName(self):
        return 'SpacialSWSelector'

    @property
    def sensorsImportanceFileName(self):
        return '{}_sensorsImportanceSlidingWindows.pkl'.format(
            self.dataFileName(self.STEP_FEATURES)[:-4])

    @property
    def cubesMatlabFileName(self):
        return '{}_cubes'.format(self.defaultFilePrefix)
