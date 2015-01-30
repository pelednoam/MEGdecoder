# '''
# Created on Jul 24, 2014
# 
# @author: noam
# '''
# 
# from src.commons.analyzer.analyzerFreqsSelector import AnalyzerFreqsSelector
# from src.commons.analyzer.analyzer import Analyzer
# import src.commons.GridSearchSteps as GSS
# from src.commons.utils import mpHelper
# from src.commons.utils import utils
# from src.commons.utils import MLUtils
# from src.commons.utils import plots
# 
# from path3 import path
# import os
# import itertools
# from sklearn.datasets.base import Bunch
# import numpy as np
# from abc import abstractmethod
# from sklearn.feature_selection import SelectKBest, f_classif
# from collections import namedtuple
# 
# class AnalyzerPowerSpectrumSelector(AnalyzerFreqsSelector):
# 
#     SPECTRAL_FOLDER = 'spectral'
# 
#     def dataGenerator(self, matlabDic=None):
#         label1Folder, label2Folder = self.spectralFolder.dirs()
#         label1 = self.LABELS[self.procID].index(label1Folder.name)
#         label2 = self.LABELS[self.procID].index(label2Folder.name)
#         spectrumFile1 = sorted(label1Folder.dirs()[0].files())[0]
#         spectrumFile2 = sorted(label2Folder.dirs()[0].files())[0]
#         freq1ID = int(path(spectrumFile1).namebase.split('_')[-1])-1 # In matlab they start counting from 1... morrons...
#         freq2ID = int(path(spectrumFile2).namebase.split('_')[-1])-1
#         if (freq1ID!=freq2ID):
#             utils.throwException('freqs not equall!!!') 
#         matlabDic1 = utils.loadMatlab(spectrumFile1)
#         matlabDic2 = utils.loadMatlab(spectrumFile2)
#         x1 = matlabDic1['trialData{}'.format(label1Folder.name)]
#         x2 = matlabDic2['trialData{}'.format(label2Folder.name)]
#         for n1 in range(x1.shape[0]):
#             yield ((n1,label1),{})
#         for n2 in range(x2.shape[0]):
#             yield ((n2,label2),{})
# 
#     def process(self,foldsNum=5,Cs=[1],gammas=[0],kernels=['rbf'], 
#                                     sigSectionMinLengths=[10],sigSectionAlphas=[0.05],
#                                     minFreqs=[10], maxFreqs=[40], onlyMidValueOptions=[False], 
#                                     topPValsNums=[100],n_jobs=-2):
#         ''' Step 3) Processing the data ''' 
#         print('Proccessing the data')
#         x,y,trialsInfo = self.getXY(self.STEP_SPLIT_DATA) 
# 
#         utils.save(Bunch(foldsNum=foldsNum), self.predictionsParamtersFileName) # T=T,
#         cv = self.featuresCV(y,trialsInfo,foldsNum)
#         params = self._prepareCVParams(utils.BunchDic(locals()))
#         func = self._preparePredictionsParameters
#         # no parallel here, only in spectralFitTransform
#         mapResults = [func(p) for p in params] 
#         utils.save(mapResults, self.paramsFileName)
# 
#     def _prepareCVParams(self, p):
#         params = []
#         paramsNum = len(list(p.cv))
#         index = 0
#         for fold, (train_index, test_index) in enumerate(p.cv):
#             xtrain = mpHelper.ForkedData(p.x[train_index])
#             ytrain = mpHelper.ForkedData(p.y[train_index])
#             xtest = mpHelper.ForkedData(p.x[test_index])
#             ytest = mpHelper.ForkedData(p.y[test_index])
#             params.append(Bunch(xtrain=xtrain, ytrain=ytrain, xtest=xtest,
#                 ytest=ytest, trainTrialsInfo=p.trialsInfo[train_index],
#                 testTrialsInfo=p.trialsInfo[test_index], fold=fold,
#                 paramsNum=paramsNum, minFreqs=p.minFreqs,
#                 maxFreqs=p.maxFreqs, onlyMidValueOptions=p.onlyMidValueOptions,
#                 kernels=p.kernels, Cs=p.Cs, gammas=p.gammas,
#                 topPValsNums=p.topPValsNums, index=index,
#                 sigSectionMinLengths=p.sigSectionMinLengths,
#                 sigSectionAlphas=p.sigSectionAlphas, n_jobs=p.n_jobs))
#             index += 1
#         return params
# 
#     def _preparePredictionsParameters(self,p,overwriteResultsFile=True):
#         resultsFileName = '{}__SpectralSelector_{}_results.pkl'.format(self.dataFileName(self.STEP_FEATURES,self.tempFolder)[:-4],p.fold)
#         if (not overwriteResultsFile and utils.fileExists(resultsFileName)):
#             print('{} already exist, continue to the next one'.format(resultsFileName))
#             return resultsFileName
# 
# #         tic = utils.ticToc()
#         results = []
#         xtrain,ytrain,xtest,ytest = p.xtrain.value, p.ytrain.value, p.xtest.value, p.ytest.value        
#         allTransformParams = itertools.product(*(p.sigSectionMinLengths,p.sigSectionAlphas))
#         allFreqsParams = itertools.product(*(p.minFreqs,p.maxFreqs))
#         for minSegSectionLen,alpha in allTransformParams:
#             print(alpha,minSegSectionLen)
#             xtrainTrans,xtestTrans,minPVals,sigFreqs = self.spectralFitTransform(xtrain, ytrain, xtest, ytest,alpha,minSegSectionLen, p.n_jobs)
#             for minFreq,maxFreq in allFreqsParams:
#                 xtrainTransFiltered,xtestTransFiltered,minPValsFiltered = self.freqsFilter(xtrainTrans, xtestTrans, minPVals, sigFreqs, minFreq, maxFreq)
#                 pvalsIndices = np.argsort(minPValsFiltered)
#                 for topPValsNum in p.topPValsNums:
#                     indices = pvalsIndices[:topPValsNum]
#                     xtrainTransFilteredMin = xtrainTransFiltered[:,indices]
#                     xtestTransFilteredMin = xtestTransFiltered[:,indices]
# #                     print(xtrainTransFilteredMin.shape,xtestTransFilteredMin.shape)                
#                     if (xtrainTransFilteredMin.shape[0]==0 or xtestTransFilteredMin.shape[0]==0):
# #                         print(alpha, minSegSectionLen, minFreq, maxFreq, 'shape 0!')
#                         pass
#                     else:
#                         xtrainTransFilteredMinBoost, ytrainBoost = MLUtils.boost(xtrainTransFilteredMin, ytrain)
#                         res = self._predict(Bunch(xtrainFeatures=xtrainTransFilteredMinBoost,ytrain=ytrainBoost,
#                             xtestFeatures=xtestTransFilteredMin,kernels=p.kernels,Cs=p.Cs,gammas=p.gammas))
#                         results.append(Bunch(predResults=res,fold=p.fold,ytest=ytest,channelsNum=248,
#                             featureExtractorName=Analyzer.FE_ALL, timePercentile=100, topPValsNum = topPValsNum,
#                             sections = [], minFreq=minFreq, maxFreq=maxFreq, onlyMidValue=True,
#                             sigSectionMinLength=minSegSectionLen, sigSectionAlpha=alpha))
#         
#         #         utils.howMuchTimeFromTic(tic, '_preparePredictionsParametersChannelsFrequenciesSelector')
#         utils.save(results, resultsFileName)
#         return resultsFileName
#     
#     def freqsFilter(self,xtrainTrans,xtestTrans,minPVals,freqs,minFreq,maxFreq):
#         indices = np.where((freqs>=minFreq) & (freqs<=maxFreq))[0]
#         xtrainTrans = xtrainTrans[:,indices]
#         xtestTrans = xtestTrans[:,indices]
#         minPVals = minPVals[indices]
#         return (xtrainTrans,xtestTrans,minPVals)
#         
#     def spectralFitTransform(self,xtrain,ytrain,xtest,ytest,alpha,minSegSectionLen,n_jobs):
#         results, minPVals, retFreqs, params, errFiles = [],[],[],[],[]
#         xtrainTrans,xtestTrans = None,None
#         freqs = self.loadFreqs()
#         selector = SelectKBest(f_classif, k=50)
#         label1Folder, label2Folder = self.spectralFolder.dirs() 
#         label1 = self.LABELS[self.procID].index(label1Folder.name)
#         label2 = self.LABELS[self.procID].index(label2Folder.name)
#         x1Inds = xtrain[ytrain==label1]
#         x2Inds = xtrain[ytrain==label2] # should be len(set(x1Inds).intersection(set(x2Inds)))==0
#         x1IndsTest = xtest[ytest==label1]
#         x2IndsTest = xtest[ytest==label2]     
# #         print('results in {}'.format(self.spectralFitTransformResultsFileName))  
#         for sensor1, sensor2 in zip(sorted(label1Folder.dirs()),sorted(label2Folder.dirs())):
#             params.append(Bunch(ytrain=ytrain,ytest=ytest,label1=label1,label2=label2,selector=selector,freqs=freqs,alpha=alpha,minSegSectionLen=minSegSectionLen,
#                                 sensor1=sensor1,sensor2=sensor2,label1Folder=label1Folder,label2Folder=label2Folder,x1Inds=x1Inds,x2Inds=x2Inds,x1IndsTest=x1IndsTest,x2IndsTest=x2IndsTest))
#   
#         print('n_jobs = {}'.format(n_jobs))
#         func = self._spectralFitTransform
# #         mapResults = [func(params[0])]
#         if n_jobs==1: mapResults = [func(p) for p in params] # For debugging
#         else: mapResults = utils.parmap(func, params, n_jobs)
# 
#         for p in mapResults:
#             if (p.xtrainTrans is not None):
#                 xtrainTrans = p.xtrainTrans if xtrainTrans is None else np.hstack((xtrainTrans,p.xtrainTrans))
#                 xtestTrans = p.xtestTrans if xtestTrans is None else np.hstack((xtestTrans,p.xtestTrans))
#                 minPVals.extend(p.minPVals)
#                 retFreqs.extend(p.retFreqs)
#                 results.extend(p.results)
#                 errFiles.extend(errFiles)
#   
#         utils.save(errFiles, 'errFiles')
#         utils.save((minPVals,results), self.spectralFitTransformResultsFileName)
#         return (xtrainTrans,xtestTrans,np.array(minPVals),np.array(retFreqs))
#             
#     def _spectralFitTransform(self,p):
#         xtrainTrans,xtestTrans=None,None
#         minPVals,results,retFreqs,errFiles=[],[],[],[]
#         print('sensor {}'.format(p.sensor1.name))
#         for spectralFile1,spectralFile2 in zip(sorted(p.sensor1.files()),sorted(p.sensor2.files())):
#             try:
#                 freq1ID = int(path(spectralFile1).namebase.split('_')[-1])-1
#                 freq2ID = int(path(spectralFile1).namebase.split('_')[-1])-1
#                 if (freq1ID!=freq2ID): utils.throwException('freq1ID!=freq2ID! {},{}'.format(freq1ID,freq2ID))
#                 freq = p.freqs[p.label1Folder.name][freq1ID]
#                 matlabDic1 = utils.loadMatlab(spectralFile1)
#                 matlabDic2 = utils.loadMatlab(spectralFile2)
#                 x1 = matlabDic1['trialData{}'.format(p.label1Folder.name)]
#                 x2 = matlabDic2['trialData{}'.format(p.label2Folder.name)]                
#                 x1 = x1[:,~np.isnan(x1[0])]
#                 x2 = x2[:,~np.isnan(x2[0])] # should be x1.shape[1]==x2.shape[1]
#                 if (x1.shape[1]!=x2.shape[1]): utils.throwException('x1.shape[1]!=x2.shape[1] {},{}'.format(x1.shape[1],x2.shape[1]))
#                 X = np.zeros((len(p.x1Inds)+len(p.x2Inds),x1.shape[1]))
#                 X.fill(np.nan)
#                 X[p.ytrain==p.label1,:] = x1[p.x1Inds,:]
#                 X[p.ytrain==p.label2,:] = x2[p.x2Inds,:]
#                 if (not np.all(~np.isnan(X))): utils.throwException('not np.all(~np.isnan(X))!')
#                 (sigIndices, sigPVals) = self.findSigSections(X,p.ytrain,p.selector,p.alpha,p.minSegSectionLen,True)
#                 if (len(sigIndices)>0):
#                     for sigInd,pval in zip(sigIndices,sigPVals):
#                         minPVals.append(pval)
#                         retFreqs.append(freq)
#                         results.append(Bunch(sigInd=sigInd,sensor=p.sensor1.name))
#                     xtrainTrans = X[:,sigIndices] if xtrainTrans is None else np.hstack((xtrainTrans,X[:,sigIndices]))
#     
#                     Xtest = np.zeros((len(p.x1IndsTest)+len(p.x2IndsTest),x1.shape[1]))
#                     Xtest.fill(np.nan)
#                     Xtest[p.ytest==p.label1,:] = x1[p.x1IndsTest,:]
#                     Xtest[p.ytest==p.label2,:] = x2[p.x2IndsTest,:]
#                     xtestTrans = Xtest[:,sigIndices] if xtestTrans is None else np.hstack((xtestTrans,Xtest[:,sigIndices]))
#             except:
#                 print('error reading {}'.format(spectralFile1))
#                 errFiles.append((spectralFile1,spectralFile2))
# 
#         return Bunch(xtrainTrans=xtrainTrans,xtestTrans=xtestTrans,minPVals=minPVals,retFreqs=retFreqs,results=results,errFiles=errFiles)        
# 
#     def findSigSections(self,x,y,selector,alpha,minSegSectionLen,onlyMidValue=False):
#         model = selector.fit(x,y)
#         sections = utils.findSectionSmallerThan(model.pvalues_, alpha, minSegSectionLen)
#         indices,pvals=[],[]
#         for sec in sections:
#             ind = np.argmin(model.pvalues_[sec[0]:sec[1]])+sec[0]
#             pvals.append(model.pvalues_[ind])
#             indices.append(ind)
#         return (np.array(indices), np.array(pvals))
# 
#     def calculatePredictionsScores(self):
# #         minPVals,results = utils.load(self.spectralFitTransformResultsFileName)
# #         plots.graph(None, minPVals)
#         super(AnalyzerPowerSpectrumSelector, self).calculatePredictionsScores()
# 
#     def _predictorParamtersKeyClass(self):
#         return namedtuple('predictorParamters',['sigSectionMinLength','sigSectionAlpha','minFreq','maxFreq','topPValsNum','kernel','C','gamma'])
#     
#     def predictorParamtersKeyItem(self, res, predRes):
#         PredictorParamtersKey = self._predictorParamtersKeyClass()
#         return PredictorParamtersKey(sigSectionMinLength=res.sigSectionMinLength, sigSectionAlpha=res.sigSectionAlpha,
#                             minFreq=res.minFreq, maxFreq=res.maxFreq, topPValsNum=res.topPValsNum,                    
#                             kernel=predRes.kernel,C=predRes.C,gamma=predRes.gamma)
#  
#     def printBestPredictorResults(self, bestEstimator):
#         bep = bestEstimator.parameters
#         print('Best results: sigSectionMinLength: {}, sigSectionAlpha:{}, minFreqs: {}, maxFreqs: {}, topPValsNum:{}, kernel: {}, c: {}, gamma: {}'.format(
#             bep.sigSectionMinLength,bep.sigSectionAlpha,
#             bep.minFreq, bep.maxFreq, bep.topPValsNum, bep.kernel,bep.C,bep.gamma))
# 
# 
#     @abstractmethod
#     def loadFreqs(self): 
#         ''' Load the freqs arrays '''        
# 
#     def loadOriginalTimeAxis(self, T=None):
#         return
#         # Load the time axis     
#         timeAxisFullPath = os.path.join(self.spectralFolder, 'timeAxis.mat')
#         timeAxisDic = utils.loadMatlab(timeAxisFullPath)
#         timeAxis = np.array(timeAxisDic['timeAxis'][0])
#         timeAxis = timeAxis[self.nonnanInds]
#         self.T = len(timeAxis)
#         np.save(self.timeAxisFileName,timeAxis)
# 
#     def getTrialsTimeLength(self,matlabDic):
#         labels = self.LABELS[self.procID]
#         return matlabDic[labels[0]][0]['x'].shape[1]
# 
# 
#     @property
#     def spectralFitTransformResultsFileName(self):
#         return '{}_spectralFitTransformResultsFileName.pkl'.format(self.dataFileName(self.STEP_FEATURES)[:-4])
#     
#     @property
#     def spectralFolder(self): 
#         return path(os.path.join(self.folder,self.subject,self.SPECTRAL_FOLDER))
# 
#     