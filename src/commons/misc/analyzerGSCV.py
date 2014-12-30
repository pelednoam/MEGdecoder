'''
Created on Feb 12, 2014

@author: noampeled
'''
from analyzer import Analyzer

import numpy as np
from scipy.stats import sem
from sklearn import feature_selection
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import classification_report, make_scorer

from src.commons.utils import utils
from src.commons.utils import MLUtils
import src.commons.GridSearchSteps as GSS

class AnalyzerGSCV(Analyzer):
    '''
    Analyzer the runs all the processing steps inside the GridSearchCV
    '''
        
    def process(self,percentiles=[0],foldsNum=5,Cs=[1],gammas=[0],channelsNums=[30], kernels=['rbf'],k=5):
        ''' Step 3) Processing the data ''' 
        print('Proccessing the data')
        x,y = self.getXY(self.STEP_SPLIT_DATA) 
                
        svc = GSS.TSVC(C=1.0,kernel=kernels[0])
        cv = StratifiedShuffleSplit(y,k,0.2,random_state=0)
        rms = GSS.RMS()
        gmeanScore = make_scorer(GSS.gmeanScore, greater_is_better=True) # needs_proba
        selectChannels = GSS.SelectChannels(30)

        transform = feature_selection.SelectPercentile(feature_selection.f_classif)
        pipe = Pipeline([('selectChannels', selectChannels), ('rms', rms), ('anova', transform), ('svc', svc)])
#         pipe = Pipeline([('anova', transform), ('svc', svc)])
        estimator = GridSearchCV(pipe, dict(selectChannels__k=channelsNums, anova__percentile=percentiles, svc__C=Cs, svc__kernel = kernels, svc__gamma=gammas), cv=cv, scoring=gmeanScore, verbose=2)            
        utils.save(estimator, self.resultsFileName)

    def process(self, n_jobs=-2):
        x,y = self.getXY(self.STEP_SPLIT_DATA) 
        estimator = utils.load(self.resultsFileName)
        estimator.n_jobs = n_jobs
        estimator.fit(x, y)
        utils.save(estimator, self.resultsFileName)
    
    def calculatePredictionsScores(self,foldsNum):
        pass

    def analyzeResults(self,ROCFigType='jpg'):
        estimator = utils.load(self.resultsFileName)
        best_estimator_dict = estimator.best_estimator_.named_steps  
#         best_n_components = best_estimator_dict['pca'].n_components
#         best_channles = best_estimator_dict['selectChannels'].k
        best_channles=30
        best_percentiles = best_estimator_dict['anova'].percentile
        best_c = best_estimator_dict['svc'].C
        best_gamma = best_estimator_dict['svc'].gamma
        for params, mean_score, scores in estimator.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))
#         print('best n_components: {}, best c: {}'.format(best_n_components,best_c))
        print('best channels: {}, best percentiles: {}, best c: {}, best gamma: {}'.format(best_channles,best_percentiles,best_c,best_gamma))
        
        print('*** classification report on train data ***')
        x_train, y_train = self.getXY(self.STEP_SPLIT_DATA)
        cv = StratifiedShuffleSplit(y_train,test_size=0.2,random_state=0)
        scores = cross_val_score(estimator.best_estimator_, x_train, y_train, cv=cv, n_jobs=-2, scoring='roc_auc')
        print ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))
                
        print('*** classification report on heldout data ***')
        x_heldout, y_heldout = self.getXY(self.STEP_SAVE_HELDOUT_DATA)
        y_pred = estimator.best_estimator_.predict(x_heldout)
        print(classification_report(y_heldout, y_pred))   
        MLUtils.calcConfusionMatrix(y_heldout, y_pred, self.LABELS[self.procID]) 

