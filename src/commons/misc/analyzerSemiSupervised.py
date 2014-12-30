'''
Created on Feb 12, 2014

@author: noampeled
'''

from analyzer import Analyzer

class AnalyzerSS(Analyzer):
    '''
    Semi-supervised version, where the channels selection is being performed outside the CV
    This must be performed using PCA and not sklearn.feature_selection.SelectKBest,
    because SelectKBest is using the labels
    '''
        
    def featuresGenerator(self,x,y,cv,channelsNums,featureExtractors,n_jobs):
        return self.featuresGeneratorChannlesFolds(x, y, cv, channelsNums, featureExtractors) 

        