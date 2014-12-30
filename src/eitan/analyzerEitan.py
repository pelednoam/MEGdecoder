'''
Created on Nov 28, 2013

@author: noampeled
'''
import numpy as np


from commons.analyzer import Analyzer
from commons import utils

class AnalyzerEitan(Analyzer):
    PROCS_NAMES = ['2_5', '1_2']
    PROC_2_5, PROC_1_2 = range(2)
    LABELS = [['2','5'],['1','2']]
    
    def __init__(self,folder,matlabFile,subject,procID,analID,indetifier='eitan'):
        Analyzer.__init__(self, folder, matlabFile, subject, procID, analID, indetifier)
    
    def dataGenerator(self, matlabDic):
        integralConditions = matlabDic['integral_conditions']
        N,M = integralConditions.shape
        label = 0
        for cond1 in range(N):
            for cond2 in range(M):
                label+=1
                featuresNum = integralConditions[cond1,cond2].shape[0]
                trials = integralConditions[cond1,cond2].reshape((featuresNum,-1)).T
                for trial in trials:
                    yield ((trial, label), {'cond1':cond1,'cond2':cond2})
                
    def trialCond(self,label,trialInfo):
        flag = False
        if (self.procID==self.PROC_2_5):
            flag = (label==2 or label==5)
        elif (self.procID==self.PROC_1_2):
            flag = (label==1 or label==2)
        else:
            utils.throwException('wrong procID!')
        return flag
    
    def trialLabel(self,label,trialInfo):
        if (self.procID==self.PROC_2_5):
            y = 0 if label==2 else 1 if label==5 else -1
        elif (self.procID==self.PROC_1_2):
            y = 0 if label==1 else 1 if label==2 else -1
        else:
            utils.throwException('wrong procID!')
        return y
    


