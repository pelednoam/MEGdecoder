'''
Created on Nov 24, 2013

@author: noampeled
'''

import sys
from multiprocessing import Pool

from commons import utils
from commons.analyzer import Analyzer

from analyzerEitan import AnalyzerEitan 

FOLDER = '/Users/noampeled/Copy/Data/MEG/eitan';
SUBJECT = '104'
MATLAB_FILE = 'integral_conditions_{}_2'.format(SUBJECT)

if __name__ == '__main__':
    args = sys.argv[1:]
    cpuNum = 1 if (len(args) < 1) else int(args[0])
    if (cpuNum > 1): print('cpuNum = %d' % cpuNum)
    pool = Pool(processes=cpuNum)
    
    t = utils.ticToc()
    
    analyze = AnalyzerEitan(FOLDER, MATLAB_FILE, SUBJECT,analID=Analyzer.ANAL_NONE, procID=AnalyzerEitan.PROC_2_5)
    analyze.preProcess(False)
    analyze.plot()
    analyze.ttest()
    analyze.process(foldsNum=3,doPCA=True,PCAcompsNum=5, kernelType=2)
    analyze.plot()
    analyze.ttest()
    analyze.analyzeResults()
        
    utils.howMuchTimeFromTic(t)

    