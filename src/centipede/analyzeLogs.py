'''
Created on Jun 9, 2014

@author: noampeled
'''

import os
from src.commons.utils import utils

ROOT = '/Users/noampeled/Copy/Data/MEG/data'
SUBJECTS_FOLDER = ['darya/data','dor','eitan', 'idan', 'liron', 'mosheB', 'raniB', 'shira', 'TalR2', 'yoni']
TRIGGERS_FILE = 'triggers.mat'

def readTriggers(root,subjectFolders,triggerFile):
    for subjectFolder in subjectFolders:
        print(subjectFolder)
        triggerFullFileName = os.path.join(root,subjectFolder,triggerFile)
        triggersMatlab = utils.loadMatlab(triggerFullFileName)
        triggers = triggersMatlab['trialsInfo']
        print(triggers.shape)

if __name__ == '__main__':
    readTriggers(ROOT,SUBJECTS_FOLDER,TRIGGERS_FILE)