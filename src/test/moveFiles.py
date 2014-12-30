'''
Created on Dec 29, 2014

@author: noampeled
'''
import os
import shutil
from path3 import path

FOLDER = '/home/noam/Documents/MEGdata/centipede/data/svmFiles/temp'

def moveFiles(subject):
    files = path(FOLDER).files('*{}*'.format(subject))
    for file in files:
        shutil.move(os.path.join(sourceFolder, file), os.path.join(destFolder,fileName))

if __name__ == '__main__':
    pass