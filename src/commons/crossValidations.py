'''
Created on May 1, 2014

@author: noampeled
'''

import numpy as np
from src.commons.utils import utils

class IndicesKFold(object):

    def __init__(self,y,k):
        self.y = y
        self.k = k
        self.n = len(y)
        self.bin = int(self.n/self.k)

    def __iter__(self):
        indices = np.array(range(self.n))
        ind=0
        for _ in xrange(self.k):
            test_index=indices[ind:ind+self.bin]
            train_index=np.hstack((indices[0:ind],indices[ind+self.bin:]))
            ind+=self.bin
            yield train_index, test_index

class SubjectsLeaveOneOut(object):

    def __init__(self,subjects):
        self.subjects = subjects
        self.usubjects = np.unique(subjects) 
        self.k = len(self.usubjects)
        self.n = len(subjects)

    def __iter__(self):
        indices = np.array(range(self.n))
        for i in xrange(self.k):
            test_index=indices[self.subjects==self.usubjects[i]]
            train_index=indices[self.subjects!=self.usubjects[i]]
            yield train_index, test_index

class SubjectsKFold(object):

    def __init__(self,subjects,foldsNum):
        self.subjects = subjects
        self.usubjects = np.unique(subjects) 
        self.k = foldsNum
        self.n = len(subjects)

    def __iter__(self):
        for test_subjects in utils.chunks(self.usubjects, self.k):
            train_subjects = set(self.subjects)-set(test_subjects)
            print(test_subjects,train_subjects) 
            test_index=[ind for ind,x in enumerate(self.subjects) if x in test_subjects] 
            train_index=[ind for ind,x in enumerate(self.subjects) if x in train_subjects] 
            yield train_index, test_index 

class SubjectsPreset(object):

    def __init__(self,subjects, train_set, test_set):
        self.subjects = subjects
        self.usubjects = np.unique(subjects) 
        self.train_set = train_set
        self.test_set = test_set

    def __iter__(self):
        for train,test in zip(self.train_set,self.test_set):
            test_index=[ind for ind,x in enumerate(self.subjects) if x in test] 
            train_index=[ind for ind,x in enumerate(self.subjects) if x in train] 
            yield train_index, test_index

class TestCV(object):

    def __iter__(self):
        yield range(200),range(201,400)
