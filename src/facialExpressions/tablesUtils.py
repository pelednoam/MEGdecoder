'''
Created on Jan 7, 2013

@author: noam
'''
import tables
import numpy as np


class TablesUtils():

    def __init__(self, h5file):
        self.h5file = h5file

    def findOrCreateMatTalbe(self, tableName, groupName, dtype=np.float64, shape=[]):
        table = self.findTable(tableName, groupName)
        if (table is None):
            atom = tables.Atom.from_dtype(dtype)
            group = self.findGroup(groupName)
            table = self.h5file.createCArray(group, tableName, atom, shape)
        return table

    def createMatTable(self, tableName, groupName, dtype=np.float64, shape=[]):
        table = self.findTable(tableName, groupName)
        if (table is not None):
            table.remove()
        atom = tables.Atom.from_dtype(dtype)
        group = self.findGroup(groupName)
        table = self.h5file.createCArray(group, tableName, atom, shape)
        return table

    def checkIfRecordExist(self, table, keyVal, keyField='id'):
        try:
            next(table.where('(%s==%r)' % (keyField, keyVal)))
        except StopIteration:
            return False
        else:
            return True

    def findGroup(self, name=''):
        if (name == ''): 
            group = self.h5file.root
        else:
            try:
                group = self.h5file.getNode('/%s' % name)
            except:
                group = self.h5file.createGroup('/', name)
        return group

    def findTable(self, tableName, groupName=''):
        try:
            path = '/%s' % (tableName) if groupName == '' else '/%s/%s' % (groupName, tableName)
            table = self.h5file.getNode(path)
        except:
            table = None
        return table

    def findOrCreateTable(self, tableName, descClass, tableDesc='', groupName=''):
        tab = self.findTable(tableName, groupName)
        if (tab is None):
            group = self.h5file.root if groupName == '' else self.findGroup(groupName)
            tab = self.h5file.createTable(group, tableName, descClass, tableDesc)
        return tab

    def removeTable(self, tableName, groupName=''): 
        try:
            self.h5file.removeNode('/%s' % groupName, tableName)
        except:
            pass

    def removeGroup(self, groupName):
        try:
            self.h5file.removeNode('/', groupName, recursive=True)
        except:
            print('Cant remove group %s' % groupName)
