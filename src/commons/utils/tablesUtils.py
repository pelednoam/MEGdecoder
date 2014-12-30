'''
Created on Nov 3, 2014

@author: noampeled
'''

import numpy as np

try:
    import tables
    DEF_TABLES = True
except:
    print('no pytables!')
    DEF_TABLES = False

# http://stackoverflow.com/questions/9002433/how-should-python-dictionaries-be-stored-in-pytables
tables_dict = {
    'key': tables.StringCol(itemsize=40),
    'value': tables.Int32Col(),
}


def createHDF5File(fileName):
    try:
        return tables.open_file(fileName, mode='w')
    except:
        return tables.openFile(fileName, mode='w')


def openHDF5File(fileName):
    try:
        return tables.open_file(fileName, mode='a')
    except:
        return tables.openFile(fileName, mode='a')


# dtype = np.dtype('int16') / np.dtype('float64')
def createHDF5ArrTable(hdfFile, group, arrayName,
        dtype=np.dtype('float64'), shape=(), arr=None,
        complib='blosc', complevel=5):
    atom = tables.Atom.from_dtype(dtype)
    if (arr is not None):
        shape = arr.shape
#     filters = tables.Filters(complib=complib, complevel=complevel)
    if (not arrayName in group._v_children):
        try:
            ds = hdfFile.create_carray(group, arrayName, atom, shape)
        except:
            ds = hdfFile.createArray(group, arrayName, atom, shape)
    else:
        ds = group._v_children[arrayName]

    if (arr is not None):
        ds[:] = arr
    return ds


def createHDFTable(hdf5File, group, tableName, tableDesc):
    if (not tableName in group._v_children):
        tab = hdf5File.createTable(group, tableName, tableDesc)
    else:
        tab = group._v_children[tableName]
#     tab.cols.key.createIndex()
    return tab


def addDicitemsIntoTable(tab, d):
    tab.append(d.items())


def readDicFromTable(tab, keyVal):
    vals = [row['value'] for row in tab.where('key == keyVal')]
    if (len(vals) > 0):
        return vals[0]
    else:
        return None


def findOrCreateGroup(h5file, name=''):
    if (name == ''):
        group = h5file.root
    else:
        try:
            group = h5file.getNode('/{}'.format(name))
        except:
            group = h5file.createGroup('/', name)
    return group


def findTable(h5file, tableName, groupName=''):
    try:
        path = '/{}'.format(tableName) if groupName == '' else \
               '/{}/{}'.format(groupName, tableName)
        table = h5file.getNode(path)
    except:
        table = None
    return table
