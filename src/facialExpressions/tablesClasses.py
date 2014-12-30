# -*- coding: utf-8 -*-
'''
Created on Dec 23, 2012

@author: noam
'''
import tables
# tables.HDF5_DISABLE_VERSION_CHECK = 1
import numpy as np 
import itertools
from tablesUtils import TablesUtils
# import utils

# HDF5_FILE_NAME = '/Users/noampeled/Copy/Big files/centipedeTwoPlayersNew/rawPoints.h5'
# HDF5_FILE_NAME = '/home/noam/Documents/TwoPlayersResults/twoPlayers.h5'
# HDF5_FILE_NAME = '/home/noam/Documents/TwoPlayersResults/twoPlayersNew.h5'
HDF5_FILE_NAME = '/Users/noampeled/Copy/Big files/centipede/PhdTwoPlayersRawPoints.h5'
# HDF5_FILE_NAME = '/home/noam/Documents/uni/centipede/singlePlayer.h5'
HDF5_AU_FILE_NAME = '/home/noam/Copy/Big files/centipede/AU.h5'
# HDF5_AU_FILE_NAME = '/home/ftp_folder/centipede/AU.h5'
# HDF5_FILE_NAME = '/home/ftp_folder/centipede/rawPoints.h5'
# HDF5_FILE_NAME = '/home/noam/Documents/uni/centipede/rawPointsSmooth.h5'

GROUP_RAW_DATA = 'rawData'
GROUP_KALMAN = 'kalman'
GROUP_NORM = 'norm'
GROUP_PARAMETRISE = 'parametrise'

FEATURES_COVA = 'cova'
FEATURES_REAL_COV = 'real_cov'
FEATURES_SUMS = 'sums'
FEATURES_MOVEMENT = 'mov'
FEATURES_COSINE = 'cos'

MAIN_CENTROIDS_NAMES = ['eyesNoseRatios', 'rightEyeCentroids', 'leftEyeCentroids', 'noseCentroids', 'eyesCentroids', 'noseAngles', 'noseLocations']

class Tables():
    def __init__(self, h5FileName, read=True):
        if (read):
            self.h5file = tables.openFile(h5FileName, mode="r+", title="Centipede Logs")
            self.tu = TablesUtils(self.h5file)
            self.playerTable = self.tu.findTable('players')
            self.gamesTable = self.tu.findTable('games')
        else:
            self.h5file = tables.openFile(h5FileName, mode="w", title="Centipede Logs")
            self.tu = TablesUtils(self.h5file)
            self.createTables()

    def createTables(self):
        self.rawDataGroup = self.h5file.createGroup('/', GROUP_RAW_DATA)        
        self.kalmanDataGroup = self.h5file.createGroup('/', GROUP_KALMAN)
        self.playerTable = self.h5file.createTable(self.h5file.root, 'players', PlayerDesc, "players details")
        self.gamesTable = self.h5file.createTable(self.h5file.root, 'games', GameDesc, "raw data details")

    def close(self):
        self.h5file.close()

    def addRound(self, data, subjectID, gameID, roundID, index, playerAction, oppAction, frameRate):
        self.addDataTable(data, subjectID, gameID, roundID, GROUP_RAW_DATA)
        if (not self.tu.checkIfRecordExist(self.gamesTable, index)):
            game = self.gamesTable.row
            game['id'] = index
            game['subject'] = subjectID
            game['game'] = gameID
            game['round'] = roundID
            game['playerAction'] = playerAction
            game['oppAction'] = oppAction
            game['frameRate'] = frameRate
            game.append()

    def addPlayerDemographic(self, playerName, demoElm):
        playerID = self.playerID(playerName)
        self.demogTable = self.tu.findOrCreateTable('playersDemographic', PlayerDemographicDesc, 'players Demographic')
        if (not self.tu.checkIfRecordExist(self.demogTable, playerID)):
            demo = self.demogTable.row
            demo['id'] = playerID
            demo['age'] = int(demoElm.get('Age'))
            demo['gender'] = demoElm.get('Gender')
            demo['countryOfBirth'] = demoElm.get('CountryOfBirth')
            demo['parentsCountryOfBirth'] = demoElm.get('ParentsCountryOfBirth')
            demo['educationType'] = demoElm.get('EducationType')
            demo['educationField'] = demoElm.get('EducationField')
            demo['isStudent'] = utils.boolStrToInt(demoElm.get('IsStudent'))
            demo.append()

    def addPlayersCouple(self, name1, name2):
        self.couplesTable = self.tu.findOrCreateTable('playersCouples', PlayersCouples, 'players couples')
        try:
            id1 = self.playerID(name1)
        except:
            print('no id for {}, abort!'.format(name1))
            raise Exception('Q@$#%@#$TWEG')
        try:
            id2 = self.playerID(name2)
        except:
            print('{} has no opp (suppose to be {}), let it be his own opp'.format(name1,name2))
            id2=id1
        couple = self.couplesTable.row
        couple['id1']=id1
        couple['id2']=id2
        couple.append()

    def getCoupledPlayer(self, playerID):
        self.couplesTable = self.tu.findOrCreateTable('playersCouples', PlayersCouples, 'players couples')
        games = self.couplesTable.where('(id1==%r)' % (playerID))
        return [game['id2'] for game in games][0]

    def getDemographics(self):
        demo = {}
        self.demogTable = self.tu.findTable('playersDemographic') 
        for recName in ['age', 'gender']:
            demo[recName] = []
        for rec in self.demogTable:
            for recName in ['age', 'gender']:
                demo[recName].append(rec[recName])
        return demo

    def addDataTable(self, data, subjectID, gameID, roundID, groupName):
        self.points = self.tu.findOrCreateMatTalbe('T%d_%d_%d' % (subjectID, gameID, roundID), groupName, data.dtype, data.shape)
        self.points[:] = data

    def removeFeatures(self, groupType, featuresType, polarFeaturesCalc):
        self.tu.removeTable(self.featuresType(groupType, featuresType, polarFeaturesCalc))

    def removeDemographic(self):
        self.tu.removeTable('playersDemographic')

    def removePlayersCouples(self):
        self.tu.removeTable('playersCouples')
        
    def addFeatures(self, features, groupType, index, featuresType, polarFeaturesCalc):
        shape = 1 if len(features.shape) == 0 else features.shape[0]
        self.features = self.tu.findOrCreateMatTalbe(self.featuresType(groupType, featuresType, polarFeaturesCalc), '', features.dtype, (len(self.gamesTable), shape))
        self.features[index] = features
    
    def writePostFeatures(self, x, groupType, featuresType, compNum):
        tableName = self.postFeaturesType(groupType, featuresType, compNum)
        self.tu.removeTable(tableName)
        self.postFeatures = self.tu.findOrCreateMatTalbe(tableName, '', x.dtype, x.shape)
               
    def saveMainCentroids(self, mainCentroids, dataGroup, subjectID, gameID, roundID):
        for index, centroid in enumerate(mainCentroids):
            table = self.tu.createMatTable('T%d_%d_%d_%s' % (subjectID, gameID, roundID, MAIN_CENTROIDS_NAMES[index]), dataGroup, centroid.dtype, centroid.shape)
            table[:] = centroid        
   
    def loadMainCentroids(self, dataGroup, subjectID, gameID, roundID):
        centroids = []
        for centroidName in MAIN_CENTROIDS_NAMES:
            centroids.append(self.getCentoirdTable(centroidName, dataGroup, subjectID, gameID, roundID)[:])
        return list(centroids)
   
    def addPlayer(self, subjectID, subjectName):
        if (not self.tu.checkIfRecordExist(self.playerTable, subjectID)):
            player = self.playerTable.row
            player['id'] = subjectID
            player['name'] = subjectName
            player.append()

    def getTables(self, groupName):
        for game in self.gamesTable:
            yield self.getDataTable(groupName, game), game['id'], game['subject'], game['game'], game['round'], game['frameRate']

    def get2Tables(self, groupName1, groupName2):
        for game in self.gamesTable:
            yield self.getDataTable(groupName1, game), self.getDataTable(groupName2, game), game['id'], game['subject'], game['game'], game['round']

    def getDataTable(self, groupName, game):
        return eval('self.h5file.root.%s.T%d_%d_%d' % (groupName, game['subject'], game['game'], game['round']))

    def getCentoirdTable(self, centroidName, dataGroup, subjectID, gameID, roundID):
        return eval('self.h5file.root.%s.T%d_%d_%d_%s' % (dataGroup, subjectID, gameID, roundID, centroidName))
    
    def gamesGenerator(self):
        for game in self.gamesTable:
            yield game['id'], game['subject'], game['game'], game['round']
    
    def getPlayersRounds(self):
        group = {}
        for subject, games_grouped_by_subject in itertools.groupby(self.gamesTable, lambda row:row['subject']):
            group[subject] = [game['id'] for game in games_grouped_by_subject]
        return group
    
    def getPlayerRounds(self, playerID):
        games = self.gamesTable.where('(subject==%r)' % (playerID))
        return [game['id'] for game in games]
    
    def getPlayerActionIndices(self, playerID, action):
        games = self.gamesTable.where('(playerAction==%r)&(subject==%r)' % (action, playerID))
        return [game['id'] for game in games]

    def getPlayerActionHistoryIndices(self, playerID, action, upperIndex):
        games = self.gamesTable.where('(playerAction==%r)&(subject==%r)&(id<%r)' % (action, playerID, upperIndex))
        return [game['id'] for game in games]
   
    def getActions(self):
        return np.array([game['playerAction'] for game in self.gamesTable])
        
    def getFeatures(self, groupType, featuresType, polarFeaturesCalc):
        return self.tu.findTable(self.featuresType(groupType, featuresType, polarFeaturesCalc))[:]
    
    def postFeatures(self, groupType, featuresType, compsNum): 
        return self.tu.findTable(self.postFeaturesType(groupType, featuresType, compsNum))
    
    def playerName(self, playerID):
        player = next(self.playerTable.where('(id==%r)' % (playerID)))
        return player['name']

    def playerID(self, playerName):
        player = next(self.playerTable.where('(name==%r)' % (playerName)))
        return player['id']
    
    def frameRate(self, roundID):
        game = next(self.gamesTable.where('(id==%r)' % (roundID)))
        return game['frameRate']
    
    def createGamesTable(self):    
        self.tu.removeTable('games')
        self.gamesTable = self.h5file.createTable(self.h5file.root, 'games', GameDesc, "games details")
    
    def featuresType(self, groupType, featuresType, polarFeaturesCalc):
        return '%sFeatures_%s%s' % (groupType, featuresType, '' if polarFeaturesCalc else '_nonPolar')

    def postFeaturesType(self, groupType, featuresType, compNum):
        return '%sPostFeatures_%s_%d' % (groupType, featuresType, compNum)
    
    @property
    def subjectsIDs(self): return [game['subject'] for game in self.gamesTable]
    @property
    def players(self): return self.playerTable    
    @property
    def gamesLen(self): return self.gamesTable.shape[0]
    

class PlayerDesc(tables.IsDescription):
    id = tables.UInt16Col()
    name = tables.StringCol(16)
    
class PlayerDemographicDesc(tables.IsDescription):    
    id = tables.UInt16Col()
    age = tables.UInt16Col()
    gender = tables.StringCol(16)
    countryOfBirth = tables.StringCol(16)
    parentsCountryOfBirth = tables.StringCol(16)
    educationType = tables.StringCol(16)
    educationField = tables.StringCol(16)
    isStudent = tables.UInt16Col()

class GameDesc(tables.IsDescription):
    id = tables.UInt16Col()
    subject = tables.UInt16Col()
    game = tables.UInt16Col()
    round = tables.UInt16Col()
    playerAction = tables.UInt16Col()
    oppAction = tables.UInt16Col()
    frameRate = tables.Float32Col()

class PlayersCouples(tables.IsDescription):
    id1 = tables.UInt16Col()
    id2 = tables.UInt16Col()