# -*- coding: utf-8 -*-
'''
Created on Dec 10, 2012

@author: noam
'''
import os
from path3 import path
from datetime import datetime, timedelta
import numpy as np
import utils
import MLUtils
import featuresExtraction as fe
import tablesClasses as tc
# from GameObjs import Round, Subject, Log, Game
from gameClasses import Round
import gameClasses as gc
import PCA
import plots

REWARDS = 'rewards.pkl'
SUMMARY_FILE_NAME = 'summery.xml'
MAX_ZEROS_POINTS = 0
KALMAN_DATA_FOLDER = '/home/noam/Documents/uni/centipede/kalmanData'
MAX_SECONDS_FOR_ROUNDS=30


def loadRawDataToTables(folder, groups, withInterpolation=False, badSubjects=[],duplicateNames=[]):
    folders = utils.sortFolders(path(folder).dirs()) 
    db = tc.Tables(tc.HDF5_FILE_NAME, read=False)
    db.createGamesTable()
    index = 0
    subjectID = -1
    for subFolder in folders:
        subjectID += 1
        playerName = fixPlayerNames(subFolder.name)
        if (playerName in badSubjects):
            print('{} is a bad subject').format(playerName)
            continue
        print('%d/%d: %s' % (subjectID + 1, len(folders), playerName))
        if (not isPlayerIsDuplicates(playerName,duplicateNames)):
            db.addPlayer(subjectID, playerName)
            duplicatePlayer = False
        else:
            duplicatePlayer = True
            lastGameID = getLastGameIDForDupPlayer(playerName, duplicateNames, folder) 
            playerName = getOriginalPlayerName(playerName,duplicateNames)
            subjectID = subjectID-1
        studentFolders = subFolder.dirs('Game*')
        studentFolders.sort()
        xmlFile = [file + '//' + SUMMARY_FILE_NAME for file in [studentFolders[-1],subFolder] if utils.fileExists(file + '//' + SUMMARY_FILE_NAME)][0]
        summary = utils.parseXMLFile(xmlFile)
        playerIndex = gc.getPlayerIndex(summary, playerName) 
        for gamesFolder in studentFolders:
#             if (not utils.fileExists(gamesFolder + '//' + SUMMARY_FILE_NAME)):
#                 print('In {} there is no summary file!'.format(gamesFolder))
#                 continue
            gameID, roundID = utils.getGameAndRoundID(gamesFolder.name)
#             print(gameID, roundID)
#             gameRound  = Round.xmlToRound(summary, playerName, gameID, roundID)
            roundElm = getRoundSummary(summary, gameID, roundID)
            roundStartTime = getRoundStartTime(roundElm)
            roundEndTime = getRoundEndTime(roundElm)
            playerActionTime = getRoudPlayerActionTime(roundElm, roundStartTime, playerIndex)
            if (playerActionTime is None):
                print('no move time for {}!'.format(gamesFolder))
                continue
            if ((playerActionTime-roundStartTime).seconds>MAX_SECONDS_FOR_ROUNDS):
                print('action took to much time, dont analyze')
                continue
            playerAction, oppAction = getPlayersActions(roundElm, playerIndex)
            if (len(gamesFolder.files('points.txt'))>0):
                pointsFile = gamesFolder.files('points.txt')[0]
            else:
                pointsFile = gamesFolder.files('Points*.txt')[0]
#             if (withInterpolation):
#                 interpolationFile = gamesFolder.files('interpolation*.txt')[0]
#             (_, _, errLines, _) = utils.readCSVFile(pointsFile, delimiter='\t', useNP=False, dtype=None, startRowIndex=2, doPrint=False)
#             dataFile = interpolationFile if withInterpolation else pointsFile
            (featuresData, _, errLines, timeData) = utils.readCSVFile(pointsFile, delimiter='\t', useNP=False, dtype=None, startRowIndex=2, doPrint=False)
            if (errLines <= MAX_ZEROS_POINTS):
#                 roundStartIndex, actionIndex = findActionIndices(gameRound, timeData, withInterpolation)
                actionIndex = findActionIndices(playerActionTime, timeData,
                    roundStartTime, roundEndTime, withInterpolation)
                try:
                    data = fe.convertRawDataToMatrix(featuresData[0:actionIndex])
                except:
                    print('missing file %s!!!' % pointsFile)
                frameRate = len(data) / utils.timeDiff(playerActionTime, roundStartTime)
                if (duplicatePlayer):
                    gameID = gameID + lastGameID + 1
                db.addRound(data, subjectID, gameID, roundID, index, playerAction, oppAction, frameRate)
                featuresCalculator = fe.FEPointsDistances(data, groups, calcMainCentroids=True) 
                db.saveMainCentroids(featuresCalculator.mainCentroids, tc.GROUP_RAW_DATA, subjectID, gameID, roundID)                
                index += 1
            else:
                print("Too many err lines in %s (%d)"%(pointsFile,errLines))
    db.close()    

def isPlayerIsDuplicates(name, duplicates):
    for name1,name2 in duplicates:
        if (name==name2):
            return True
    return False

def getOriginalPlayerName(name,duplicates):
    for name1,name2 in duplicates:
        if (name==name2):
            return name1
    return name

def getLastGameIDForDupPlayer(name,duplicates,rootFolder):
    for name1,name2 in duplicates:
        if (name==name2):
            games1 = utils.sortFolders(path('{}/{}'.format(rootFolder,name1)).dirs())
            lastGameID = int(games1[-1].name[4:6])
            return lastGameID+1
    raise Exception('cant find dup player!')

def mergeDuplicatesFolders(playerFolder,duplicateFolders):
    for (folder1,folder2) in duplicateFolders:
        if (not utils.dirExists(path('{}/{}'.format(playerFolder,folder2)))): continue
        games1 = utils.sortFolders(path('{}/{}'.format(playerFolder,folder1)).dirs())
        games2 = utils.sortFolders(path('{}/{}'.format(playerFolder,folder2)).dirs())
        lastGameID = int(games1[-1].name[4:6])
        for game in games2:
            gameID,roundID = int(game.name[4:6]), int(game.name[11:13]) 
            newFolderName = 'Game%02dRound%02d'%(gameID+lastGameID+1,roundID)
            utils.copyFolder(game, '{}/{}/{}'.format(playerFolder,folder1,newFolderName))
#         utils.deleteFolder(games2)

def loadAU(folder, groups, dataGroup, featuresType):
    import actionUnits as au
    db = tc.Tables(tc.HDF5_AU_FILE_NAME, read=False)
    db.createGamesTable()    
    index = 0
    folders = utils.sortFolders(path(folder).dirs())
    for subjectID, subFolder in enumerate(folders):
        playerName = subFolder.name
        print('%d/%d: %s' % (subjectID + 1, len(folders), playerName))
        summary = utils.parseXMLFile(subFolder + '//' + SUMMARY_FILE_NAME)
        db.addPlayer(subjectID, playerName) 
        studentFolders = subFolder.dirs('Game*')
        studentFolders.sort()
        for gamesFolder in studentFolders:
            gameID, roundID = utils.getGameAndRoundID(gamesFolder.name)
            gameRound  = Round.xmlToRound(summary, playerName, gameID, roundID)
            AUFiles = gamesFolder.files('AU Points.txt')
            if (len(AUFiles)==1): 
                AUFile = AUFiles[0]
            else:
                print('No AU Points.txt file!')
                continue    
            pointsFile = gamesFolder.files('Points*.txt')[0]
            (featuresData,  errLines) = au.readAUFile(AUFile)
            (_, _, _, timeData) = utils.readCSVFile(pointsFile, delimiter='\t', useNP=False, dtype=None, startRowIndex=2, doPrint=False) 
            if (errLines <= MAX_ZEROS_POINTS):
#                 roundStartIndex, actionIndex = findActionIndices(gameRound, timeData, False)
                roundStartIndex, actionIndex = findActionIndices(gameRound.playerActionTime, timeData, gameRound.roundStartTime, False)
                data = featuresData[roundStartIndex:actionIndex]
                db.addRound(data, subjectID, gameID, roundID, index, gameRound.playerActionCode, gameRound.oppActionCode,0)
                index += 1
#             else:
#                 print("Too many err lines in %s (%d)"%(AUFile,errLines))
    db.close()                    

def calcAUFeaturesFromDB(groups, dataGroup, featuresType, readOnly=False):
    db = tc.Tables(tc.HDF5_AU_FILE_NAME, read=True)
    if (not readOnly): db.removeFeatures(dataGroup, featuresType, False)
    for dataTable, index, subjectID, gameID, roundID in db.getTables(dataGroup):
#        subject = db.playerName(subjectID)
        data = dataTable[:]
        if (utils.nanExist(data)):
            raise Exception('nan exists!!!')
        cov = np.cov(data.T)
        features = MLUtils.halfMat(cov, True)
        db.addFeatures(features, dataGroup, index, featuresType, False) 
        if (not readOnly): db.addFeatures(features, dataGroup, index, featuresType, False)
        if (index % 10 == 0): print('%d out of %d' % (index, db.gamesLen))
    db.close()


def calcAUPostFeatures(dataGroup, featuresType, compNum=10, polarFeaturesCalc=True):
    db = tc.Tables(tc.HDF5_AU_FILE_NAME, read=True)
    x = db.getFeatures(dataGroup, featuresType, polarFeaturesCalc)
    x = PCA.transform(x, compNum, doPrint=True)
    db.writePostFeatures(x, dataGroup, featuresType, compNum)
    db.close()


def copyFiles(folder,destFolder):
#     utils.deleteDirectory(LOCAL_FOLDER)
    playersFolders = utils.sortFolders(path(folder).dirs()) 
    for playerID, playerFolder in enumerate(playersFolders):
#         utils.createDirectory('%s/%s'%(LOCAL_FOLDER,playerFolder.name))
#         utils.copyFile(playerFolder,'Summary.Final.xml', '%s/%s/%s'%(LOCAL_FOLDER,playerFolder.name,'Summary.Final.xml'))
        playerName = playerFolder.name
        print('%s %d-%d'%(playerName, playerID+1,len(playersFolders)))
        for gameFolder in sorted(playerFolder.dirs('Game*')):
#             utils.createDirectory('%s/%s/%s'%(LOCAL_FOLDER,playerFolder.name,gameFolder.name))
            gameDestFolder = '%s/%s/%s'%(destFolder,playerName,gameFolder.name)
            utils.copyFile(gameFolder, gameDestFolder, 'AU Points.txt')    


def createCouplesTable(folder, dbName=tc.HDF5_FILE_NAME, badSubjects=[],duplicates=[]):
    folders = utils.sortFolders(path(folder).dirs()) 
    db = tc.Tables(dbName, read=True)
    db.removePlayersCouples()
    for subFolder in folders:
        orgPlayerName = subFolder.name
        playerName = fixPlayerNames(subFolder.name)
        if (playerName=='vladimir'):
            print("sdf")
        if (playerName in badSubjects):
            print('{} is a bad subject').format(playerName)
            continue        
        studentFolders = subFolder.dirs('Game*')
        studentFolders.sort()
        xmlFile = [file + '//' + SUMMARY_FILE_NAME for file in [studentFolders[-1],subFolder] if utils.fileExists(file + '//' + SUMMARY_FILE_NAME)][0]
        summary = utils.parseXMLFile(xmlFile)
        opponentName = getOppNameFromSummary(summary, playerName)
        playerName = getOriginalPlayerName(playerName, duplicates)
        opponentName = fixPlayerNames(getOriginalPlayerName(opponentName, duplicates))
        db.addPlayersCouple(playerName, opponentName)
    db.close()

def fixPlayerNameInXML(name):
    if (name=='vadim'): return 'Vadim'
    else: return name
    
def getOppNameFromSummary(summary, playerName):
    #summary.xpath('//Player[not(@Name="%s")]' % (orgPlayerName))
    player1 = fixPlayerNames(fixPlayerNameInXML(summary.xpath('//Player[(@Index="1")]')[0].get('Name')))
    player2 = fixPlayerNames(fixPlayerNameInXML(summary.xpath('//Player[(@Index="2")]')[0].get('Name')))
    if (player1==playerName):
        return player2
    elif (player2==playerName):
        return player1
    else:
        raise Exception('cant find opp name in summary!')
    

def loadDemographic(folder, removeOldTable=False):
    folders = utils.sortFolders(path(folder).dirs()) 
    db = tc.Tables(tc.HDF5_FILE_NAME, read=True)
    if (removeOldTable): db.removeDemographic()
    for subjectID, subFolder in enumerate(folders):
        playerName = subFolder.name
        print('%d/%d: %s' % (subjectID + 1, len(folders), playerName))
        summary = utils.parseXMLFile(subFolder + '//' + SUMMARY_FILE_NAME)
        playerDemList = summary.xpath('//Player[@Name="%s"]/Demographics' % (playerName))
        if (len(playerDemList) == 0):
            print('player %s has no demographics!' % playerName)
        else:
            db.addPlayerDemographic(playerName, playerDemList[0])
    db.close()

def demographicSummary():
    db = tc.Tables(tc.HDF5_FILE_NAME, read=True)
    demo = db.getDemographics()
    print(utils.count(demo['age']))
    print('min: %d, max: %d' % (min(demo['age']), max(demo['age'])))
    print(utils.count(demo['gender']))
    db.close()

def calcFeaturesFromDB(groups, dataGroup, featuresType, centroidGroupType, k=10, polarFeaturesCalc=True, readOnly=False):
    db = tc.Tables(tc.HDF5_FILE_NAME, read=True)
    if (not readOnly): db.removeFeatures(dataGroup, featuresType, polarFeaturesCalc)
    for dataTable, index, subjectID, gameID, roundID in db.getTables(dataGroup):
#        subject = db.playerName(subjectID)
        data = dataTable[:]
        calcMainCentroids = dataGroup in [tc.GROUP_KALMAN, tc.GROUP_RAW_DATA]
        mainCentroids = db.loadMainCentroids(centroidGroupType, subjectID, gameID, roundID) if (not calcMainCentroids) else []
        featuresCalculator = fe.FEPointsDistances(data, groups, calcMainCentroids=calcMainCentroids, mainCentroids=mainCentroids)
        if (calcMainCentroids): 
            if (not readOnly): db.saveMainCentroids(featuresCalculator.mainCentroids, dataGroup, subjectID, gameID, roundID)
        else:
            featuresCalculator.loadMainCentroids(mainCentroids)
        parameteriseData = data if dataGroup == tc.GROUP_PARAMETRISE else []
        dists = featuresCalculator.calcPointsDistanceFromNose(parameteriation=dataGroup == tc.GROUP_PARAMETRISE, features=parameteriseData, polar=polarFeaturesCalc, doPlot=False)
        if (featuresType == tc.FEATURES_COVA):
            features = calcDistancesCovaFromDists(dists)
        elif (featuresType == tc.FEATURES_SUMS):
            features = calcDistanceSums(dists, k)
        elif (featuresType == tc.FEATURES_REAL_COV):
            features = calcDistancesRealCovFromDists(dists)
        elif (featuresType == tc.FEATURES_MOVEMENT):
            features = calcMovementMeasure(dists)
        elif (featuresType == tc.FEATURES_COSINE):
            features = calcDistancesCosineSimilartiy(dists)
        if (not readOnly): db.addFeatures(features, dataGroup, index, featuresType, polarFeaturesCalc)
        if (index % 10 == 0): print('%d out of %d' % (index, db.gamesLen))
    db.close()

def calcFeaturesFromParametrisedDB(groups):
    db = tc.Tables(tc.HDF5_FILE_NAME, read=True)
    db.removeFeatures(tc.GROUP_PARAMETRISE)
    for dataTable, dataParamTable, index, subjectID, gameID, roundID in db.get2Tables(tc.GROUP_KALMAN, tc.GROUP_PARAMETRISE):
#        subject = db.playerName(subjectID)
        dataNorm = dataTable[:]
        dataParam = dataParamTable[:]
        featuresCalculator = fe.FEPointsDistances(dataNorm, groups, normalizeData=False, smoothData=False, parametrizedData=False, doPlotPoints=False, convertRawData=False)
        dists = featuresCalculator.calcPointsDistanceFromNose(parameteriation=True, features=dataParam, doPlot=False)
        features = calcDistancesCovaFromDists(dists)
        db.addFeatures(features, tc.GROUP_PARAMETRISE, subjectID, gameID, roundID, index)
        if (index % 10 == 0): print('%d out of %d' % (index, db.gamesLen))
    db.close() 
  
def writeKalmanPointsToDB(groups, noiseSTD=1):
    db = tc.Tables(tc.HDF5_FILE_NAME, read=True)
    for dataTable, index, subjectID, gameID, roundID in db.getTables(tc.GROUP_RAW_DATA):
        data = dataTable[:]
        featuresCalculator = fe.FEPointsDistances(data, groups, smoothData=True, pointsNoiseSTD=noiseSTD, calcMainCentroids=True) 
        db.saveMainCentroids(featuresCalculator.mainCentroids, tc.GROUP_KALMAN, subjectID, gameID, roundID)
        db.addDataTable(featuresCalculator.data, subjectID, gameID, roundID, tc.GROUP_KALMAN)
        if (index % 10 == 0): print('%d out of %d' % (index, db.gamesLen))
    db.close()

def normalizePointsInDB(groups, groupName=tc.GROUP_KALMAN):
    db = tc.Tables(tc.HDF5_FILE_NAME, read=True)
    for dataTable, index, subjectID, gameID, roundID in db.getTables(groupName):
        data = dataTable[:]
        mainCentroids = db.loadMainCentroids(groupName, subjectID, gameID, roundID)
        data = fe.FEPointsDistances(data, groups, normalizeData=True, mainCentroids=mainCentroids).data
        db.addDataTable(data, subjectID, gameID, roundID, tc.GROUP_NORM)
        if (index % 10 == 0): print('%d out of %d' % (index, db.gamesLen))
    db.close()

def parametrisePointsInDB(groups):
    db = tc.Tables(tc.HDF5_FILE_NAME, read=True)
    db.removeGroup(tc.GROUP_PARAMETRISE)
    for dataTable, index, subjectID, gameID, roundID in db.getTables(tc.GROUP_NORM):
        data = dataTable[:]
        data = fe.FEPointsDistances(data, groups, parametrizedData=True).features
        db.addDataTable(data, subjectID, gameID, roundID, tc.GROUP_PARAMETRISE)
        if (index % 10 == 0): print('%d out of %d' % (index, db.gamesLen))
    db.close()

def calcPostFeatures(dataGroup, featuresType, compNum=10, polarFeaturesCalc=True):
    db = tc.Tables(tc.HDF5_FILE_NAME, read=True)
    x = db.getFeatures(dataGroup, featuresType, polarFeaturesCalc)
    x = PCA.transform(x, compNum, doPrint=True)
    db.writePostFeatures(x, dataGroup, featuresType, compNum)
    db.close()

def calcLeaveFreq(window=10):
    db = tc.Tables(tc.HDF5_FILE_NAME, read=True)
    playersRounds = db.getPlayersRounds()
    leaves = {}
    players = 0
    for playerID, playerRounds in playersRounds.iteritems():
        players+=1
        for index in playerRounds:
            subjectHistoryLen = len(range(playerRounds[0], index))
            playersLeaveHistoryIndices = db.getPlayerActionHistoryIndices(playerID, utils.LEAVE_CODE, index)
            key = utils.kbin(subjectHistoryLen, window)
            if (not key in leaves): leaves[key]=0 
            leaves[key]+=len(playersLeaveHistoryIndices)
    db.close()
    x = range(min(leaves.keys()),max(leaves.keys()),window)
    y,cy=[],[] 
    for i in x:
        y.append(leaves[i]/float(players))
        cy.append(sum(y[:i+1]))
    plots.graph(x,cy)

def plotDistances(groups, dataGroup=tc.GROUP_NORM, featureIndex=4, ylabel='Distance from nose', title=''):
    db = tc.Tables(tc.HDF5_FILE_NAME, read=True)
    tables = db.getTables(dataGroup)
    dataTable, index, subjectID, gameID, roundID = next(tables)
    data = dataTable[:]
    mainCentroids = db.loadMainCentroids(tc.GROUP_KALMAN, subjectID, gameID, roundID)
    featuresCalculator = fe.FEPointsDistances(data, groups, calcMainCentroids=False, mainCentroids=mainCentroids,)
    featuresCalculator.plotPoints()
    featuresCalculator.loadMainCentroids(mainCentroids)
    parameteriseData = data if dataGroup == tc.GROUP_PARAMETRISE else []
    dists = featuresCalculator.calcPointsDistanceFromNose(parameteriation=dataGroup == tc.GROUP_PARAMETRISE, features=parameteriseData, doPlot=False)
    cov = np.cov(dists)
    plots.matShow(cov)
#    fig=plots.plt.figure(facecolor='white')
#    ax1 = plots.plt.axes()
#    ax1.axes.get_yaxis().set_visible(False)
#    ax1.axes.get_xaxis().set_visible(False)
##    plots.plt.hold(True)
#    plots.plt.plot(dists[:, :])
#    plots.plt.xlabel('time')
##    plots.plt.ylabel(ylabel)
#    plots.plt.xlim((0, dists.shape[0]))
#    plots.plt.ylim((-2,5))
##    plots.plt.title(title)
#    plots.plt.show()
    db.close()
    
def plotFeature(dataGroup=tc.GROUP_NORM, featuresType=tc.FEATURES_REAL_COV):
    db = tc.Tables(tc.HDF5_FILE_NAME, read=True)
    features = db.getFeatures(dataGroup, featuresType, polarFeaturesCalc=True)
    print (features.shape)
    plots.plt.plot(features[0, :])
    plots.plt.show()
    plots.plt.xlabel('features')
    plots.plt.ylabel('covariance')
    plots.plt.title('')
    db.close()

def readFaceToFaceResults(folder, groups):
    folders = utils.sortFolders(path(folder).dirs()) 
    fileIndex, facialFeaturesIndex = 0, 0
    game = None
    logger = Log()
    rewards = utils.load(REWARDS)
    print('Reading %d folders' % len(folders))
    for studentID, subFolder in enumerate(folders):
        print('%d/%d: %s (%.2f)' % (studentID + 1, len(folders), subFolder.name, rewards[subFolder.name]))
        summary = utils.parseXMLFile(subFolder + '//' + SUMMARY_FILE_NAME)
#         student = Subject(subFolder.name if subFolder.name.find('Results') == -1 else subFolder.name[:-7])
        playerIndex = getPlayerIndex(summary, student.name) 
        studentFolders = subFolder.dirs('Game*')
        studentFolders.sort()
        lastGameID = -1
        studentRoundFacialIndex = 0
        for roundInd, gamesFolder in enumerate(studentFolders):
            gameID, roundID = utils.getGameAndRoundID(gamesFolder.name)
            roundElm = getRoundSummary(summary, gameID, roundID)
            roundStartTime = getRoundStartTime(roundElm)
            playerActionTime = getRoudPlayerActionTime(roundElm, roundStartTime, playerIndex)
            if (gameID != lastGameID):
                lastGameID = gameID
                if (game is not None): 
                    student.games.append(game)
                game = Game(gameID)
            pointsFile = gamesFolder.files('Points*.txt')[0]
            actions = utils.actionsFromFileName(pointsFile.name)
            playerAction = utils.typeStrToNum(actions[0])
            oppAction = utils.typeStrToNum(actions[1])
            if (playerAction == utils.LEAVE_CODE): 
                game.playerLeft = roundID
            if (oppAction == utils.LEAVE_CODE): 
                game.agentLeft = roundID  
            
            cov_features, features = calcPointsFeaturesFromFile(pointsFile, student.name, gameID, roundID, groups, roundStartTime, playerActionTime, 98)
            logger.namesIDS.append(studentID)
            if (len(cov_features) > 0):
#                f = np.hstack((studentID, f))
                logger.facialFeaturesIndices.append(fileIndex)
                logger.facialFeaturesList[0] = utils.arrAppend(logger.facialFeaturesList[0], cov_features)
#                student.facialFeatures = utils.arrAppend(student.facialFeatures,allF)
                student.facialFeaturesIndices.append(studentRoundFacialIndex)
                student.outerFacialFeaturesIndices.append(facialFeaturesIndex)
                if (playerAction == utils.LEAVE_CODE):
                    student.leaveFacialRoundsIndices.append(studentRoundFacialIndex)
                else:
                    student.stayFacialRoundsIndices.append(studentRoundFacialIndex)
                game.rounds.append(Round(roundID, gameID, roundInd, fileIndex, (playerActionTime - roundStartTime).microseconds, playerAction, oppAction, facialFeaturesIndex, studentRoundFacialIndex, True, [], []))
                studentRoundFacialIndex += 1
                facialFeaturesIndex += 1
#                featuresIndex = 1
#                for elmNum in np.linspace(maxElmsNum - 1, minElmsNum, maxElmsNum - minElmsNum):
#                    f = calcPointsFeaturesFromFile(pointsFile, groups, elmNum)
#                    logger.facialFeaturesList[featuresIndex] = utils.arrAppend(logger.facialFeaturesList[featuresIndex], f)
#                    featuresIndex += 1
            else:
                game.rounds.append(Round(roundID, gameID, roundInd, fileIndex, (playerActionTime - roundStartTime).microseconds, playerAction, oppAction, facialFeaturesIndex, studentRoundFacialIndex, False, [], []))
            logger.actions.append(playerAction)
            logger.files.append(pointsFile)            
            fileIndex += 1
        student.games.append(game)
        logger.subjects.append(student)
#        print('%s: %d games, %d rounds' % (student.name, len(student.games), len(student.rounds)))
        game = None
    return logger 

def readOneSubjectLogs(folder, groups, maxElmsNum=50, minElmsNum=50):
    folders = os.listdir(folder)
    fileIndex = 0
    game = None
    logger = Log(maxElmsNum - minElmsNum + 1)
    for studentID, subFolder in enumerate(folders): 
        student = Subject(subFolder)
#        if (student.name  in SUBJECTS_LESS_RECORDS): continue
        subFolder = utils.concatFolders(folder, subFolder)
        studentFolders = os.listdir(subFolder)
        studentFolders.sort()
        lastGameID = -1
        for folderIndex, gamesFolder in enumerate(studentFolders):
            gameID, roundID = utils.getGameAndRoundID(gamesFolder)
            if (gameID != lastGameID):
                lastGameID = gameID
                if (game is not None): 
                    student.games.append(game)
                game = Game(gameID)
            if (folderIndex < len(studentFolders) - 1):  # Check if this is not the last student's game
                _, nextRoundID = utils.getGameAndRoundID(studentFolders[folderIndex + 1])
            else:
                _, nextRoundID = 1, 1
            gamesFolder = utils.concatFolders(subFolder, gamesFolder)
            results = os.listdir(gamesFolder)
            resFiles = [resFile for resFile in results if utils.getFileType(resFile) == 'txt' and resFile.startswith((utils.STAY, utils.LEAVE))]
            pointsFiles = [resFile for resFile in results if utils.getFileType(resFile) == 'txt' and resFile.startswith('Points')]
            featuresFile = resFiles[0]   
            actionID = utils.typeStrToNum(utils.typeFromFileName(featuresFile))
            agentAction = utils.STAY_CODE
            if (actionID == utils.LEAVE_CODE): 
                game.playerLeft = roundID
            elif (nextRoundID == 1): 
                game.agentLeft = roundID
                agentAction = utils.LEAVE_CODE  

            if (len(pointsFiles) > 0): 
                featuresFile = utils.concatFolders(gamesFolder, featuresFile)
                pointsFile = utils.concatFolders(gamesFolder, pointsFiles[0])
#                f, timeElapsed = calcPointsFeaturesFromFile(pointsFile, groups, maxElmsNum)
                f, timeElapsed = calcPointsFeaturesFromFile(pointsFile, studentID, gameID, roundID, groups)
                if (f):
                    logger.facialFeaturesIndices.append(fileIndex)
                    logger.facialFeaturesList[0] = utils.arrAppend(logger.facialFeaturesList[0], f)
                    featuresIndex = 1
                    for elmNum in np.linspace(maxElmsNum - 1, minElmsNum, maxElmsNum - minElmsNum):
                        f = calcPointsFeaturesFromFile(pointsFile, groups, elmNum)
                        logger.facialFeaturesList[featuresIndex] = utils.arrAppend(logger.facialFeaturesList[featuresIndex], f)
                        featuresIndex += 1
            else:
                timeElapsed = -1
                print('No points file! %s' % (gamesFolder))
#            game.rounds.append(Round(roundID, gameID, fileIndex, timeElapsed, actionID, agentAction, f))
            game.rounds.append(Round(roundID, gameID, -1, fileIndex, None, actionID, agentAction, -1, -1, -1, f, None))
            logger.actions.append(actionID)
            logger.files.append(featuresFile)            
            fileIndex += 1
        student.games.append(game)
        logger.subjects.append(student)
#        print('%s: %d games, %d rounds' % (student.name, len(student.games), len(student.rounds)))
        game = None
    return logger
    
def calcPointsFeaturesFromFile(fileName, subject, gameID, roundID, groups, roundStartTime=None, playerActionTime=None, elmsNum=98, loadKalman=False):
    (featuresData, headData, errLines, timeData) = utils.readCSVFile(fileName, delimiter='\t', useNP=False, dtype=None, startRowIndex=2, doPrint=False)
    if (not loadKalman):
        roundStartIndex, actionIndex = findActionIndices(playerActionTime, timeData, roundStartTime)
    if (errLines <= MAX_ZEROS_POINTS):
        if (not loadKalman):
            cova, dists = calcDistancesCova(featuresData[roundStartIndex:actionIndex], subject, gameID, roundID, groups)  # , calcCentroids=True, elmsNum=elmsNum,fileName=fileName) 
        else:
            featuresData = loadKalmanFile(subject, gameID, roundID)  
            cova, dists = calcDistancesCova(featuresData, subject, gameID, roundID, groups)            
        return cova, dists 
    else:
        print('%d Zeros! %s' % (errLines, fileName))
        return [], []

def loadKalmanFile(subject, gameID, roundID):
    kalmanData = utils.loadMatlab('%s/%s_%d_%d_kalman' % (KALMAN_DATA_FOLDER, subject, gameID, roundID), True)
    data = kalmanData['ret']     
    return data

# def findActionIndices(gameRound, timeData, withInterpolation=False):
#     if (gameRound.playerActionTime is not None):
#         actionIndex = -1
#         roundStartIndex = -1
#         firstFrameTime = getFullFrameTime(timeData[0], gameRound.playerActionTime)
#         for ind, timeRec in enumerate(timeData):
#             frameTime = getFullFrameTime(timeRec, gameRound.playerActionTime)
#             frameAccTime = frameTime-firstFrameTime
#             if (frameAccTime >= gameRound.playerActionTimeDelta):
#                 print('pass!')
#             print(frameAccTime)
# # timedelta(frameTime,firstFrameTime)
#             if (frameTime > gameRound.roundStartTime and roundStartIndex == -1):
#                 roundStartIndex = ind 
#             if (frameTime > gameRound.playerActionTime and actionIndex == -1):
#                 actionIndex = ind 
#         if (actionIndex == -1):
#     #        print(fileName, 'action timediff after last frame:', (playerActionTime - frameTime))  # .microseconds / 1000)
#             actionIndex = len(timeData) 
#     else:
#         roundStartIndex = 0
#         actionIndex = len(timeData) 
#     return roundStartIndex, actionIndex

def getFullFrameTime(timeRec, playerActionTime):
    timeField = timeRec[2:-1]
    fullTime = '%s %s' % (playerActionTime.strftime('%Y-%m-%d'), timeField)
    frameTime = datetime.strptime(fullTime, '%Y-%m-%d %H:%M:%S.%f')  
    return frameTime


def findActionIndices(playerActionTime, timeData, roundStartTime,
                      roundEndTime, withInterpolation=False):
    if (utils.timeDiff(playerActionTime, roundEndTime) <= 0.001):
        return len(timeData)

#     firstTimeData = datetime.strptime(timeData[0][2:-1], '%H:%M:%S.%f')
#     shift = utils.timeToMicroseconds(firstTimeData) - \
#         utils.timeToMicroseconds(roundStartTime)
#     shift = utils.microsecondsShiftToDateDelta(shift)
#     if (firstTimeData - roundStartTime)
    playerActionShift = utils.microsecondsShiftToDateDelta(utils.timeToMicroseconds(playerActionTime))
    dt = (datetime.strptime(timeData[1][2:-1], '%H:%M:%S.%f') -
          datetime.strptime(timeData[0][2:-1], '%H:%M:%S.%f')).microseconds
    dt = utils.microsecondsShiftToDateDelta(dt)
    timeZero = datetime(year=1900, month=1, day=1)
    if (playerActionTime is not None):
        actionIndex = -1
#         roundStartIndex = -1
#         firstFrameTime = timeData[0]
#         firstFrameTime = firstFrameTime[:-1] if withInterpolation else firstFrameTime[2:-1]
#         firstFrameTime = datetime.strptime(firstFrameTime, '%H:%M:%S.%f')
#         playerActionDelta = playerActionTime - firstFrameTime
        for ind, timeRec in enumerate(timeData):
            frameTime = timeRec[:-1] if withInterpolation else timeRec[2:-1]
#             fullTime = '%s %s' % (playerActionTime.strftime('%Y-%m-%d'), timeField)
#             frameTime = datetime.strptime(timeField, '%Y-%m-%d %H:%M:%S.%f')  # 0:15:58:39.0161070
            frameTime = datetime.strptime(frameTime, '%H:%M:%S.%f')  # 0:15:58:39.0161070
#             frameTime -= shift
#             frameDelta = frameTime - firstFrameTime
#             delta = playerActionTime - frameTime
            if (frameTime - playerActionShift + dt >= timeZero):
#             if (delta.seconds == 0 and delta.microseconds <= dt):
                actionIndex = ind
                break
#             if (frameTime > roundStartTime and roundStartIndex == -1):
#                 roundStartIndex = ind 
#             if (frameTime > playerActionTime and actionIndex == -1):
#                 actionIndex = ind 
        if (actionIndex == -1):
            if (frameTime - playerActionShift + 2 * dt >= timeZero):
                actionIndex = len(timeData)
            else:
                raise Exception('player action index not found!')
    else:
        raise Exception('playerActionTime is None!')
#         roundStartIndex = 0
#         actionIndex = len(timeData)
#     if (actionIndex < len(timeData) - 2):
#         print('subject pressed before the round ended.')
    return actionIndex


def calcDistanceAndAnglesCova(data, headData, groups, groupNum=0, calcCentroids=False):
    (dists, angles) = fe.FEPointsDistances(data, groups).calcDistancesFronHeadTimeSeries(headData, groupNum, calcCentroids)      
    distsCova = MLUtils.calcCova(dists)
    distsCova = MLUtils.halfMat(distsCova, True)
    return distsCova  
#    return np.hstack((distsCova, angsCova))
    
def calcDistancesCova(data, subject, gameID, roundID, groups, groupNum=0, calcCentroids=True, elmsNum=50, fileName=''):
    dists = fe.FEPointsDistances(data, groups, subject, gameID, roundID, True, True, False, True, convertRawData=True).calcPointsDistanceFromNose(polar=False)
    distsCova = calcMovementMeasure(dists)  # calcDistancesCovaFromDists(dists)
    return distsCova, dists

def calcDistanceSums(dists, k):
    t = int(dists.shape[0] / k)
    features = []
    for i in range(k):
        features = np.hstack((features, sum(dists[t * i:t]))) if i < (k - 1) else np.hstack((features, sum(dists[t * (k - 1):])))  
    return features

def calcDistancesCovaFromDists(dists):
    distsCova = MLUtils.calcCova(dists.T)
#    distsCova = np.cov(dists.T)
    features = MLUtils.halfMat(distsCova, True)
#    distsCova1 = MLUtils.calcCova(dists[:t].T, elmsNum)
#    distsCova2 = MLUtils.calcCova(dists[t:t*2].T, elmsNum)
#    distsCova3 = MLUtils.calcCova(dists[t*2:].T, elmsNum)
#    distsCova1 = MLUtils.halfMat(distsCova1, True)
#    distsCova2 = MLUtils.halfMat(distsCova2, True)
#    distsCova3 = MLUtils.halfMat(distsCova3, True)
#    distsCova = np.hstack((distsCova1,distsCova2,distsCova3))
#    distsCova = np.cov(dists.T) 
#    distsCovaEigVals = np.real(scipy.linalg.eigvals(distsCova))
#    distsCovaMaxEigVals = utils.maxN(distsCovaEigVals, 10)
#    return ret,[]
    return features

def calcDistancesRealCovFromDists(dists, lastFrams= -1):
    if (lastFrams == -1):
        distsCova = np.cov(dists.T)
    else:
        distsCova = np.cov(dists[-lastFrams:].T)
    features = MLUtils.halfMat(distsCova, True)
    return features

def calcDistancesCosineSimilartiy(dists):
    return MLUtils.cosineDistanceFlatMatrix(dists)
    
def calcMovementMeasure(dists):
    return np.sum(np.mean(abs(np.diff(dists[:, 4:].T)), 1))

### Delete those ugly functions!
def getRoundSummary(summary, gameID, RoundID):
    gameElm = summary.xpath('//Game[@Index="%d"]' % (gameID - 1))[0]
    roundElm = gameElm.xpath('Round[@Index="%d"]' % (RoundID - 1))[0]
    return roundElm


def getRoundStartTime(roundElm):
    startTime = roundElm.get('StartTime')
    return datetime.strptime(startTime, '%Y-%m-%d %H:%M:%S.%f')


def getRoundEndTime(roundElm):
    startTime = roundElm.get('EndTime')
    return datetime.strptime(startTime, '%Y-%m-%d %H:%M:%S.%f')


def getRoudPlayerActionTime(roundElm, roundStartTime, playerIndex):
    try:
        actionElm = roundElm.xpath('Action[@PlayerIndex="%d"]' % (playerIndex))[0]
        actionTime = datetime.strptime(actionElm.get('Time'), '%H:%M:%S.%f') 
        actionTimeDelta = timedelta(milliseconds=utils.timeToMiliseconds(actionTime))
        actionFullTime = roundStartTime + actionTimeDelta
        return actionFullTime
    except:
        return None

def getPlayersActions(roundElm, playerIndex):
        actionElm = roundElm.xpath('Action[@PlayerIndex="%d"]' % (playerIndex))[0]
        playerAction = utils.actionNameToCode(actionElm.get('Move'))
        opIndex = 1 if playerIndex==2 else 1
        opActionElm = roundElm.xpath('Action[@PlayerIndex="%d"]' % (opIndex))[0]
        opAction = utils.actionNameToCode(opActionElm.get('Move'))
        return (playerAction, opAction)
    
def fixPlayerNames(name):    
    if (name==u'שקד ארנפלד'.encode('utf8') or name.encode('utf8')==u'◊©◊ß◊ì ◊ê◊®◊†◊§◊ú◊ì'): return 'shaked'
    elif (name==u'ולדימיר נקריטין'.encode('utf8') or name.encode('utf8')==u'◊ï◊ú◊ì◊ô◊û◊ô◊® ◊†◊ß◊®◊ô◊ò◊ô◊ü'): return 'vladimir'
#     elif (name=='Laura Gaspar2'): return 'Laura Gaspar'
#     elif (name=='Boaz Hamenachem 2'): return 'Boaz Hamenachem'
    else: return name
