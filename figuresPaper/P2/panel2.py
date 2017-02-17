from __future__ import division

import sys
sys.path.append('../../utils')

from py_utils import *
from P2utils import *

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import pandas as pd
import time
import cPickle as pickle
import seaborn as sns
sns.set(style='white')
import pyautogui

class panel2(object):

    def __init__(self, savePath = './', accStepsToPlot = [0,1,-1], plotP2Flag=True, plotTrajFlag=True):

        def getLastSession(subfolders):
            if len(subfolders) == 0:
                lastIndex = 0
            else:
                subfolders = natural_sort(subfolders)[::-1]
                lastIndex = int(subfolders[0].split('_')[-1])
                subfolder =  subfolders[0]
            return subfolder

        #set path to save panel2
        self.savePath = savePath
        # get screen size
        self.screenW, self.screenH = np.asarray(pyautogui.size()) / 96
        # create panel figure
        self.fig = plt.figure(figsize =(8.27, 11.69))
        # retrieve info about video
        self.videoPath = selectFile()
        self.video = os.path.basename(self.videoPath)
        self.folder = os.path.dirname(self.videoPath)
        self.filename, self.extension = os.path.splitext(self.video)
        # go to CNN models
        self.CNNModels = self.folder + '/CNN_models'
        # read sessions and get the path of the last one
        self.sessions = glob.glob(self.CNNModels +"/*")
        self.sessionPath = getLastSession(self.sessions)
        self.accumPaths = natural_sort(glob.glob(self.sessionPath + '/AccumulationStep*'))
        # get path of session's dicts
        self.lossAccPath = self.sessionPath + '/lossAcc.pkl'
        self.statsPath = self.sessionPath + '/statistics.pkl'
        self.preprocPath = self.sessionPath + '/preprocparams.pkl'
        # load session's dicts
        self.portraits = loadFile(self.videoPath, 'portraits')
        videoInfo = loadFile(self.videoPath, 'videoInfo', hdfpkl='pkl')
        self.numAnimals = videoInfo['numAnimals']
        # read plot params
        self.accStepsToPlot = accStepsToPlot
        self.plotP2Flag = plotP2Flag
        self.plotTrajFlag = plotTrajFlag
        self.accumDicts = []
        # store paths to accumulation dictionaries
        if self.accStepsToPlot != []:
            self.accDictPaths = [self.accumPaths[i] + '/accumDict.pkl' for i in self.accStepsToPlot]
            self.trainDictPaths = [self.accumPaths[i] + '/trainDict.pkl' for i in self.accStepsToPlot]
            self.lossAccDictPaths = [self.accumPaths[i] + '/model/lossAcc.pkl' for i in self.accStepsToPlot]
            self.lossAccDicts = []

            for i, accInd in enumerate(self.lossAccDictPaths):
                lossAccDict = self.loadPickle(accInd)
                self.lossAccDicts.append(lossAccDict)

        # load accumDicts
        for accInd in self.accStepsToPlot:
            self.accumDicts.append(self.loadPickle(self.accDictPaths[accInd]))

    def loadPickle(self,path):
        return pickle.load(open(path, "rb"))

    def plotP2(self):
        self.P2Path = self.accDictPaths[-1]
        self.loadP2 = self.loadPickle(self.P2Path)
        self.P2 = self.loadP2['overallP2'][1:]
        axP2 = plt.subplot2grid((12, 6), (6, 0), colspan=6, rowspan=2)
        sns.despine(fig=self.fig, ax=axP2)
        axP2.plot(range(len(self.P2)),self.P2,'or-')

    def computeTraj(self):
        self.statistics = self.loadPickle(self.statsPath)
        allFragIds = self.statistics['fragmentIds']
        self.trajDict = idTrajectories(self.videoPath, self.sessionPath, allFragIds, self.portraits, self.numAnimals)

    def plotTraj(self,framesToPlot=[]):
        axT1 = plt.subplot2grid((12, 6), (8, 0), colspan=2, rowspan=4, projection='3d')
        axT2 = plt.subplot2grid((12, 6), (8, 2), colspan=2, rowspan=4, projection='3d')
        axT3 = plt.subplot2grid((12, 6), (8, 4), colspan=2, rowspan=4, projection='3d')
        plotTrajectories(self.trajDict, self.numAnimals, self.fig, axT1, axT2, axT3, framesToPlot, plotBoth=True)

    def plotAccFrags(self):
        # pos = [1,3,5]
        pos = [(0,0), (2,0),(4,0)]
        for i,accInd in enumerate(self.accStepsToPlot):
            trainDict = self.loadPickle(self.trainDictPaths[accInd])
            fragmentsDict = loadFile(self.videoPath, 'fragments', hdfpkl='pkl')
            usedIndivIntervals = trainDict['usedIndivIntervals']
            idUsedIntervals = trainDict['idUsedIntervals']
            idUsedIndivIntervals = zip(usedIndivIntervals,idUsedIntervals)
            axF = plt.subplot2grid((12, 6), pos[i], colspan=4, rowspan=2)
            plotFragments(self.accumDicts[i], fragmentsDict, self.portraits, idUsedIndivIntervals, self.numAnimals, axF)

    def plotAccuracyAccum(self):
        pos = [(0,4), (2,4),(4,4)]
        for i, d in enumerate(self.lossAccDicts):
            accPlot = d['acc']
            valAccPlot = d['valAcc']
            axA = plt.subplot2grid((12, 6), pos[i], colspan=3, rowspan=2)
            xlim = len(accPlot)
            plotAccLoss(accPlot, valAccPlot, axA, xlim)

    def plotLossAccum(self):
        pos = [(1,4), (3,4),(5,4)]
        for i, d in enumerate(self.lossAccDicts):
            lossPlot = d['loss']
            lossValPlot = d['valLoss']
            axL = plt.subplot2grid((12, 6), pos[i], colspan=3)
            xlim = len(lossPlot)
            ylim = np.max(lossValPlot)
            plotAccLoss(lossPlot, lossValPlot, axL, xlim, ylim)

    def show(self):
        plt.tight_layout()
        plt.show()

    def save(self):
        figPath = os.path.join(self.savePath,'panel2.pdf')

        self.fig.savefig(figPath)

    def generatePanel(self):
        self.plotP2()
        self.computeTraj()
        self.plotTraj([1000,1200])
        self.plotAccFrags()
        self.plotAccuracyAccum()
        # self.plotLossAccum()
        self.show()
        self.save()

p = panel2()
p.generatePanel()
