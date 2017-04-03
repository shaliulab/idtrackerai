# Import standard libraries
import os
from os.path import isdir, isfile
from glob import glob
import sys
import numpy as np
import cPickle as pickle

# Import third party libraries
from pprint import pprint

# sns.set(style="darkgrid")

# Import application/library specifics
sys.path.append('IdTrackerDeep/utils')
sys.path.append('IdTrackerDeep/CNN')

from idTrainer import *
from input_data_cnn import *
from py_utils import *

class P1B1plotter(object):

    def getListsFromPath(self):

        # get folder
        CNN_modelsPath = os.path.dirname(self.dictPath)
        self.condition = CNN_modelsPath.split('_models')[-1]
        print CNN_modelsPath
        print 'Condition, ', self.condition

        # get groupSizesCNN
        self.groupSizesCNN= sorted([int(a.split('/')[-2].split('_')[-1]) for a in glob.glob(CNN_modelsPath + "/*/")])
        self.numGroupsCNN = len(self.groupSizesCNN)
        print 'Group Sizes CNN, ', self.groupSizesCNN

        # get groupSizes
        self.groupSizes = sorted([int(a.split('/')[-2].split('_')[-1]) for a in glob.glob(glob.glob(CNN_modelsPath + "/*/")[0]+"*/")])
        self.numGroups = len(self.groupSizes)
        print 'Group Sizes ', self.groupSizes

        # get IMDBSizes
        self.IMDBSizes = sorted([int(a.split('/')[-2].split('_')[-1]) for a in glob.glob(glob.glob(glob.glob(CNN_modelsPath + "/*/")[0]+"*/")[0]+"*/")])
        self.numIMDBSizes = len(self.IMDBSizes)
        print 'IMDB Sizes, ', self.IMDBSizes

        # get repetitions
        self.repList = sorted([int(a.split('/')[-2].split('_')[-1]) for a in glob.glob(glob.glob(glob.glob(glob.glob(CNN_modelsPath + "/*/")[0]+"*/")[0]+"*/")[0]+"*/")])
        self.numRepetitions = len(self.repList)
        print 'Repetitions, ', self.repList

    def initializeDicts(self):
        # Main loop
        self.LossAccDicts = {}
        for gCNN in self.groupSizesCNN:
            self.LossAccDicts[gCNN] = {}

            for g in self.groupSizes:
                self.LossAccDicts[gCNN][g] = {}

                for n in self.IMDBSizes:
                    self.LossAccDicts[gCNN][g][n] = []

        # Initialize figure arrays
        self.trainAccs = np.ones((self.numIMDBSizes, self.numGroups, self.numGroupsCNN, self.numRepetitions)) * np.nan
        self.valAccs = np.ones((self.numIMDBSizes, self.numGroups, self.numGroupsCNN, self.numRepetitions)) * np.nan
        self.testAccs = np.ones((self.numIMDBSizes, self.numGroups, self.numGroupsCNN, self.numRepetitions)) * np.nan

    def rebuildDictFromFolders(self):

        self.getListsFromPath()
        self.initializeDicts()

        # Main loop
        for gCNN, groupSizeCNN in enumerate(self.groupSizesCNN): # Group size of the pre trained CNN model

            for g, groupSize in enumerate(self.groupSizes): # Group size for the current training

                for n, numImForTrain in enumerate(self.IMDBSizes): # Number of images/individual for training

                    for r, repetition in enumerate(self.repList): # Repetitions

                        print '\nGroupSizeCNN, ', groupSizeCNN
                        print 'Group size, ', groupSize
                        print 'numImForTrain, ', numImForTrain
                        print 'Repetition, ', repetition

                        lossAccPath = 'IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/CNN_%i/numIndiv_%i/numImages_%i/rep_%i/model/lossAcc.pkl' %(self.condition, groupSizeCNN, groupSize, numImForTrain, repetition)
                        print 'loading, ', lossAccPath
                        lossAccDict = pickle.load(open(lossAccPath,'rb'))

                        self.LossAccDicts[groupSizeCNN][groupSize][numImForTrain].append(lossAccDict)
                        self.trainAccs[n,g,gCNN,r] = lossAccDict['acc'][-1]
                        self.valAccs[n,g,gCNN,r] = lossAccDict['valAcc'][-1]
                        self.testAccs[n,g,gCNN,r] = lossAccDict['testAcc'][-1]

                        print '\nSaving dictionary...'
                        pickle.dump(self.__dict__,open('IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/P1B1Dict.pkl' %self.condition,'wb'))
                        print 'Dictionary saved in ', 'IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/P1B1Dict.pkl' %self.condition

                        print '******************************************************************************************************************************'
                        print '******************************************************************************************************************************\n'


    def computeTimes(self,accTh = 0.83):
        self.accTh = accTh
        self.totalTime = np.ones((self.numIMDBSizes, self.numGroups, self.numGroupsCNN, self.numRepetitions)) * np.nan
        self.epochTime = np.ones((self.numIMDBSizes, self.numGroups, self.numGroupsCNN, self.numRepetitions)) * np.nan
        self.totalEpochs = np.ones((self.numIMDBSizes, self.numGroups, self.numGroupsCNN, self.numRepetitions)) * np.nan
        self.timeToAcc = np.ones((self.numIMDBSizes, self.numGroups, self.numGroupsCNN, self.numRepetitions)) * np.nan
        self.epochsToAcc = np.ones((self.numIMDBSizes, self.numGroups, self.numGroupsCNN, self.numRepetitions)) * np.nan

        for gCNN, groupSizeCNN in enumerate(self.groupSizesCNN):
            for g,groupSize in enumerate(self.groupSizes):
                for n,IMDBSize in enumerate(self.IMDBSizes):
                    print 'Group size CNN %i Group size %i IMDB size %i' %(groupSizeCNN,groupSize,IMDBSize)
                    for r, lossAccDict in enumerate(self.LossAccDicts[groupSizeCNN][groupSize][IMDBSize]):
                        self.totalTime[n,g,gCNN,r] = np.sum(lossAccDict['epochTime'])
                        self.epochTime[n,g,gCNN,r] = np.mean(lossAccDict['epochTime'])
                        self.totalEpochs[n,g,gCNN,r] = len(lossAccDict['epochTime'])
                        if np.where(np.asarray(lossAccDict['valAcc'])>=accTh)[0].any():

                            self.epochsToAcc[n,g,gCNN,r] = np.where(np.asarray(lossAccDict['valAcc'])>=accTh)[0][0]+1
                            self.timeToAcc[n,g,gCNN,r] = np.sum(lossAccDict['epochTime'][:int(self.epochsToAcc[n,g,gCNN,r])])

    def plotArray(self,arrayName,ax,ylim = (),ylabel='',title='',lineStyle='-'):
        import seaborn as sns
        from matplotlib import pyplot as plt
        from cycler import cycler
        colormap = plt.cm.jet
        colors = [colormap(i) for i in np.linspace(0, 0.9, 8)]
        ax.set_prop_cycle(cycler('color',colors))
        sns.set(style="white")
        array = getattr(self, arrayName)
        arrayMedian = np.nanmean(array,axis = 3)

        ax.plot(self.groupSizes, np.squeeze(arrayMedian).T,lineStyle)
        if arrayName == 'valAccs':
            ax.axhline(y=self.accTh,xmin=0,xmax=30./self.groupSizes[-2],c='k')
            ax.axhline(y=self.accTh,xmin=30./self.groupSizes[-2],xmax=1,c='k',linestyle='dashed')
        ax.set_xlabel('Group size')
        if ylim:
            ax.set_ylim(ylim)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_xticks(self.groupSizes)
        ax.set_xticklabels(self.groupSizes)
        if arrayName == 'valAccs':
            ax.legend(self.IMDBSizes,title='Images/individual',loc=0)
        if title:
            ax.set_title(title)

    def plotResults(self):
        import seaborn as sns
        from matplotlib import pyplot as plt
        sns.set(style="white")

        fig, axarr = plt.subplots(2,2,sharex=True, figsize=(20, 10), dpi=300)
        if self.condition == 'S':
            fig.suptitle('Training from strach with uncorrelated images')
        if self.condition == 'SD':
            fig.suptitle('Training from strach with uncorrelated images with dropout of .7')
        elif self.condition == 'SC':
            fig.suptitle('Training from scratch with correlated images')
        elif self.condition == 'SCV':
            fig.suptitle('Training from scratch with correlated images stopping like the video')
        elif self.condition == 'SCA':
            fig.suptitle('Training from scratch with correlated images and data augmentation')
        elif self.condition == 'SCAV':
            fig.suptitle('Training from scratch with correlated images and data augmentation stopping like the video')
        elif self.condition == 'KTCV':
            fig.suptitle('Training with knowledge transfer with correlated images stopping like the video')
        elif self.condition == 'KTCVA':
            fig.suptitle('Training with knowledge transfer with correlated images stopping like the video')
        elif self.condition == 'KTCVX':
            fig.suptitle('Training with knowledge transfer with correlated images stopping like the video only the softmax')
        elif self.condition == 'KTCVFX':
            fig.suptitle('Training with knowledge transfer with correlated images stopping like the video softmax and fully connected')
        # fig.suptitle('Training from strach (group size vs number of images per individual)')
        self.plotArray('valAccs',axarr[0,0],ylim=(0.,1.),ylabel='Accuracy',title = 'Accuracy in validation')
        self.plotArray('testAccs',axarr[0,0],lineStyle=':')
        # self.plotArray2('epochsToAcc',axarr[0,1],ylim=(0,400),ylabel='Number of pochs',title = 'Number of epochs to accuracy threshold %.2f' %self.accTh)
        self.plotArray('timeToAcc',axarr[1,0],ylabel='Time (sec)',title = 'Time to accuracy threshold  %.2f (sec)' %self.accTh)
        self.plotArray('totalTime',axarr[1,1],ylabel='Time (sec)',title = 'Total time in (sec)')
        self.plotArray('epochTime',axarr[0,1],ylabel='Time (sec)',title = 'Single epoch time in (sec)')
        # self.plotArray2('totalEpochs',axarr[1,0],ylim=(0,800),ylabel='Number of epochs',title = 'Total number of epochs')

        figname = 'IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/P1B1_results%s.pdf' %(self.condition, self.condition)
        print 'Saving figure'
        fig.savefig(figname)
        print 'Figure saved'



if __name__ == '__main__':
    '''
    argv[1]: condition: 'S'-scratch, 'KT'-knowledgeT, 'KTC'-knowledgeTCorrelated, 'ALL'
    P1B1plotter.py KTC
    '''
    condition = sys.argv[1]
    p = P1B1plotter()
    P1B1InfoPath = 'IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/info.pkl' %sys.argv[1]
    P1B1DictPath = 'IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/P1B1Dict.pkl' %sys.argv[1]
    print 'P1B1InfoPath, ', P1B1InfoPath
    p.dictPath = P1B1InfoPath
    if os.path.isfile(P1B1DictPath):
        print 'P1B1Dict.pkl already exist'
        rebuildDict = raw_input('Do you wanna rebuild P1B1Dict?')
    p.rebuildDictFromFolders()
    p.computeTimes()
    p.plotResults()
