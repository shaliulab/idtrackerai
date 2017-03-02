# Import standard libraries
import os
from os.path import isdir, isfile
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

    def __init__(self, condition = 'S'):

        def getIMDBNameFromPath(IMDBPath):
            filename, extension = os.path.splitext(IMDBPath)
            IMDBName = '_'.join(filename.split('/')[-1].split('_')[:-1])

            return IMDBName

        # Job counter for condor
        self.job = job
        self.condition = condition
        # Figure parameters
        self.groupSizesCNN = map(int,groupSizesCNN.split('_'))
        self.numGroupsCNN = len(self.groupSizesCNN)
        self.groupSizes = [2, 5, 10, 25, 50]
        # self.groupSizes = [10, 25, 50]
        self.numGroups = len(self.groupSizes)
        self.repList = map(int,repList.split('_'))
        self.numRepetitions = len(self.repList)
        self.IMDBSizes = [20,50,100,250,500,750,1000,3000,23000] # Images for training
        # self.IMDBSizes = [20,50,100,250]
        self.numIMDBSizes = len(self.IMDBSizes)

        # Initialize figure arrays
        self.trainAccs = np.ones((self.numIMDBSizes, self.numGroups, self.numGroupsCNN, self.numRepetitions)) * np.nan
        self.valAccs = np.ones((self.numIMDBSizes, self.numGroups, self.numGroupsCNN, self.numRepetitions)) * np.nan

        # Set CNN training parameters
        self.batchSize = 250
        self.numEpochs = 5000
        self.lr = 0.01
        self.keep_prob = 1.0

        # Set flag to indicate knowledge transfer
        if 'KT' in self.condition:
            self.kt = True
        else:
            self.kt = False

        # Set flag to stop the training when it is not learning much
        if 'V' in self.condition:
            self.checkLearningFlag = True
        else:
            self.checkLearningFlag = False

        # Set flag to only train softmax
        if 'X' in self.condition:
            self.onlySoftmax = True
        else:
            self.onlySoftmax = False

        # Dataset from which to load the images for training
    	if IMDBPath == 'd':
    	    if self.kt == False:
                	IMDBPath = 'IdTrackerDeep/data/36dpf_60indiv_29754ImPerInd_curvaturePortrait_0.hdf5'
    	    elif self.kt == True:
                	IMDBPath = 'IdTrackerDeep/data/25dpf_60indiv_26142imperind_curvatureportrait2_0.hdf5'

            print 'Using default library, ', IMDBPath
        self.IMDBPath = IMDBPath
        self.IMDBName = getIMDBNameFromPath(self.IMDBPath)

        # Initialize dictionaries where the data is going to be stored
        self.initializeDicts()

    def initializeDicts(self):
        # Main loop
        self.IndivIndices = {}
        self.ImagesIndices = {}
        self.LossAccDicts = {}
        for gCNN in self.groupSizesCNN:
            self.IndivIndices[gCNN] = {}
            self.ImagesIndices[gCNN] = {}
            self.LossAccDicts[gCNN] = {}

            for g in self.groupSizes:
                self.IndivIndices[gCNN][g] = []
                self.ImagesIndices[gCNN][g] = {}
                self.LossAccDicts[gCNN][g] = {}

                for n in self.IMDBSizes:
                    self.ImagesIndices[gCNN][g][n] = []
                    self.LossAccDicts[gCNN][g][n] = []

    def joinDicts(self,P1B1Dict='IdTrackerDeep/figuresPaper/P1B1/CNN_modelsS/P1B1Dict_job_1.pkl'):
        P1B1DictsPaths = scanFolder(P1B1Dict)
        print 'P1B1DictsPaths, '
        pprint(P1B1DictsPaths)

        # Update dictionary
        repList = []
        trainAccs = []
        valAccs = []
        IndivIndices = {}
        ImagesIndices = {}
        LossAccDicts = {}
        print '\nUpdating repetitions...'
        for P1B1DictPath in P1B1DictsPaths:
            print P1B1DictPath
            P1B1Dict = pickle.load(open(P1B1DictPath,'rb'))

            print 'Appeding repList, trainAccs and valAccs'
            repList.append(P1B1Dict['repList'])
            trainAccs.append(P1B1Dict['trainAccs'])
            valAccs.append(P1B1Dict['valAccs'])

            print 'Updating IndivIndices, ImagesIndices and LossAccDicts'
            for gCNN in self.groupSizesCNN:
                if gCNN not in IndivIndices.keys():
                    IndivIndices[gCNN] = {}
                    ImagesIndices[gCNN] = {}
                    LossAccDicts[gCNN] = {}

                for g in self.groupSizes:
                    if g not in IndivIndices[gCNN].keys():
                        IndivIndices[gCNN][g] = []
                        ImagesIndices[gCNN][g] = {}
                        LossAccDicts[gCNN][g] = {}

                    for indivIndices in P1B1Dict['IndivIndices'][gCNN][g]:
                        IndivIndices[gCNN][g].append(indivIndices)

                    for n in self.IMDBSizes:
                        print 'Group size CNN %i Group size %i IMDB size %i' %(gCNN,g,n)
                        if n not in ImagesIndices[gCNN][g].keys():
                            print 'Initializing lists ImagesIndices, LossAccDicts for the new IMDBSize', n
                            ImagesIndices[gCNN][g][n] = []
                            LossAccDicts[gCNN][g][n] = []

                        for r, (imagesIndices, lossAccDict) in enumerate(zip(P1B1Dict['ImagesIndices'][gCNN][g][n],P1B1Dict['LossAccDicts'][gCNN][g][n])):
                            print 'Group size CNN %i Group size %i IMDB size %i REp %i' %(gCNN,g,n,r)
                            ImagesIndices[gCNN][g][n].append(imagesIndices)
                            LossAccDicts[gCNN][g][n].append(lossAccDict)

        # Update object
        self.repList = flatten(repList)
        self.numRepetitions = len(self.repList)
        print 'repList, ', self.repList
        print 'trainAccs shape, ',trainAccs[0].shape
        print 'valAccs shape, ', valAccs[0].shape
        print 'trainAccs shape, ',trainAccs[1].shape
        print 'valAccs shape, ', valAccs[1].shape
        self.trainAccs = np.concatenate(trainAccs,axis=3)
        self.valAccs = np.concatenate(valAccs,axis=3)

        print 'trainAccs, ', self.trainAccs
        print 'valAccs, ', self.valAccs
        self.IndivIndices = IndivIndices
        self.ImagesIndices = ImagesIndices
        self.LossAccDicts = LossAccDicts
        print 'Num of lossAccDicts', len(IndivIndices[gCNN][g]), len(ImagesIndices[gCNN][g][n]), len(LossAccDicts[gCNN][g][n])

        # Save dictionary
        print '\nSaving dictionary...'
        pickle.dump(self.__dict__,open('IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/P1B1Dict.pkl' %self.condition,'wb'))

    def plotArray(self,arrayName,ax,title):
        import seaborn as sns
        sns.set(style="white")
        array = getattr(self, arrayName)
        arrayMedian = np.nanmedian(array,axis = 3)

        sns.heatmap(arrayMedian[:,:,0], annot=True, ax=ax, cbar=False, fmt='.2f')
        ax.set_xlabel('Group size')
        ax.set_ylabel('Num im/indiv')
        ax.set_xticklabels(self.groupSizes)
        ax.set_yticklabels(self.IMDBSizes[::-1])
        ax.invert_yaxis()
        ax.set_title(title)

    def plotArray2(self,arrayName,ax,title,ylim,ylabel):
        import seaborn as sns
        from matplotlib import pyplot as plt
        from cycler import cycler
        colormap = plt.cm.jet
        colors = [colormap(i) for i in np.linspace(0, 0.9, len(self.IMDBSizes))]
        ax.set_prop_cycle(cycler('color',colors))
        sns.set(style="white")
        array = getattr(self, arrayName)
        arrayMedian = np.nanmedian(array[:-1],axis = 3)

        ax.plot(self.groupSizes, np.squeeze(arrayMedian).T)
        ax.set_xlabel('Group size')
        if ylim:
            ax.set_ylim(ylim)
        ax.set_ylabel(ylabel)
        ax.set_xticks(self.groupSizes)
        ax.set_xticklabels(self.groupSizes)
        if arrayName == 'valAccs':
            ax.legend(self.IMDBSizes,title='Images/individual')
        ax.set_title(title)


    def plotResults(self):
        import seaborn as sns
        from matplotlib import pyplot as plt
        sns.set(style="white")

        fig, axarr = plt.subplots(2,3,sharex=True,sharey=True, figsize=(20, 10), dpi=300)
        if self.condition == 'S':
            fig.suptitle('Scratch')
        elif self.condition == 'SC':
            fig.suptitle('Scratch with correlated images')
        elif self.condition == 'SCV':
            fig.suptitle('Scratch with correlated images stopping like tracker')
        elif self.condition == 'KT':
            fig.suptitle('Knowledge transfer (from CNN of 50 indiv 25000 images)')
        elif self.condition == 'KTC':
            fig.suptitle('Knowledge transfer with correlated images')
        elif self.condition == 'KTV':
            fig.suptitle('Knowledge transfer stopping like the tracker')
        elif self.condition == 'KTCV':
            fig.suptitle('Knowledge transfer with correlated images stopping like the tracker')
        elif self.condition == 'KTCXV':
            fig.suptitle('Knowledge transfer with correlated images stopping like the tracker only SoftMax')
        # fig.suptitle('Training from strach (group size vs number of images per individual)')
        self.plotArray('valAccs',axarr[0,0],'Accuracy in validation')
        self.plotArray('epochsToAcc',axarr[0,1],'Number of epochs to accuracy threshold %.2f' %self.accTh)
        self.plotArray('timeToAcc',axarr[0,2],'Time to accuracy threshold  %.2f (sec)' %self.accTh)
        self.plotArray('totalTime',axarr[1,1],'Total time in (sec)')
        self.plotArray('epochTime',axarr[1,2],'Single epoch time in (sec)')
        self.plotArray('totalEpochs',axarr[1,0],'Total number of epochs')

        figname = 'IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/P1B1_results%s.pdf' %(self.condition, self.condition)
        print 'Saving figure'
        fig.savefig(figname)
        print 'Figure saved'

    def plotResults2(self):
        import seaborn as sns
        from matplotlib import pyplot as plt
        sns.set(style="white")

        fig, axarr = plt.subplots(2,3,sharex=True, figsize=(20, 10), dpi=300)
        if self.condition == 'S':
            fig.suptitle('Training from strach')
        elif self.condition == 'KT':
            fig.suptitle('Knowledge transfer from CNN of 50 indiv 25000 images')
        elif self.condition == 'KTC':
            fig.suptitle('Knowledge transfer from CNN of 50 indiv 25000 images (correlated images)')
        # fig.suptitle('Training from strach (group size vs number of images per individual)')
        self.plotArray2('valAccs',axarr[0,0],'Accuracy in validation',ylim=(0.,1.),ylabel='Accuracy')
        self.plotArray2('epochsToAcc',axarr[0,1],'Number of epochs to accuracy threshold %.2f' %self.accTh,ylim=(0,400),ylabel='Number of pochs')
        self.plotArray2('timeToAcc',axarr[0,2],'Time to accuracy threshold  %.2f (sec)' %self.accTh,ylim=(0,250),ylabel='Time (sec)')
        self.plotArray2('totalTime',axarr[1,1],'Total time in (sec)',ylim=(0,300),ylabel='Time (sec)')
        self.plotArray2('epochTime',axarr[1,2],'Single epoch time in (sec)',ylim=(0,2.5),ylabel='Time (sec)')
        self.plotArray2('totalEpochs',axarr[1,0],'Total number of epochs',ylim=(0,800),ylabel='Number of epochs')

        figname = 'IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/P1B1_results2%s.pdf' %(self.condition, self.condition)
        print 'Saving figure'
        fig.savefig(figname)
        print 'Figure saved'



if __name__ == '__main__':
    '''
    argv[1]: condition: 'S'-scratch, 'KT'-knowledgeT, 'KTC'-knowledgeTCorrelated, 'ALL'
    P1B1plotter.py KTC
    '''
    p = P1B1plotter(condition = condition)
    P1B1DictPath = 'IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/P1B1Dict.pkl' %sys.argv[6]
    print 'P1B1DictPath, ', P1B1DictPath
    condition = sys.argv[6]
    P1B1DictsPaths = scanFolder(P1B1DictPath)
    manyDicts = False
    if len(P1B1DictsPaths) > 1:
        manyDicts = True
        print 'Joining dictionaries from multiple jobs...'
        p.joinDicts(P1B1Dict=P1B1DictPath)
    else:
        print 'Loading dictionary to update p object...'
        P1B1Dict = pickle.load(open(P1B1DictPath,'rb'))
        p.__dict__.update(P1B1Dict)

    p = P1B1plotter(condition = condition)
    p.plotResults2()
