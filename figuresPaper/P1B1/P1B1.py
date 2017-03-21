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

class P1B1(object):

    def __init__(self, job = 1, IMDBPath = 'IdTrackerDeep/data/36dpf_60indiv_29754ImPerInd_curvaturePortrait_0.hdf5', repList = '1', groupSizesCNN = '0'):

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

        if IMDBPath == 'd':
            IMDBPath = 'IdTrackerDeep/data/36dpf_60indiv_29754ImPerInd_curvaturePortrait_0.hdf5'
            # IMDBPath = 'IdTrackerDeep/data/25dpf_60indiv_26142imperind_curvatureportrait2_0.hdf5'

            print 'Using default library, ', IMDBPath

        # Dataset from which to load the images for training
        self.IMDBPath = IMDBPath
        self.IMDBName = getIMDBNameFromPath(self.IMDBPath)

        # Job counter for condor
        self.job = job

        # Figure parameters
        self.groupSizesCNN = map(int,groupSizesCNN.split('_'))
        self.numGroupsCNN = len(self.groupSizesCNN)
        self.groupSizes = [2, 5, 10, 25, 50]
        self.numGroups = len(self.groupSizes)
        self.repList = map(int,repList.split('_'))
        self.numRepetitions = len(self.repList)
        self.IMDBSizes = [20,50,100,250,500,750]
        self.numIMDBSizes = len(self.IMDBSizes)

        # Initialize figure arrays
        self.trainAccs = np.ones((self.numIMDBSizes, self.numGroups, self.numGroupsCNN, self.numRepetitions)) * np.nan
        self.valAccs = np.ones((self.numIMDBSizes, self.numGroups, self.numGroupsCNN, self.numRepetitions)) * np.nan

        # CNN parameters
        if self.groupSizesCNN == [0]:
            self.kt = False
        else:
            self.kt = True
        self.batchSize = 250
        self.numEpochs = 5000
        self.lr = 0.01
        self.keep_prob = 1.0

        # Initialize variables
        initializeDicts(self)
        # self.IndivIndices, self.ImageIndices, self.LossAccDicts = initializeDicts(self)

    def compute(self):

        def runRepetition(self, loadCkpt_folder, images, labels, gCNN, g, n, r):
            groupSizeCNN = self.groupSizesCNN[gCNN]
            groupSize = self.groupSizes[g]
            numImagesToUse = self.IMDBSizes[n]
            rep = self.repList[r]

            # Get individuals for this repetition
            print 'Individual indices per group size', self.IndivIndices
            print 'Number of repetitions already computed', len(self.IndivIndices[groupSizeCNN][groupSize])
            if len(self.IndivIndices[groupSizeCNN][groupSize]) >= r + 1:
                print 'Restoring individual indices for rep ', rep
                indivIndices = self.IndivIndices[groupSizeCNN][groupSize][r]

            else:
                print 'Seeding the random generator...'
                np.random.seed(rep)

                permIndiv = permuter(self.numIndivImdb,'individualsTrain',[])
                indivIndices = permIndiv[:groupSize]
                self.IndivIndices[groupSizeCNN][groupSize].append(indivIndices)


            # Get images of the current individuals
            imagesS, labelsS = sliceDatabase(images, labels, indivIndices)

            # Get permutation of for this repetition
            if len(self.ImagesIndices[groupSizeCNN][groupSize][numImagesToUse]) >= r + 1:
                print 'Restoring images permutation for rep ', rep
                permImages = self.ImagesIndices[groupSizeCNN][groupSize][numImagesToUse][r]

            else:
                permImages = permuter(len(labelsS),'imagesTrain',[])
                self.ImagesIndices[groupSizeCNN][groupSize][numImagesToUse].append(permImages)

            # Permute images
            imagesS = imagesS[permImages]
            labelsS = labelsS[permImages]

            # Select images needed
            imagesS = imagesS[:numImagesToUse * groupSize]
            labelsS = labelsS[:numImagesToUse * groupSize]

            # # Data augmentation ### TODO code data augmentation
            # imagesSA, labelsSA = dataAugmenter(imagesS, labelsS)


            # Split in train and validation
            X_train, Y_train, X_val, Y_val = splitter(imagesS, labelsS, numImagesToUse, groupSize, self.imSize)
            print 'len Y_train + Y_val, ', len(Y_train) + len(Y_val)

            # check train's dimensions
            cardTrain = int(np.ceil(np.true_divide(np.multiply(numImagesToUse,9),10)))*groupSize
            dimTrainL = (cardTrain, groupSize)
            dimTrainI = (cardTrain, images.shape[2]*images.shape[3])
            dimensionChecker(X_train.shape, dimTrainI)
            dimensionChecker(Y_train.shape, dimTrainL)

            # check val's dimensions
            cardVal = int(np.ceil(np.true_divide(numImagesToUse,10)))*groupSize
            dimValL = (cardVal, groupSize)
            dimValI = (cardVal, images.shape[2] * images.shape[3])
            dimensionChecker(X_val.shape, dimValI)
            dimensionChecker(Y_val.shape, dimValL)

            # Update ckpt_dir
            if not self.kt:
                ckpt_dir = 'IdTrackerDeep/figuresPaper/P1B1/CNN_models/numIndiv_%i/numImages_%i/rep_%i' %(groupSize, numImagesToUse, rep)
            elif self.kt:
                ckpt_dir = 'IdTrackerDeep/figuresPaper/P1B1/CNN_modelsKT/CNN_%i/numIndiv_%i/numImages_%i/rep_%i' %(groupSizeCNN, groupSize, numImagesToUse, rep)

            # Compute index batches
            numImagesT = Y_train.shape[0]
            numImagesV = Y_val.shape[0]
            Tindices, Titer_per_epoch = get_batch_indices(numImagesT,self.batchSize)
            Vindices, Viter_per_epoch = get_batch_indices(numImagesV,self.batchSize)

            # Run training
            lossAccDict = run_training(X_train, Y_train, X_val, Y_val,
                                        self.width, self.height, self.channels, groupSize, self.resolution,
                                        ckpt_dir, loadCkpt_folder,
                                        self.batchSize, self.numEpochs,
                                        Tindices, Titer_per_epoch,
                                        Vindices, Viter_per_epoch,
                                        self.keep_prob,self.lr)

            print 'Time in seconds, ', np.sum(lossAccDict['epochTime'])
            self.LossAccDicts[groupSizeCNN][groupSize][numImagesToUse].append(lossAccDict)
            self.trainAccs[n,g,gCNN,r] = lossAccDict['acc'][-1]
            self.valAccs[n,g,gCNN,r] = lossAccDict['valAcc'][-1]

        # Load IMDB
        _, images, labels, self.imSize, self.numIndivImdb, self.numImagesPerIndiv = loadIMDB(self.IMDBPath)

        # Training parameters
        self.channels, self.width, self.height = self.imSize
        self.resolution = np.prod(self.imSize)

        # Main loop
        for gCNN in range(self.numGroupsCNN):
            if not self.kt:
                print 'No knowledge transfer'
                loadCkpt_folder = ''
            elif self.kt:
                # By default we will use the first repetition and the model train with the whole library 25000 images.
                loadCkpt_folder = 'IdTrackerDeep/figuresPaper/P1B1/CNN_models/numIndiv_%i/numImages_%i/rep_%i' %(self.groupSizesCNN[gCNN], 25000, 1)
                print 'Knowledge transfer from ', loadCkpt_folder

            for g in range(self.numGroups):

                for n in range(self.numIMDBSizes):

                    for r in range(self.numRepetitions):
                        print '\n******************************************************************************************************************************'
                        print '******************************************************************************************************************************'
                        if not self.kt:
                            print 'No knowledge transfer'
                        elif self.kt:
                            print 'Knowledge transfer from ', loadCkpt_folder
                            print 'GroupSizeCNN, ',self.groupSizesCNN[gCNN]
                        print 'Group size, ', self.groupSizes[g]
                        print 'numImagesToUse, ', self.IMDBSizes[n]
                        print 'Repetition, ', self.repList[r]

                        runRepetition(self, loadCkpt_folder, images, labels, gCNN, g, n, r)

                        print '\nSaving dictionary...'
                        if not self.kt:
                            pickle.dump(self.__dict__,open('IdTrackerDeep/figuresPaper/P1B1/CNN_models/P1B1Dict_job_%i.pkl' %self.job,'wb'))
                        elif self.kt:
                            pickle.dump(self.__dict__,open('IdTrackerDeep/figuresPaper/P1B1/CNN_modelsKT/P1B1Dict_job_%i.pkl' %self.job,'wb'))
                        print '******************************************************************************************************************************'
                        print '******************************************************************************************************************************\n'

    def joinDicts(self,P1B1Dict='IdTrackerDeep/figuresPaper/P1B1/CNN_models/P1B1Dict_job_1.pkl'):
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
        pickle.dump(self.__dict__,open('IdTrackerDeep/figuresPaper/P1B1/CNN_models/P1B1Dict.pkl','wb'))

    def computeTimes(self,accTh = 0.8):
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

                            self.epochsToAcc[n,g,gCNN,r] = np.where(np.asarray(lossAccDict['valAcc'])>=accTh)[0][0]
                            self.timeToAcc[n,g,gCNN,r] = np.sum(lossAccDict['epochTime'][:int(self.epochsToAcc[n,g,gCNN,r])])

    def plotArray(self,arrayName,ax,title):
        import seaborn as sns
        sns.set(style="white")
        arrayMedian = np.nanmedian(getattr(self, arrayName),axis = 3)

        sns.heatmap(arrayMedian[:,:,0], annot=True, ax=ax, cbar=False, fmt='.2f')
        ax.set_xlabel('Group size')
        ax.set_ylabel('Num im/indiv')
        ax.set_xticklabels(p.groupSizes)
        ax.set_yticklabels(p.IMDBSizes[::-1])
        ax.invert_yaxis()
        ax.set_title(title)

    def plotResults(self):
        import seaborn as sns
        sns.set(style="white")

        fig, axarr = plt.subplots(2,3,sharex=True,sharey=True, figsize=(20, 10), dpi=300)
        fig.suptitle('Knowlede transfer from CNN of 50 indiv 25000 images (group size vs number of images per individual)')
        # fig.suptitle('Training from strach (group size vs number of images per individual)')
        self.plotArray('valAccs',axarr[0,0],'Accuracy in validation')
        self.plotArray('epochsToAcc',axarr[0,1],'Number of epochs to accuracy threshold %.2f' %self.accTh)
        self.plotArray('timeToAcc',axarr[0,2],'Time to accuracy threshold  %.2f (sec)' %self.accTh)
        self.plotArray('totalTime',axarr[1,1],'Total time in (sec)')
        self.plotArray('epochTime',axarr[1,2],'Single epoch time in (sec)')
        self.plotArray('totalEpochs',axarr[1,0],'Total number of epochs')

        figname = 'IdTrackerDeep/figuresPaper/P1B1/CNN_modelsKT/P1B1_resultsKT.pdf'

        print 'Saving figure'
        fig.savefig(figname)
        print 'Figure saved'

if __name__ == '__main__':
    '''
    argv[1]: useCondor (bool flag)
    argv[2]: job number
    argv[3]: dataBase if 'd' is default
    argv[4]: repetitions
    argv[5]: groupSizeCNN
    P1B1.py 1 1 d 1_2 2_5 (useCondor,job1,default library,repetitions[1 2],groupSizesCNN[2 5])
    '''

    plotFlag = int(sys.argv[1])
    if not plotFlag:
        p = P1B1()
        P1B1DictPath = selectFile()
        P1B1DictsPaths = scanFolder(P1B1DictPath)
        manyDicts = False
        if len(P1B1DictsPaths) > 1:
            manyDicts = True
        print 'Loading dictionary to update p object...'
        P1B1Dict = pickle.load(open(P1B1DictPath,'rb'))
        p.__dict__.update(P1B1Dict)
        print 'The group sizes are, ', p.groupSizes
        print 'The IMDB sizes are, ', p.IMDBSizes
        if manyDicts:
            p.joinDicts(P1B1Dict=P1B1DictPath)
        p.computeTimes(accTh = 0.8)
        p.plotResults()

    elif plotFlag:
        p = P1B1(job = int(sys.argv[2]), IMDBPath = sys.argv[3], repList = sys.argv[4], groupSizesCNN = sys.argv[5])
        p.compute()
        # p.computeTimes(accTh = 0.7)
        # p.plotResults()
