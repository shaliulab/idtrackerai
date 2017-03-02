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

    def __init__(self, job = 1, IMDBPath = 'IdTrackerDeep/data/36dpf_60indiv_29754ImPerInd_curvaturePortrait_0.hdf5', repList = '1', groupSizesCNN = '0', condition = 'S'):

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

    def compute(self):

        def runRepetition(self, loadCkpt_folder, images, labels, gCNN, g, n, r):

            def getCorrelatedImages(images,labels,numImages):
                '''
                This functions assumes that images and labels have not been permuted
                and they are temporarly ordered for each animals
                '''
                correlatedImages = []
                correlatedLabels = []
                firstFrameIndices= []
                if numImages*1.2 <= 25000:
                    numImages = int(numImages*1.2)
                for i in np.unique(labels):
                    thisIndivImages = images[labels==i]
                    thisIndivLabels = labels[labels==i]
                    framePos = np.random.randint(0,len(thisIndivImages) - numImages)
                    correlatedImages.append(thisIndivImages[framePos:framePos+numImages])
                    correlatedLabels.append(thisIndivLabels[framePos:framePos+numImages])
                    firstFrameIndices.append(framePos)

                imagesS = flatten(correlatedImages)
                imagesS = np.asarray(imagesS)
                labelsS = flatten(correlatedLabels)
                labelsS = np.asarray(labelsS)

                return imagesS, labelsS, firstFrameIndices

            # Get values of variables from counters
            groupSizeCNN = self.groupSizesCNN[gCNN]
            groupSize = self.groupSizes[g]
            numImForTrain = self.IMDBSizes[n]
            rep = self.repList[r]

            # Total number of images needed considering training and 10% more for validation
            numImToUse = numImForTrain + int(numImForTrain*.1)
            print 'Number of images for training, ', numImForTrain
            print 'Total number of images per animal, ', numImToUse

            # Get individuals indices for this repetition
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

            # Get current individuals images
            imagesS, labelsS = sliceDatabase(images, labels, indivIndices)

            # Get correlated images if needed
            if 'C' in self.condition:
                print 'Extracting correlated images...'
                imagesS, labelsS, firstFrameIndices = getCorrelatedImages(imagesS,labelsS,numImToUse)

            # Get images permutation of for this repetition
            if len(self.ImagesIndices[groupSizeCNN][groupSize][numImForTrain]) >= r + 1:
                print 'Restoring images permutation for rep ', rep
                permImages = self.ImagesIndices[groupSizeCNN][groupSize][numImForTrain][r]

            else:
                print 'Creating new permutation of images'
                permImages = permuter(len(labelsS),'imagesTrain',[])
                self.ImagesIndices[groupSizeCNN][groupSize][numImForTrain].append(permImages)

            # Permute images
            imagesS = imagesS[permImages]
            labelsS = labelsS[permImages]

            # Select images needed
            imagesS = imagesS[:numImToUse * groupSize]
            labelsS = labelsS[:numImToUse * groupSize]

            # # Data augmentation ### TODO code data augmentation
            # imagesSA, labelsSA = dataAugmenter(imagesS, labelsS)

            # Split in train and validation
            X_train, Y_train, X_val, Y_val = splitter(imagesS, labelsS, numImToUse, groupSize, self.imSize, numImForTrain * groupSize)
            print 'X_val shape', X_val.shape
            print 'Y_val shape', Y_val.shape
            print 'X_train shape', X_train.shape
            print 'Y_train shape', Y_train.shape

            # check train's dimensions
            cardTrain = int(numImForTrain)*groupSize
            dimTrainL = (cardTrain, groupSize)
            dimTrainI = (cardTrain, images.shape[2]*images.shape[3])
            dimensionChecker(X_train.shape, dimTrainI)
            dimensionChecker(Y_train.shape, dimTrainL)

            # check val's dimensions
            cardVal = int(np.ceil(numImForTrain*.1))*groupSize
            dimValL = (cardVal, groupSize)
            dimValI = (cardVal, images.shape[2] * images.shape[3])
            dimensionChecker(X_val.shape, dimValI)
            dimensionChecker(Y_val.shape, dimValL)

            # Update ckpt_dir
            if not self.kt:
                ckpt_dir = 'IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/numIndiv_%i/numImages_%i/rep_%i' %(self.condition,groupSize, numImForTrain, rep)
            elif self.kt:
                ckpt_dir = 'IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/CNN_%i/numIndiv_%i/numImages_%i/rep_%i' %(self.condition, groupSizeCNN, groupSize, numImForTrain, rep)

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
                                        self.keep_prob,self.lr,
                                        checkLearningFlag = self.checkLearningFlag,
                                        onlySoftmax=self.onlySoftmax)

            print 'Time in seconds, ', np.sum(lossAccDict['epochTime'])
            self.LossAccDicts[groupSizeCNN][groupSize][numImForTrain].append(lossAccDict)
            self.trainAccs[n,g,gCNN,r] = lossAccDict['acc'][-1]
            self.valAccs[n,g,gCNN,r] = lossAccDict['valAcc'][-1]

        # Load IMDB
        _, images, labels, self.imSize, self.numIndivImdb, self.numImagesPerIndiv = loadIMDB(self.IMDBName)

        # Training parameters
        self.channels, self.width, self.height = self.imSize
        self.resolution = np.prod(self.imSize)

        # Main loop
        for gCNN in range(self.numGroupsCNN): # Group size of the pre trained CNN model
            if not self.kt:
                loadCkpt_folder = ''
            elif self.kt:
                # By default we will use the first repetition and the model train with the whole library 25000 images.
                # FIXME be aware that the 25000 is hardcoded and can give errors if we train for another number of images for training...
                # loadCkpt_folder = 'IdTrackerDeep/figuresPaper/P1B1/CNN_modelsS/numIndiv_%i/numImages_%i/rep_%i' %(self.groupSizesCNN[gCNN], 25000, 1)
                loadCkpt_folder = 'IdTrackerDeep/videos/cafeina5peces/CNN_models/Session_2/AccumulationStep_131'


            for g in range(self.numGroups): # Group size for the current training

                for n in range(self.numIMDBSizes): # Number of images/individual for training

                    for r in range(self.numRepetitions): # Repetitions
                        print '\n******************************************************************************************************************************'
                        print '******************************************************************************************************************************'
                        if not self.kt:
                            print 'No knowledge transfer'
                        elif self.kt:
                            print 'Knowledge transfer from ', loadCkpt_folder
                            print 'GroupSizeCNN, ',self.groupSizesCNN[gCNN]
                        print 'Group size, ', self.groupSizes[g]
                        print 'numImForTrain, ', self.IMDBSizes[n]
                        print 'Repetition, ', self.repList[r]

                        runRepetition(self, loadCkpt_folder, images, labels, gCNN, g, n, r)

                        print '\nSaving dictionary...'
                        if not self.kt:
                            pickle.dump(self.__dict__,open('IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/P1B1Dict_job_%i.pkl' %(self.condition, self.job),'wb'))
                            print 'Dictionary saved in ', 'IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/P1B1Dict_job_%i.pkl' %(self.condition, self.job)
                        elif self.kt:
                            pickle.dump(self.__dict__,open('IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/P1B1Dict_job_%i.pkl' %(self.condition, self.job),'wb'))
                            print 'Dictionary saved in ', 'IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/P1B1Dict_job_%i.pkl' %(self.condition, self.job)
                        print '******************************************************************************************************************************'
                        print '******************************************************************************************************************************\n'

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

                            self.epochsToAcc[n,g,gCNN,r] = np.where(np.asarray(lossAccDict['valAcc'])>=accTh)[0][0]+1
                            self.timeToAcc[n,g,gCNN,r] = np.sum(lossAccDict['epochTime'][:int(self.epochsToAcc[n,g,gCNN,r])])

        print '\nSaving dictionary with times...'
        if not self.kt:
            pickle.dump(self.__dict__,open('IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/P1B1Dict_job_%i.pkl' %(self.condition, self.job),'wb'))
            print 'Dictionary saved in ', 'IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/P1B1Dict_job_%i.pkl' %(self.condition, self.job)
        elif self.kt:
            pickle.dump(self.__dict__,open('IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/P1B1Dict_job_%i.pkl' %(self.condition, self.job),'wb'))
            print 'Dictionary saved in ', 'IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/P1B1Dict_job_%i.pkl' %(self.condition, self.job)

if __name__ == '__main__':
    '''
    argv[1]: job number
    argv[2]: dataBase if 'd' is default
    argv[3]: repetitions
    argv[4]: groupSizeCNN
    argv[5]: condition: 'S'-scratch, 'KT'-knowledgeT, 'KTC'-knowledgeTCorrelated
    P1B1.py 1 d 1_2 2_5 S (job1,default library,repetitions[1 2],groupSizesCNN[2 5],from scratch)
    '''

    p = P1B1(job = int(sys.argv[1]), IMDBPath = sys.argv[2], repList = sys.argv[3], groupSizesCNN = sys.argv[4], condition = sys.argv[5])
    p.compute()
    p.computeTimes(accTh = 0.8)
