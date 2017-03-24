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

    def __init__(self, job = 1, IMDBPath = 'IdTrackerDeep/data/TU20170201_31pdf_72indiv_38494ImPerInd_curvaturePortrait_0.hdf5', repList = '1', groupSizesCNN = '0', condition = 'S'):

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
        # self.groupSizes = [2, 5, 10, 25, 50]
        self.groupSizes = [40]
        # self.groupSizes = [10, 25, 50]
        self.numGroups = len(self.groupSizes)
        self.repList = map(int,repList.split('_'))
        self.numRepetitions = len(self.repList)
        # self.IMDBSizes = [20,50,100,250,500,750,1000,3000,23000] # Images for training
        self.IMDBSizes = [30]
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
                	IMDBPath = 'IdTrackerDeep/data/TU20170131_31dpf_40indiv_31902ImPerInd_curvaturePortrait_0.hdf5'
    	    elif self.kt == True:
                	IMDBPath = 'IdTrackerDeep/data/TU20170131_31dpf_40indiv_31902ImPerInd_curvaturePortrait_0.hdf5'

            print '\nUsing default library, ', IMDBPath
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
                print 'indivIndices, ', indivIndices
            else:
                print 'Seeding the random generator...'
                np.random.seed(rep)
                permIndiv = permuter(self.numIndivImdb,'individualsTrain',[])
                indivIndices = permIndiv[:groupSize]
                self.IndivIndices[groupSizeCNN][groupSize].append(indivIndices)
                print 'indivIndices, ', indivIndices

            # Get current individuals images
            images, labels = sliceDatabase(images, labels, indivIndices)
            print 'Num train images per id, ', [np.sum(labels==i) for i in np.unique(labels)]
            images = np.expand_dims(images,axis=3)

            # Get permutations for images
            if len(self.ImagesIndices[groupSizeCNN][groupSize][numImForTrain]) >= r + 1:
                print 'Restoring images permutation for rep ', rep
                permImages = self.ImagesIndices[groupSizeCNN][groupSize][numImForTrain][r]

            else:
                print 'Creating new permutation of images'
                permImages = permuter(len(labels),'imagesTrain',[])
                self.ImagesIndices[groupSizeCNN][groupSize][numImForTrain].append(permImages)

            # Separate images from training-validation and testing
            if 'C' in self.condition:
                # Get images that are correlated in time
                print 'Extracting correlated images...'
                X_train, Y_train, X_val, Y_val, X_test, Y_test, firstFrameIndices = getCorrelatedImages(images,labels,numImForTrain, self.numImagesPerIndiv)

            else:
                # Permute images to uncorrelated
                imagesTrain = images[permImages[:numImToUse * groupSize]]
                labelsTrain = labels[permImages[:numImToUse * groupSize]]
                imagesTest= images[permImages[numImToUse * groupSize:]]
                labelsTest = labels[permImages[numImToUse * groupSize:]]

                # Split in train and validation
                X_train, Y_train, X_val, Y_val = splitter(imagesTrain, labelsTrain, numImToUse, groupSize, self.imSize, numImForTrain*groupSize)

            # Data Augmentation only to train and validation data
            X_train, Y_train = dataAugment(X_train,Y_train,flag = True)
            X_val, Y_val = dataAugment(X_val,Y_val,flag = False)
            X_test, Y_test = dataAugment(X_test,Y_test,flag = False)
            self.width, self.height, self.channels = X_train.shape[1:]

            # Pass labels from dense_to_one_hot
            Y_train = dense_to_one_hot(Y_train, n_classes=groupSize)
            Y_val = dense_to_one_hot(Y_val, n_classes=groupSize)
            Y_test = dense_to_one_hot(Y_test, n_classes=groupSize)
            print 'X_train shape', X_train.shape
            print 'Y_train shape', Y_train.shape
            print 'Num train images per id, ', np.sum(Y_train,axis=0)
            print 'X_val shape', X_val.shape
            print 'Y_val shape', Y_val.shape
            print 'Num val images per id, ', np.sum(Y_val,axis=0)
            print 'X_test shape', X_test.shape
            print 'Y_test shape', Y_test.shape
            print 'Num test images per id, ', np.sum(Y_test,axis=0)

            # Update ckpt_dir
            if not self.kt:
                ckpt_dir = 'IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/numIndiv_%i/numImages_%i/rep_%i' %(self.condition,groupSize, numImForTrain, rep)
            elif self.kt:
                ckpt_dir = 'IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/CNN_%i/numIndiv_%i/numImages_%i/rep_%i' %(self.condition, groupSizeCNN, groupSize, numImForTrain, rep)

            # Compute index batches
            numImagesT = Y_train.shape[0]
            numImagesV = Y_val.shape[0]
            numImagesTest = Y_test.shape[0]
            Tindices, Titer_per_epoch = get_batch_indices(numImagesT,self.batchSize)
            Vindices, Viter_per_epoch = get_batch_indices(numImagesV,self.batchSize)
            TestIndices, TestIter_per_epoch = get_batch_indices(numImagesTest,self.batchSize)

            # Run training
            lossAccDict = run_training(X_train, Y_train, X_val, Y_val, X_test, Y_test,
                                        self.width, self.height, self.channels, groupSize,
                                        ckpt_dir, loadCkpt_folder,
                                        self.batchSize, self.numEpochs,
                                        Tindices, Titer_per_epoch,
                                        Vindices, Viter_per_epoch,
                                        TestIndices, TestIter_per_epoch,
                                        self.keep_prob,self.lr,
                                        checkLearningFlag = self.checkLearningFlag,
                                        onlySoftmax=self.onlySoftmax)

            print 'Time in seconds, ', np.sum(lossAccDict['epochTime'])
            self.LossAccDicts[groupSizeCNN][groupSize][numImForTrain].append(lossAccDict)
            self.trainAccs[n,g,gCNN,r] = lossAccDict['acc'][-1]
            self.valAccs[n,g,gCNN,r] = lossAccDict['valAcc'][-1]

        # Prepare
        _, images, labels, self.imSize, self.numIndivImdb, self.numImagesPerIndiv = loadIMDB(self.IMDBPath)

        # Standarization of images
        images = images/255.
        meanIm = np.mean(images, axis=0)
        stdIm = np.std(images,axis=0)
        images = (images-meanIm)/stdIm

        # Training parameters
        # self.channels = 1
        # self.width = 32
        # self.height = 32
        # self.resolution = np.prod(self.imSize)

        # Main loop
        for gCNN in range(self.numGroupsCNN): # Group size of the pre trained CNN model
            if not self.kt:
                loadCkpt_folder = ''
            elif self.kt:
                # By default we will use the first repetition and the model train with the whole library 25000 images.
                # FIXME be aware that the 25000 is hardcoded and can give errors if we train for another number of images for training...
                loadCkpt_folder = 'IdTrackerDeep/figuresPaper/P1B1/CNN_modelsS/numIndiv_%i/numImages_%i/rep_%i' %(self.groupSizesCNN[gCNN], 25000, 1)


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
