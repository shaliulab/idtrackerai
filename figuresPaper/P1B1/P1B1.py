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
import gc

class P1B1(object):

    def __init__(self, cluster = 0, job = 1, IMDBCode = 'A', idsCode = 'a', repList = '1', groupSizesCNN = '0', condition = 'S'):

        def getIMDBNameFromPath(IMDBPath):
            filename, extension = os.path.splitext(IMDBPath)
            IMDBName = '_'.join(filename.split('/')[-1].split('_')[:-1])

            return IMDBName

        # Job counter for condor
        self.cluster = cluster
        self.job = job
        self.condition = condition
        # Figure parameters
        self.groupSizesCNN = map(int,groupSizesCNN.split('_'))
        self.numGroupsCNN = len(self.groupSizesCNN)
        #self.groupSizes = [2, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
        self.groupSizes = [2, 5, 10, 25, 50, 75, 90]
        # self.groupSizes = [10, 25, 50]
        self.numGroups = len(self.groupSizes)
        self.repList = map(int,repList.split('_'))
        self.numRepetitions = len(self.repList)
        self.IMDBSizes = [20,50,100,250,500,1000,3000,28000] # Images for training
        # self.IMDBSizes = [3000]
        # self.IMDBSizes = [20,50,100,250]
        self.numIMDBSizes = len(self.IMDBSizes)

        # Initialize figure arrays
        self.trainAccs = np.ones((self.numIMDBSizes, self.numGroups, self.numGroupsCNN, self.numRepetitions)) * np.nan
        self.valAccs = np.ones((self.numIMDBSizes, self.numGroups, self.numGroupsCNN, self.numRepetitions)) * np.nan
        self.testAccs = np.ones((self.numIMDBSizes, self.numGroups, self.numGroupsCNN, self.numRepetitions)) * np.nan

        # Set CNN training parameters
        self.batchSize = 250
        self.numEpochs = 2000
        self.lr = 0.01

    	# Set keep probability for dropout
    	self.keep_prob = 1.0
    	if 'D' in self.condition:
       	    self.keep_prob = 0.7

        # Set flag to indicate knowledge transfer
        self.kt = False
        if 'KT' in self.condition:
            self.kt = True

        # Set flag to stop the training when it is not learning much
        self.checkLearningFlag = False
        if 'V' in self.condition: # 'V' stands for video condition for early stopping
            self.checkLearningFlag = True

        # Set flag to only train softmax
        self.onlySoftmax = False
        if 'X' in self.condition:
            self.trainSoftmax = True

        # Set flag to train fully connected and softmax
        self.trainFullyConnected = False
        if 'FX' in self.condition:
            self.trainFullyConnected = True
            self.trainSoftmax = True

        # Set flag for data Augmentation
        self.dataAugmentation = False
        if 'A' in self.condition:
            self.dataAugmentation = True
            self.IMDBSizes = [20,50,100,250,500]
	    self.numIMDBSizes = len(self.IMDBSizes)

        # Set flag for correlated iamges
        self.correlatedImages = False
        if 'C' in self.condition:
            self.correlatedImages = True

        # Get list of IMDBPaths form IMDBCode
        print '\nReading IMDBCode and idsCode...'
        if not self.cluster:
            datafolder = '/home/chaos/Desktop/IdTrackerDeep/' 
        elif self.cluster:
            datafolder = '/admin/'
        IMDBsDict = {
                    'A': datafolder + 'data/TU20160413_36dpf_60indiv_29938ImPerInd_curvaturePortrait_0.hdf5',
                    'B': datafolder + 'data/TU20160428_36dpf_60indiv_28010ImPerInd_curvaturePortrait_0.hdf5',
                    'C': datafolder + 'data/TU20160920_36dpf_64indiv_7731ImPerInd_curvaturePortrait_0.hdf5',
                    'D': datafolder + 'data/TU20170131_31dpf_40indiv_34770ImPerInd_curvaturePortrait_0.hdf5',
                    'E': datafolder + 'data/TU20170201_31pdf_72indiv_38739ImPerInd_curvaturePortrait_0.hdf5',
                    'F': datafolder + 'data/TU20170202_31pdf_72indiv_38913ImPerInd_curvaturePortrait_0.hdf5',
                    'a': datafolder + 'data/TU20160413_36dpf_16indiv_29938ImPerInd_curvaturePortrait_0.hdf5',
                    'b': datafolder + 'data/TU20160428_36dpf_16indiv_28818ImPerInd_curvaturePortrait_0.hdf5',
                    'c': datafolder + 'data/TU20160920_36dpf_16indiv_7731ImPerInd_curvaturePortrait_0.hdf5',
                    'd': datafolder + 'data/TU20170131_31dpf_16indiv_38989ImPerInd_curvaturePortrait_0.hdf5',
                    'e': datafolder + 'data/TU20170201_31pdf_16indiv_38997ImPerInd_curvaturePortrait_0.hdf5',
                    'f': datafolder + 'data/TU20170202_31pdf_16indiv_38998ImPerInd_curvaturePortrait_0.hdf5'
                    }
        self.IMDBPaths = []
        self.idsInIMDBs = []
        for (letter1,letter2) in zip(IMDBCode,idsCode):
            print '\nletter1, ', letter1
            self.IMDBPaths.append(IMDBsDict[letter1])
            IMDBName = getIMDBNameFromPath(IMDBsDict[letter1])
            print 'IMDBName, ', IMDBName
            strain, age, numIndivIMDB, numImPerIndiv = getIMDBInfoFromName(IMDBName)
            print 'numIndivIMDB', numIndivIMDB
            print 'letter2, ', letter2
            if letter2 == 'a': # all ids
                ids = range(numIndivIMDB)
            elif letter2 == 'f': # first half idsInIMDBs
                ids = range(numIndivIMDB/2)
            elif letter2 == 's': # first half idsInIMDBs
                ids = range(numIndivIMDB/2,numIndivIMDB)
            print 'ids selected, ', ids
            self.idsInIMDBs.append(ids)
        print 'IMDBPaths, ', self.IMDBPaths
        print 'idsInIMDBs, ', self.idsInIMDBs

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
                permIndiv = np.random.permutation(self.numIndivImdb)
                indivIndices = permIndiv[:groupSize]
                self.IndivIndices[groupSizeCNN][groupSize].append(indivIndices)
                print 'indivIndices, ', indivIndices

            # Get current individuals images
            images, labels = sliceDatabase(images, labels, indivIndices)
            print 'Num train images per id, ', [np.sum(labels==i) for i in np.unique(labels)]
            images = np.expand_dims(images,axis=3)

            # Separate images from training, validation and testing
            if self.correlatedImages:

                # Get images that are correlated in time
                X_train, Y_train, X_val, Y_val, X_test, Y_test, firstFrameIndex = getCorrelatedImages(images, labels, numImForTrain, self.minNumImagesPerIndiv)
                self.ImagesIndices[groupSizeCNN][groupSize][numImForTrain].append(firstFrameIndex)
            else:

                # Get images that are uncorrelated in time
                print 'Extracting uncorrelated images...'
                # Get permutations for images
                if len(self.ImagesIndices[groupSizeCNN][groupSize][numImForTrain]) >= r + 1:
                    print 'Restoring images permutation for rep ', rep
                    permImages = self.ImagesIndices[groupSizeCNN][groupSize][numImForTrain][r]

                else:
                    print 'Creating new permutation of images'
                    permImages = np.random.permutation(len(labels))
                    self.ImagesIndices[groupSizeCNN][groupSize][numImForTrain].append(permImages)

                images = images[permImages]
                labels = labels[permImages]
                X_train, Y_train, X_val, Y_val, X_test, Y_test = getUncorrelatedImages(images, labels, numImForTrain, self.minNumImagesPerIndiv)

            # Data Augmentation only to train and validation data
            print 'X_train shape', X_train.shape
            print 'X_val shape', X_val.shape
            print 'X_test shape', X_test.shape
            if self.dataAugmentation:
                X_train, Y_train = dataAugment(X_train,Y_train,dataAugment = True)
            else:
                X_train, Y_train = dataAugment(X_train,Y_train,dataAugment = False)

            X_val, Y_val = dataAugment(X_val,Y_val,dataAugment = False)
            X_test, Y_test = dataAugment(X_test,Y_test,dataAugment = False)
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
                                        onlySoftmax=self.onlySoftmax,
                                        onlyFullyConnected = self.trainFullyConnected)

            print 'Time in seconds, ', np.sum(lossAccDict['epochTime'])
            self.LossAccDicts[groupSizeCNN][groupSize][numImForTrain].append(lossAccDict)
            self.trainAccs[n,g,gCNN,r] = lossAccDict['acc'][-1]
            self.valAccs[n,g,gCNN,r] = lossAccDict['valAcc'][-1]
            self.testAccs[n,g,gCNN,r] = lossAccDict['testAcc'][-1]

        # Load data from IMDBs
        def loadIMDBs(self):
            # Initialize variables
            images = []
            labels = []
            self.numIndivImdb = 0
            self.strains = []
            self.ages = []
            for (IMDBPath,idsInIMDB) in zip(self.IMDBPaths,self.idsInIMDBs):
                IMDBName = getIMDBNameFromPath(IMDBPath)
                strain, age, numIndivIMDB, numImPerIndiv = getIMDBInfoFromName(IMDBName)
                print '\nExtracting imagaes from ', IMDBName
                print 'The individuals selected from this IMDB are ',  idsInIMDB
                print 'strain, ', strain
                print 'age, ', age
                print 'numIndivIMDB, ', numIndivIMDB
                self.strains.append(strain)
                self.ages.append(age)

                # Check whether there are enough individuals in the IMDB
                if numIndivIMDB < len(idsInIMDB):
                    raise ValueError('The number of indiv requested is bigger than the number of indiv in the IMDB')

                # Load IMDB
                _, imagesIMDB, labelsIMDB, self.imsize, _, _ = loadIMDB(IMDBPath)

                # If the number of individuals requested is smaller I need to slice the IMDB
                if numIndivIMDB > len(idsInIMDB):
                    imagesIMDB, labelsIMDB = sliceDatabase(imagesIMDB, labelsIMDB, idsInIMDB)

                ### FIXME there is some problem in the construction of the IMDBs because some of them have he channels dimension and other do not
                if len(labelsIMDB.shape) == 1:
                    imagesIMDB = np.expand_dims(imagesIMDB,axis=1)
                    labelsIMDB = np.expand_dims(labelsIMDB,axis=1)

                # Update labels values according to the number of individuals already loaded
                labelsIMDB = labelsIMDB+self.numIndivImdb

                # Append labels and images to the list
                print 'images shape ', imagesIMDB.shape
                print 'labels shape ', labelsIMDB.shape
                images.append(imagesIMDB)
                labels.append(labelsIMDB)
                print 'The labels added are, ', np.unique(labelsIMDB)

                # Update number of individuals loaded
                self.numIndivImdb += len(idsInIMDB)

                # To clear memory
                imagesIMDB = None
                labelsIMDB = None

            # Stack all images and labes
            images = np.vstack(images)
            labels = np.vstack(labels)
            print 'images shape ', images.shape
            print 'labels shape ', labels.shape
            print 'labels ', np.unique(labels)
            self.minNumImagesPerIndiv = np.min([np.sum(labels == i) for i in np.unique(labels)])
            print 'num images per label, ', self.minNumImagesPerIndiv

            return images, labels

        images, labels = loadIMDBs(self)
        gc.collect()
        
        # Standarization of images
        images = images/255.
        meanIm = np.mean(images, axis=0)
        stdIm = np.std(images,axis=0)
        images = (images-meanIm)/stdIm

        # Main loop
        for gCNN in range(self.numGroupsCNN): # Group size of the pre trained CNN model
            if not self.kt:
                loadCkpt_folder = ''
            elif self.kt:
                # By default we will use the first repetition and the model train with the whole library 25000 images.
                # FIXME be aware that the 25000 is hardcoded and can give errors if we train for another number of images for training...
                loadCkpt_folder = 'IdTrackerDeep/figuresPaper/P1B1/CNN_modelsS/numIndiv_%i/numImages_%i/rep_%i' %(self.groupSizesCNN[gCNN], 28000, 1)


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
    argv[1]: 1 = cluster, 0 = no cluster
    argv[2]: job number
    argv[3]: IMDBs (A, B, C, D, E, F)
    argv[4]: idsInIMDBs: a - all, f - first half, s - second half
    argv[5]: repetitions
    argv[6]: groupSizeCNN
    argv[7]: condition: 'S'-scratch, 'KT'-knowledgeT, 'KTC'-knowledgeTCorrelated
    P1B1.py 1 1 AB af 1_2 20_50 S (running in the cluster, job1, library A and B, all individuals in library A and first half obf B, repetitions[1 2],groupSizesCNN[2 5],from scratch)
    '''

    p = P1B1(cluster = sys.argv[1], job = int(sys.argv[2]), IMDBCode = sys.argv[3], idsCode = sys.argv[4], repList = sys.argv[5], groupSizesCNN = sys.argv[6], condition = sys.argv[7])
    p.compute()
    p.computeTimes(accTh = 0.8)
