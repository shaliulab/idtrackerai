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

        print '\n----------------------------------------------------------------'
        # Job counter for condor
        self.cluster = cluster
        self.job = job
        self.condition = condition
        self.IMDBCode = IMDBCode
        self.idsCode = idsCode
        # Figure parameters
        self.groupSizesCNN = map(int,groupSizesCNN.split('_'))
        self.numGroupsCNN = len(self.groupSizesCNN)
        self.groupSizes = [2,5,10,25]
        self.numGroups = len(self.groupSizes)
        self.repList = map(int,repList.split('_'))
        self.numRepetitions = len(self.repList)
        # self.IMDBSizes = [20,50,100,250,500,1000,3000,28000] # Images for training
        self.IMDBSizes = [20,50,100] # Images for training
        self.numIMDBSizes = len(self.IMDBSizes)

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
	    self.IMDBSizes = [20,50,100,250,500]
	    self.numIMDBSizes = len(self.IMDBSizes)

        # Set flag for accuracy by fragments
        if 'F' in self.condition:
            self.acc_by_fragments = True

        # Get list of IMDBPaths form IMDBCode
        print '\nReading IMDBCode and idsCode...'
        if not int(self.cluster):
            self.datafolder = 'IdTrackerDeep/'
        elif int(self.cluster):
            self.datafolder = '/admin/'
        self.IMDBsDict = {
                    'A': self.datafolder + 'data/TU20160413_36dpf_60indiv_29938ImPerInd_curvaturePortrait_0.hdf5',
                    'B': self.datafolder + 'data/TU20160428_36dpf_60indiv_28010ImPerInd_curvaturePortrait_0.hdf5',
                    'C': self.datafolder + 'data/TU20160920_36dpf_64indiv_7731ImPerInd_curvaturePortrait_0.hdf5',
                    'D': self.datafolder + 'data/TU20170131_31dpf_40indiv_34770ImPerInd_curvaturePortrait_0.hdf5',
                    'E': self.datafolder + 'data/TU20170201_31pdf_72indiv_38739ImPerInd_curvaturePortrait_0.hdf5',
                    'F': self.datafolder + 'data/TU20170202_31pdf_72indiv_38913ImPerInd_curvaturePortrait_0.hdf5',
                    'a': self.datafolder + 'data/TU20160413_36dpf_16indiv_29938ImPerInd_curvaturePortrait_0.hdf5',
                    'b': self.datafolder + 'data/TU20160428_36dpf_16indiv_28818ImPerInd_curvaturePortrait_0.hdf5',
                    'd': self.datafolder + 'data/TU20170131_31dpf_16indiv_38989ImPerInd_curvaturePortrait_0.hdf5',
                    'c': self.datafolder + 'data/TU20160920_36dpf_16indiv_7731ImPerInd_curvaturePortrait_0.hdf5',
                    'e': self.datafolder + 'data/TU20170201_31pdf_16indiv_38997ImPerInd_curvaturePortrait_0.hdf5',
                    'f': self.datafolder + 'data/TU20170202_31pdf_16indiv_38998ImPerInd_curvaturePortrait_0.hdf5'
                    }
        self.IMDBPaths = []
        self.idsInIMDBs = []
        for (letter1,letter2) in zip(self.IMDBCode,self.idsCode):
            print '\nletter1, ', letter1
            self.IMDBPaths.append(self.IMDBsDict[letter1])
            IMDBName = getIMDBNameFromPath(self.IMDBsDict[letter1])
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

        print '\nSaving dictionary...'
        self.CNN_modelsPath = 'IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/' %self.condition
        if not os.path.exists(self.CNN_modelsPath):
            os.makedirs(self.CNN_modelsPath)
        pickle.dump(self.__dict__,open(self.CNN_modelsPath + 'info.pkl' ,'wb'))
        print 'Dictionary saved in ', self.CNN_modelsPath + 'info.pkl'
        print '----------------------------------------------------------------\n'

    def loadIMDBs(self):

        print '\n----------------------------------------------------------------'
        print 'Loading images and labels form the IMDB selected'
        # Initialize variables
        self.images = []
        self.labels = []
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
            self.images.append(imagesIMDB)
            self.labels.append(labelsIMDB)
            print 'The labels added are, ', np.unique(labelsIMDB)

            # Update number of individuals loaded
            self.numIndivImdb += len(idsInIMDB)

            # To clear memory
            imagesIMDB = None
            labelsIMDB = None

        # Stack all images and labes
        self.images = np.vstack(self.images)
        self.labels = np.vstack(self.labels)
        print 'images shape ', self.images.shape
        print 'labels shape ', self.labels.shape
        print 'labels ', np.unique(self.labels)
        self.minNumImagesPerIndiv = np.min([np.sum(self.labels == i) for i in np.unique(self.labels)])
        print 'num images per label, ', self.minNumImagesPerIndiv

        # Standarization of images
        self.images = standarizeImages(self.images)
        print '----------------------------------------------------------------\n'

    def runRepetition(self):

        # Total number of images needed considering training and 10% more for validation
        numImToUse = self.numImForTrain + int(self.numImForTrain*.1)
        print 'Number of images for training, ', self.numImForTrain
        print 'Total number of images per animal, ', numImToUse

        # Get individuals indices for this repetition
        print 'Seeding the random generator...'
        np.random.seed(self.rep)
        permIndiv = np.random.permutation(self.numIndivImdb)
        indivIndices = permIndiv[:self.groupSize]
        print 'indivIndices, ', indivIndices

        # Get current individuals images
        images, labels = sliceDatabase(self.images, self.labels, indivIndices)
        print 'Num train images per id, ', [np.sum(labels==i) for i in np.unique(labels)]
        images = np.expand_dims(images,axis=3)

        # Separate images from training, validation and testing
        if self.correlatedImages:

            # Get images that are correlated in time
            X_train, Y_train, X_val, Y_val, X_test, Y_test, firstFrameIndex = getCorrelatedImages(images, labels, self.numImForTrain, self.minNumImagesPerIndiv,self.rep)
            imagesPermutation = None
        else:

            # Get images that are uncorrelated in time
            print 'Extracting uncorrelated images...'
            print 'Creating new permutation of images'
            np.random.seed(self.rep)

            print 'len labels, ', len(labels)
            imagesPermutation = np.random.permutation(len(labels))
            firstFrameIndex = None

            print 'len permutation, ', len(imagesPermutation)
            images = images[imagesPermutation]
            labels = labels[imagesPermutation]
            print 'Images shape', images.shape
            X_train, Y_train, X_val, Y_val, X_test, Y_test = getUncorrelatedImages(images, labels, self.numImForTrain, self.minNumImagesPerIndiv)

        images = None
        labels = None
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

        Y_train = dense_to_one_hot(Y_train, n_classes=self.groupSize)
        Y_val = dense_to_one_hot(Y_val, n_classes=self.groupSize)
        Y_test = dense_to_one_hot(Y_test, n_classes=self.groupSize)
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
        ckpt_dir = 'IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/CNN_%i/numIndiv_%i/numImages_%i/rep_%i' %(self.condition, self.groupSizeCNN, self.groupSize, self.numImForTrain, self.rep)

        # Compute index batches
        numImagesT = Y_train.shape[0]
        numImagesV = Y_val.shape[0]
        numImagesTest = Y_test.shape[0]
        Tindices, Titer_per_epoch = get_batch_indices(numImagesT,self.batchSize)
        Vindices, Viter_per_epoch = get_batch_indices(numImagesV,self.batchSize)
        TestIndices, TestIter_per_epoch = get_batch_indices(numImagesTest,self.batchSize)

        # Run training
        lossAccDict, ckpt_dir_model = run_training(X_train, Y_train, X_val, Y_val, X_test, Y_test,
                                    self.width, self.height, self.channels, self.groupSize,
                                    ckpt_dir, self.loadCkpt_folder,
                                    self.batchSize, self.numEpochs,
                                    Tindices, Titer_per_epoch,
                                    Vindices, Viter_per_epoch,
                                    TestIndices, TestIter_per_epoch,
                                    self.keep_prob,self.lr,
                                    checkLearningFlag = self.checkLearningFlag,
                                    onlySoftmax=self.onlySoftmax,
                                    onlyFullyConnected = self.trainFullyConnected,
                                    saveFlag = False)

        lossAccDict['repDict'] = {
                                'groupSizeCNN': self.groupSizeCNN,
                                'groupSize': self.groupSize,
                                'IMBDSize': self.numImForTrain,
                                'repetition': self.rep,
                                'indivIndices': indivIndices,
                                'imagesPermutation': imagesPermutation,
                                'firstFrameIndex': firstFrameIndex,
                                'numImagesT': numImagesT,
                                'numImagesV': numImagesV,
                                'numImagesTest': numImagesTest,
                                }

        info = pickle.load(open(self.CNN_modelsPath + 'info.pkl','rb'))
        lossAccDict['info'] = info
        print 'Saving lossAccDict...'
        pickle.dump( lossAccDict, open( ckpt_dir_model + "/lossAcc.pkl", "wb" ) )
        print 'lossAccDict saved'

    def compute(self):

        # Main loop
        for gCNN, self.groupSizeCNN in enumerate(self.groupSizesCNN): # Group size of the pre trained CNN model

            if not self.kt:

                self.loadCkpt_folder = ''

            elif self.kt:

                # By default we will use the first repetition and the model train with the whole library 25000 images.
                # FIXME be aware that the 25000 is hardcoded and can give errors if we train for another number of images for training...
                self.loadCkpt_folder = 'IdTrackerDeep/figuresPaper/P1B1/CNN_modelsS/CNN_0/numIndiv_%i/numImages_%i/rep_%i' %(self.groupSizeCNN, 28000, 1)


            for g, self.groupSize in enumerate(self.groupSizes): # Group size for the current training

                for n, self.numImForTrain in enumerate(self.IMDBSizes): # Number of images/individual for training

                    for r, self.rep in enumerate(self.repList): # Repetitions

                        print '\n***********************************************************'
                        if not self.kt:
                            print 'No knowledge transfer'
                        elif self.kt:
                            print 'Knowledge transfer from ', self.loadCkpt_folder
                            print 'GroupSizeCNN, ',self.groupSizeCNN
                        print 'Group size, ', self.groupSize
                        print 'numImForTrain, ', self.numImForTrain
                        print 'Repetition, ', self.rep

                        self.runRepetition()

                        print '*******************************************************************\n'

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
    p.loadIMDBs()
    p.compute()
