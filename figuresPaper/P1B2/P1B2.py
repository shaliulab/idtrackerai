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

class P1B2(object):

    def __init__(self, job = 1, IMDBPath = 'IdTrackerDeep/data/25dpf_60indiv_26142imperind_curvatureportrait2_0.hdf5', repList = ['1']):

        def getIMDBNameFromPath(IMDBPath):
            filename, extension = os.path.splitext(IMDBPath)
            IMDBName = '_'.join(filename.split('/')[-1].split('_')[:-1])

            return IMDBName

        if IMDBPath == 'd':
            IMDBPath = 'IdTrackerDeep/data/25dpf_60indiv_26142imperind_curvatureportrait2_0.hdf5'
            print 'Using default library, ', IMDBPath

        # Dataset from which the figure is created
        self.IMDBPath = IMDBPath
        self.IMDBName = getIMDBNameFromPath(self.IMDBPath)
        # Main parameters of the figure
        self.groupSizesCNN = [2, 5, 10, 25, 50]
        self.groupSizes = [2, 5, 10, 25, 50]
        self.job = job
        self.repList = map(int, repList)
        self.numRepetitions = len(repList)
        self.numImagesToUse = 25000
        self.batchSize = 250
        self.numEpochs = 5
        self.lr = 0.01
        self.keep_prob = 1.0
        # Initialize variables
        self.IndivIndices = {}
        self.ImagesIndices = {}
        self.LossAccDicts = {}
        self.trainAccs = np.ones((self.numRepetitions,len(self.groupSizes),len(self.groupSizesCNN))) * np.nan
        self.valAccs = np.ones((self.numRepetitions,len(self.groupSizes),len(self.groupSizesCNN))) * np.nan

    def compute(self):

        def runRepetition(self, loadCkpt_folder, images, labels, groupSize, groupSizeCounter, groupSizeCNN, groupSizeCNNCounter, rep, repCounter):

            # Get individual and image indices for this repetition
            print 'Individual indices dictionary', self.IndivIndices
            print 'Number of repetitions already stored', len(self.IndivIndices[groupSizeCNN][groupSize])
            if len(self.IndivIndices[groupSizeCNN][groupSize]) >= rep + 1:
                print 'Restoring individual indices for rep ', rep
                indivIndices = self.IndivIndices[groupSizeCNN][groupSize][rep]

            else:
                print 'Seeding the random generator...'
                np.random.seed(rep)

                permIndiv = permuter(self.numIndivImdb,'individualsTrain',[])
                indivIndices = permIndiv[:groupSize]


            # Get images and labels of the current individuals
            imagesS, labelsS = sliceDatabase(images, labels, indivIndices)

            if len(self.IndivIndices[groupSizeCNN][groupSize]) >= rep + 1:
                print 'Restoring images permutation for rep ', rep
                permImages = self.ImagesIndices[groupSizeCNN][groupSize][rep]

            else:
                permImages = permuter(len(labelsS),'imagesTrain',[])

            self.IndivIndices[groupSizeCNN][groupSize].append(indivIndices)
            self.ImagesIndices[groupSizeCNN][groupSize].append(permImages)

            # Permute images
            imagesS = imagesS[permImages]
            labelsS = labelsS[permImages]

            # Split in train and validation
            X_train, Y_train, X_val, Y_val = splitter(imagesS, labelsS, self.numImagesToUse, groupSize, self.imSize)

            print 'len Y_train + Y_val, ', len(Y_train) + len(Y_val)

            # check train's dimensions
            cardTrain = int(np.ceil(np.true_divide(np.multiply(self.numImagesToUse,9),10)))*groupSize
            dimTrainL = (cardTrain, groupSize)
            dimTrainI = (cardTrain, images.shape[2]*images.shape[3])
            dimensionChecker(X_train.shape, dimTrainI)
            dimensionChecker(Y_train.shape, dimTrainL)

            # check val's dimensions
            cardVal = int(np.ceil(np.true_divide(self.numImagesToUse,10)))*groupSize
            dimValL = (cardVal, groupSize)
            dimValI = (cardVal, images.shape[2] * images.shape[3])
            dimensionChecker(X_val.shape, dimValI)
            dimensionChecker(Y_val.shape, dimValL)

            # Update ckpt_dir and loadCkpt_folder
            ckpt_dir = 'IdTrackerDeep/figuresPaper/P1B2/CNN_models/CNNModel_%s_numIndiv_%s_rep_%s' %(groupSizeCNN, groupSize, rep)

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

            print 'counters, ', repCounter, groupSizeCounter, groupSizeCNNCounter
            print 'shape, ', self.trainAccs.shape
            self.LossAccDicts[groupSizeCNN][groupSize].append(lossAccDict)
            self.trainAccs[repCounter,groupSizeCounter,groupSizeCNNCounter] = lossAccDict['acc'][-1]
            self.valAccs[repCounter,groupSizeCounter,groupSizeCNNCounter] = lossAccDict['valAcc'][-1]

        # Load IMDB
        _, images, labels, self.imSize, self.numIndivImdb, self.numImagesPerIndiv = loadIMDB(self.IMDBName)

        # Training parameters
        self.channels, self.width, self.height = self.imSize
        self.resolution = np.prod(self.imSize)

        # Check arrays dimensions
        if p.trainAccs.shape[0] < p.numRepetitions or p.trainAccs.shape[1] < len(p.groupSizes) or p.trainAccs.shape[2] < len(p.groupSizesCNN):
            print 'Enlarging array to fit new repetitions or groups sizes'
            trainAccs = np.ones((p.numRepetitions,len(p.groupSizes),len(p.groupSizesCNN))) * np.nan
            valAccs = np.ones((p.numRepetitions,len(p.groupSizes),len(p.groupSizesCNN))) * np.nan
            trainAccs[:p.trainAccs.shape[0],:p.trainAccs.shape[1],:p.trainAccs.shape[2]] = p.trainAccs
            valAccs[:p.trainAccs.shape[0],:p.trainAccs.shape[1],:p.trainAccs.shape[2]] = p.valAccs
            p.trainAccs = trainAccs
            p.valAccs = valAccs

        # Main loop
        print 'repetitions, ', self.repList
        for groupSizeCNNCounter, groupSizeCNN in enumerate(self.groupSizesCNN):
            loadCkpt_folder = 'IdTrackerDee/figuresPaper/P1B1/CNN_models/numIndiv_%s_rep_%s' %(groupSizeCNN, 0) # We will use the first repetition from P1B1 as a model for the knowledge transfer

            if groupSizeCNN not in self.IndivIndices.keys():
                print 'Initializing dictionaries IndivIndices, ImagesIndices, LossAccDicts for groupSizeCNN'
                self.IndivIndices[groupSizeCNN] = {}
                self.ImagesIndices[groupSizeCNN] = {}
                self.LossAccDicts[groupSizeCNN] = {}

            for groupSizeCounter, groupSize in enumerate(self.groupSizes):
                # Initialize lists for this group size
                if groupSize not in self.IndivIndices[groupSizeCNN].keys():
                    print 'Initializing lists IndivIndices, ImagesIndices, LossAccDicts for groupSizeCNN'
                    self.IndivIndices[groupSizeCNN][groupSize] = []
                    self.ImagesIndices[groupSizeCNN][groupSize] = []
                    self.LossAccDicts[groupSizeCNN][groupSize] = []

                for repCounter, rep in enumerate(self.repList):
                    print '\n***************************************************************'
                    print '***************************************************************'
                    print 'Loading model from ', 'IdTrackerDeep_condor/figuresPaper/P1B1/CNN_models/numIndiv_%s_rep_%s' %(groupSizeCNN, 0)
                    print 'Group size, ', groupSize
                    print 'Repetition, ', rep
                    print '\n'

                    runRepetition(self, loadCkpt_folder, images, labels, groupSize, groupSizeCounter, groupSizeCNN, groupSizeCNNCounter, rep, repCounter)

                    print '\nSaving dictionary...'
                    pickle.dump(self.__dict__,open('IdTrackerDee/figuresPaper/P1B2/CNN_models/P1B2Dict_%i.pkl' %self.job,'wb'))
                    print '***************************************************************'
                    print '***************************************************************\n'
    def plot(self):
        print '\nPlotting data...'
        import seaborn as sns
        sns.set(style="white")

        trainsAccsAv = np.nanmedian(p.trainAccs,axis = 0)
        valAccsAv = np.nanmedian(p.valAccs,axis = 0)

        plt.ion()
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        sns.heatmap(trainsAccsAv, annot=True, ax=ax1,vmin=.97, vmax=1, cbar=False)
        ax1.set_xlabel('Group size CNN model')
        ax1.set_ylabel('Group size dataSet')
        ax1.set_xticklabels(p.groupSizesCNN)
        ax1.set_yticklabels(p.groupSizes[::-1])
        ax1.invert_yaxis()
        ax1.set_title('Training')


        sns.heatmap(valAccsAv, annot=True, ax=ax2,vmin=.97, vmax=1)
        ax2.set_xlabel('Group size CNN model')
        ax2.set_xticklabels(p.groupSizesCNN)
        ax2.set_yticklabels([])
        ax2.set_title('Validation')
        ax2.invert_yaxis()

        sns.plt.show()

        figname = 'IdTrackerDeep/figuresPaper/P1B2/CNN_models/P1B2.pdf'
        plt.savefig(figname)

if __name__ == '__main__':
    '''
    argv[1]: useCondor (bool flag)
    argv[2]: job number
    argv[3]: dataBase
    argv[4:]: repetitions
    '''

    useCondor = int(sys.argv[1])
    if not useCondor:
        restore =  getInput('Restore P1B2', 'Do you wanna restore P1B2 from a dictionary (P1B2Dict.pkl)? y/[n]')
        if restore == 'y':
            p = P1B2()
            P1B1Dict = pickle.load(open('IdTrackerDeep/figuresPaper/P1B2/CNN_models/P1B2Dict_1.pkl','rb'))
            p.__dict__.update(P1B1Dict)
            print 'The attributes of the class are:'
            # pprint(p.__dict__)
            print 'The group sizes are, ', p.groupSizes
            print 'The number of repetitions are, ', p.numRepetitions
            computeFlag =  getInput('Plot or compute', 'Do you wanna compute more repetitions or add more groupSizes? y/[n]')
            if computeFlag == 'y':
                repetitions =  getInput('Repetition', 'How many repetitions do you want (current %i)?' %p.numRepetitions)
                repetitions = int(repetitions)
                p.numRepetitions = repetitions

                groupSizes = getInput('GroupSizes', 'Insert the group sizes separated by comas (current ' + str(p.groupSizes) + ' )')
                groupSizes = [int(i) for i in groupSizes.split(',')]
                p.groupSizes = groupSizes

                groupSizesCNN = getInput('GroupSizes', 'Insert the group sizes (CNN) separated by comas (current ' + str(p.groupSizesCNN) + ' )')
                groupSizesCNN = [int(i) for i in groupSizesCNN.split(',')]
                p.groupSizesCNN = groupSizesCNN

                numEpochs = getInput('numEpochs', 'Insert the number of Epochs (current ' + str(p.numEpochs) + ' )')
                p.numEpochs = int(numEpochs)

                p.compute()
                p.plot()

            else:
                p.plot()

        elif restore == 'n' or restore == '':
            # IMDBPath = selectFile()
            # print 'Computing with default values'
            # p = P1B2(IMDBPath=IMDBPath)
            p = P1B2()
            pprint(p.__dict__)
            p.compute()
            p.plot()

    elif useCondor:
        p = P1B2(int(sys.argv[2]),sys.argv[3],sys.argv[4:])
