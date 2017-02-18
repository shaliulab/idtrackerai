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

class P1B1(object):

    def __init__(self, job = 1, IMDBPath = 'IdTrackerDeep/data/36dpf_60indiv_29754ImPerInd_curvaturePortrait_0.hdf5', repList = ['1']):

        def getIMDBNameFromPath(IMDBPath):
            filename, extension = os.path.splitext(IMDBPath)
            IMDBName = '_'.join(filename.split('/')[-1].split('_')[:-1])

            return IMDBName

        if IMDBPath == 'd':
            IMDBPath = 'IdTrackerDeep/data/36dpf_60indiv_29754ImPerInd_curvaturePortrait_0.hdf5'
            print 'Using default library, ', IMDBPath

        # Dataset from which the figure is created
        self.IMDBPath = IMDBPath
        self.IMDBName = getIMDBNameFromPath(self.IMDBPath)
        # Main parameters of the figure
        self.job = job
        self.groupSizes = [2, 5, 10, 25, 50]
        self.numGroups = len(self.groupSizes)
        self.repList = map(int, repList)
        self.numRepetitions = len(repList)
        self.IMDBSizes = [20,50,100,500,1000,10000,25000]
        self.numIMDBSizes = len(self.IMDBSizes)
        self.batchSize = 250
        self.numEpochs = 3
        self.lr = 0.01
        self.keep_prob = 1.0
        # Initialize variables
        self.IndivIndices = {}
        self.ImagesIndices = {}
        self.LossAccDicts = {}
        self.trainAccs = np.ones((self.numIMDBSizes,self.numGroups,self.numRepetitions)) * np.nan
        self.valAccs = np.ones((self.numIMDBSizes,self.numGroups,self.numRepetitions)) * np.nan

    def compute(self):

        def runRepetition(self, images, labels, g, n, r):
            groupSize = self.groupSizes[g]
            numImagesToUse = self.IMDBSizes[n]
            rep = self.repList[r]

            # Get individuals for this repetition
            print 'Individual indices per group size', self.IndivIndices
            print 'Number of repetitions already computed', len(self.IndivIndices[groupSize])
            if len(self.IndivIndices[groupSize]) >= r + 1:
                print 'Restoring individual indices for rep ', rep
                indivIndices = self.IndivIndices[groupSize][r]

            else:
                print 'Seeding the random generator...'
                np.random.seed(rep)

                permIndiv = permuter(self.numIndivImdb,'individualsTrain',[])
                indivIndices = permIndiv[:groupSize]
                self.IndivIndices[groupSize].append(indivIndices)


            # Get images of the current individuals
            imagesS, labelsS = sliceDatabase(images, labels, indivIndices)

            # Get permutation of for this repetition
            if len(self.ImagesIndices[groupSize][numImagesToUse]) >= r + 1:
                print 'Restoring images permutation for rep ', rep
                permImages = self.ImagesIndices[groupSize][numImagesToUse][r]

            else:
                permImages = permuter(len(labelsS),'imagesTrain',[])
                self.ImagesIndices[groupSize][numImagesToUse].append(permImages)

            # Permute images
            imagesS = imagesS[permImages]
            labelsS = labelsS[permImages]

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

            # Update ckpt_dir and loadCkpt_folder
            ckpt_dir = 'IdTrackerDeep/figuresPaper/P1B1/CNN_models/numIndiv_%i/numIndiv_%i_numImages_%i_rep_%i' %(groupSize, groupSize, numImagesToUse, rep)
            loadCkpt_folder = ''

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
            self.LossAccDicts[groupSize][numImagesToUse].append(lossAccDict)
            self.trainAccs[n,g,r] = lossAccDict['acc'][-1]
            self.valAccs[n,g,r] = lossAccDict['valAcc'][-1]

        # Load IMDB
        _, images, labels, self.imSize, self.numIndivImdb, self.numImagesPerIndiv = loadIMDB(self.IMDBName)

        # Training parameters
        self.channels, self.width, self.height = self.imSize
        self.resolution = np.prod(self.imSize)

        # Check arrays dimensions
        print 'Checking array dimension'
        if self.trainAccs.shape[0] < self.numIMDBSizes or self.trainAccs.shape[1] < self.numGroups or self.trainAccs.shape[2] < self.numRepetitions:
            print 'Enlarging array to fit new repetitions or groups sizes'
            trainAccs = np.ones((self.numIMDBSizes,self.numGroups,self.numRepetitions)) * np.nan
            valAccs = np.ones((self.numIMDBSizes,self.numGroups,self.numRepetitions)) * np.nan
            trainAccs[:self.trainAccs.shape[0],:self.trainAccs.shape[1],:self.trainAccs.shape[2]] = self.trainAccs
            valAccs[:self.trainAccs.shape[0],:self.trainAccs.shape[1],:self.trainAccs.shape[2]] = self.valAccs
            self.trainAccs = trainAccs
            self.valAccs = valAccs

        # Main loop
        for g in range(self.numGroups):

            if self.groupSizes[g] not in self.IndivIndices.keys():
                print 'Initializing lists IndivIndices, ImagesIndices, LossAccDicts for the new groupSize, ', self.groupSizes[g]
                self.IndivIndices[self.groupSizes[g]] = []
                self.ImagesIndices[self.groupSizes[g]] = {}
                self.LossAccDicts[self.groupSizes[g]] = {}

            for n in range(self.numIMDBSizes):

                if self.IMDBSizes[n] not in self.ImagesIndices[self.groupSizes[g]].keys():
                    print 'Initializing lists ImagesIndices, LossAccDicts for the new IMDBSize', self.IMDBSizes[n]
                    self.ImagesIndices[self.groupSizes[g]][self.IMDBSizes[n]] = []
                    self.LossAccDicts[self.groupSizes[g]][self.IMDBSizes[n]] = []

                for r in range(self.numRepetitions):
                    print '\n******************************************************************************************************************************'
                    print '******************************************************************************************************************************'
                    print 'Group size, ', self.groupSizes[g]
                    print 'numImagesToUse, ', self.IMDBSizes[n]
                    print 'Repetition, ', self.repList[r]

                    runRepetition(self, images, labels, g, n, r)

                    print '\nSaving dictionary...'
                    pickle.dump(self.__dict__,open('IdTrackerDeep/figuresPaper/P1B1/CNN_models/P1B1Dict_%i.pkl' %self.job,'wb'))
                    print '******************************************************************************************************************************'
                    print '******************************************************************************************************************************\n'

    def plot(self):
        print '\nPlotting data...'
        import seaborn as sns
        sns.set(style="white")

        trainsAccsAv = np.nanmedian(p.trainAccs,axis = 2)
        valAccsAv = np.nanmedian(p.valAccs,axis = 2)

        plt.ion()
        fig, ax1 = plt.subplots()
        sns.heatmap(valAccsAv, annot=True, ax=ax1,vmin=0., vmax=1, cbar=False)
        ax1.set_xlabel('Group size')
        ax1.set_ylabel('Num im/indiv')
        ax1.set_xticklabels(p.groupSizes)
        ax1.set_yticklabels(p.IMDBSizes[::-1])
        ax1.invert_yaxis()
        ax1.set_title('Validation')

        sns.plt.show()

        figname = 'IdTrackerDeep/figuresPaper/P1B1/CNN_models/P1B1.pdf'
        plt.savefig(figname)

if __name__ == '__main__':
    '''
    argv[1]: useCondor (bool flag)
    argv[2]: job number
    argv[3]: dataBase if 'd' is default
    argv[4:]: repetitions
    P1B1.py 1 1 d 1 2 (useCondor,job1,default library,repetitions[1 2])
    '''

    useCondor = int(sys.argv[1])
    if not useCondor:
        restore =  getInput('Restore P1B1', 'Do you wanna restore P1B1 from a dictionary (P1B1Dict.pkl)? y/[n]')
        # restore = 'n'
        if restore == 'y':
            p = P1B1()
            print 'Loading P1B1 dictionary'
            P1B1Dict = pickle.load(open('IdTrackerDeep/figuresPaper/P1B1/CNN_models/P1B1Dict_1.pkl','rb'))
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

                numEpochs = getInput('numEpochs', 'Insert the number of Epochs (current ' + str(p.numEpochs) + ' )')
                p.numEpochs = int(numEpochs)

                p.compute()
                p.plot()

            else:
                p.plot()

        elif restore == 'n' or restore == '':
            IMDBPath = selectFile()
            # print 'Computing with default values'
            p = P1B1(IMDBPath = IMDBPath)
            pprint(p.__dict__)
            p.compute()
            p.plot()

    elif useCondor:
        p = P1B1(job = int(sys.argv[2]), IMDBPath = sys.argv[3], repList = sys.argv[4:])
        p.compute()
