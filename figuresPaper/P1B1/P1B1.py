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
os.chdir('../')
sys.path.append('../utils')
sys.path.append('../CNN')

from idTrainer import *
from input_data_cnn import *

class P1B1(object):

    def __init__(self, job = 1, IMDBPath = 'IdTrackerDee/data/36dpf_60indiv_29754ImPerInd_curvaturePortrait_0.hdf5', repList = ['1']):

        def getIMDBNameFromPath(IMDBPath):
            filename, extension = os.path.splitext(IMDBPath)
            IMDBName = '_'.join(filename.split('/')[-1].split('_')[:-1])

            return IMDBName

        if IMDBPath == 'd':
            IMDBPath = 'IdTrackerDeep/data/36dpf_60indiv_29754ImPerInd_curvaturePortrait_0.hdf5
            print 'Using default library, ', IMDBPath

        # Dataset from which the figure is created
        self.IMDBPath = IMDBPath
        self.IMDBName = getIMDBNameFromPath(self.IMDBPath)
        # Main parameters of the figure
        self.groupSizes = [2, 3, 5, 10, 25, 50]
        self.job = job
        self.repList = map(int, repList)
        self.numRepetitions = len(repList)
        self.numImagesToUse = 25000
        self.batchSize = 250
        self.numEpochs = 300
        self.lr = 0.01
        self.keep_prob = 1.0
        # Initialize variables
        self.IndivIndices = {}
        self.ImagesIndices = {}
        self.LossAccDicts = {}
        self.trainAccs = np.ones((self.numRepetitions,len(self.groupSizes))) * np.nan
        self.valAccs = np.ones((self.numRepetitions,len(self.groupSizes))) * np.nan

    def compute(self):

        def runRepetition(self, images, labels, groupSize, groupSizeCounter, rep):

            # Get individual and image indices for this repetition
            print 'Individual indices dictionary', self.IndivIndices
            print 'Number of repetitions already stored', len(self.IndivIndices[groupSize])
            if len(self.IndivIndices[groupSize]) >= rep + 1:
                print 'Restoring individual indices for rep ', rep
                indivIndices = self.IndivIndices[groupSize][rep]

                print 'Restoring images permutation for rep ', rep
                permImages = self.ImagesIndices[groupSize][rep]
            else:
                print 'Seeding the random generator...'
                np.random.seed(rep)

                permIndiv = permuter(self.numIndivImdb,'individualsTrain',[])
                indivIndices = permIndiv[:groupSize]
                self.IndivIndices[groupSize].append(indivIndices)

                permImages = permuter(self.numImagesPerIndiv*groupSize,'imagesTrain',[])
                self.ImagesIndices[groupSize].append(permImages)

            # Get images and labels of the current individuals
            imagesS, labelsS = sliceDatabase(images, labels, indivIndices)

            # Permute images
            imagesS = imagesS[permImages]
            labelsS = labelsS[permImages]

            # Split in train and validation
            X_train, Y_train, X_val, Y_val = splitter(imagesS, labelsS, self.numImagesToUse, groupSize, self.imSize)

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
            ckpt_dir = './P1B1/CNN_models/numIndiv_%s_rep_%s' %(groupSize, rep)
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

            self.LossAccDicts[groupSize].append(lossAccDict)
            self.trainAccs[rep,groupSizeCounter] = lossAccDict['acc'][-1]
            self.valAccs[rep,groupSizeCounter] = lossAccDict['valAcc'][-1]

        # Load IMDB
        _, images, labels, self.imSize, self.numIndivImdb, self.numImagesPerIndiv = loadIMDB(self.IMDBName)

        # Training parameters
        self.channels, self.width, self.height = self.imSize
        self.resolution = np.prod(self.imSize)

        # Check arrays dimensions
        if p.trainAccs.shape[0] < p.numRepetitions or p.trainAccs.shape[1] < len(p.groupSizes):
            print 'Enlarging array to fit new repetitions or groups sizes'
            trainAccs = np.ones((p.numRepetitions,len(p.groupSizes))) * np.nan
            valAccs = np.ones((p.numRepetitions,len(p.groupSizes))) * np.nan
            trainAccs[:p.trainAccs.shape[0],:p.trainAccs.shape[1]] = p.trainAccs
            valAccs[:p.trainAccs.shape[0],:p.trainAccs.shape[1]] = p.valAccs
            p.trainAccs = trainAccs
            p.valAccs = valAccs

        # Main loop
        for groupSizeCounter, groupSize in enumerate(self.groupSizes):
            # Initialize lists for this group size
            if groupSize not in self.IndivIndices.keys():
                print 'Initializing lists IndivIndices, ImagesIndices, LossAccDicts'
                self.IndivIndices[groupSize] = []
                self.ImagesIndices[groupSize] = []
                self.LossAccDicts[groupSize] = []

            for rep in range(self.numRepetitions):
                print '\n***************************************************************'
                print '***************************************************************'
                print 'Group size, ', groupSize
                print 'Repetition, ', rep
                print '\n'

                runRepetition(self, images, labels, groupSize, groupSizeCounter, rep)

                print '\nSaving dictionary...'
                pickle.dump(self.__dict__,open('./P1B1/CNN_models/P1B1Dict.pkl','wb'))
                print '***************************************************************'
                print '***************************************************************\n'

    def plot(self):
        import seaborn as sns
        sns.set(style="white")

        plt.ion()
        plt.figure()
        sns.tsplot(data=self.trainAccs,time = self.groupSizes, err_style="unit_points", color="r",value='accuracy',estimator=np.median)
        sns.tsplot(data=self.valAccs, time = self.groupSizes ,err_style="unit_points", color="b",value='accuracy',estimator=np.median)
        sns.plt.xlabel('Group size')
        sns.plt.ylim((0.5,1.02))
        sns.plt.xlim((0,np.max(self.groupSizes)+1))
        # sns.plt.legend(['training','validation'],loc='lower left') ### FIXME plot legend
        sns.despine()
        sns.plt.show()

        figname = './P1B1/CNN_models/P1B1.pdf'
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
        restore =  getInput('Restore P1B1', 'Do you wanna restore P1B1 from a dictionary (P1B1Dict.pkl)? y/[n]')
        # restore = 'n'
        if restore == 'y':
            p = P1B1()
            P1B1Dict = pickle.load(open('./P1B1/CNN_models/P1B1Dict.pkl','rb'))
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
        p = P1B1(job = int(sys.argv[2]), IMDBPath = sys.argv[3], repList = sys.argb[4:])
        p.compute()
