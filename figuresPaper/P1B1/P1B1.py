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
        self.IMDBSizes = [20,50,100,250,500,750,1000,25000]
        # self.IMDBSizes = [20,50,100,250]
        self.numIMDBSizes = len(self.IMDBSizes)

        # Initialize figure arrays
        self.trainAccs = np.ones((self.numIMDBSizes, self.numGroups, self.numGroupsCNN, self.numRepetitions)) * np.nan
        self.valAccs = np.ones((self.numIMDBSizes, self.numGroups, self.numGroupsCNN, self.numRepetitions)) * np.nan

        # CNN parameters
        if 'KT' in self.condition:
            self.kt = True
        else:
            self.kt = False
        self.batchSize = 250
        self.numEpochs = 5000
        self.lr = 0.01
        self.keep_prob = 1.0
        if 'V' in self.condition:
            self.checkLearningFlag = True
        else:
            self.checkLearningFlag = False

	if IMDBPath == 'd':
	    if self.kt == False:
            	IMDBPath = 'IdTrackerDeep/data/36dpf_60indiv_29754ImPerInd_curvaturePortrait_0.hdf5'
	    elif self.kt == True:
            	IMDBPath = 'IdTrackerDeep/data/25dpf_60indiv_26142imperind_curvatureportrait2_0.hdf5'

            print 'Using default library, ', IMDBPath

        # Dataset from which to load the images for training
        self.IMDBPath = IMDBPath
        self.IMDBName = getIMDBNameFromPath(self.IMDBPath)

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

            # Get correlated images
            if 'C' in self.condition:
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

                print 'Extracting correlated images...'
                print 'number of images before extracting correlated ones: ', len(imagesS)
                print 'number of labels before extracting correlated ones: ', len(labelsS)

                imagesS, labelsS, firstFrameIndices = getCorrelatedImages(imagesS,labelsS,numImagesToUse)

                print 'number of images after extracting correlated ones: ', len(imagesS)
                print 'number of labels after extracting correlated ones: ', len(labelsS)

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
                ckpt_dir = 'IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/numIndiv_%i/numImages_%i/rep_%i' %(self.condition,groupSize, numImagesToUse, rep)
            elif self.kt:
                ckpt_dir = 'IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/CNN_%i/numIndiv_%i/numImages_%i/rep_%i' %(self.condition, groupSizeCNN, groupSize, numImagesToUse, rep)

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
                                        checkLearningFlag = self.checkLearningFlag)

            print 'Time in seconds, ', np.sum(lossAccDict['epochTime'])
            self.LossAccDicts[groupSizeCNN][groupSize][numImagesToUse].append(lossAccDict)
            self.trainAccs[n,g,gCNN,r] = lossAccDict['acc'][-1]
            self.valAccs[n,g,gCNN,r] = lossAccDict['valAcc'][-1]

        # Load IMDB
        _, images, labels, self.imSize, self.numIndivImdb, self.numImagesPerIndiv = loadIMDB(self.IMDBName)

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
                loadCkpt_folder = 'IdTrackerDeep/figuresPaper/P1B1/CNN_modelsS/numIndiv_%i/numImages_%i/rep_%i' %(self.groupSizesCNN[gCNN], 25000, 1)
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
                            pickle.dump(self.__dict__,open('IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/P1B1Dict_job_%i.pkl' %(self.condition, self.job),'wb'))
                        elif self.kt:
                            pickle.dump(self.__dict__,open('IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/P1B1Dict_job_%i.pkl' %(self.condition, self.job),'wb'))
                        print '******************************************************************************************************************************'
                        print '******************************************************************************************************************************\n'

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
            fig.suptitle('Training from strach')
        elif self.condition == 'KT':
            fig.suptitle('Knowledge transfer from CNN of 50 indiv 25000 images')
        elif self.condition == 'KTC':
            fig.suptitle('Knowledge transfer from CNN of 50 indiv 25000 images (correlated images)')
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
    argv[1]: useCondor (bool flag)
    argv[2]: job number
    argv[3]: dataBase if 'd' is default
    argv[4]: repetitions
    argv[5]: groupSizeCNN
    argv[6]: condition: 'S'-scratch, 'KT'-knowledgeT, 'KTC'-knowledgeTCorrelated
    P1B1.py 1 1 d 1_2 2_5 S (useCondor,job1,default library,repetitions[1 2],groupSizesCNN[2 5],from scratch)

    '''

    plotFlag = int(sys.argv[1])
    if not plotFlag:

        P1B1DictPath = 'IdTrackerDeep/figuresPaper/P1B1/CNN_models%s/P1B1Dict_job_1.pkl' %sys.argv[6]
        print 'P1B1DictPath, ', P1B1DictPath
        condition = sys.argv[6]
        P1B1DictsPaths = scanFolder(P1B1DictPath)
        manyDicts = False
        if len(P1B1DictsPaths) > 1:
            manyDicts = True
        print 'Loading dictionary to update p object...'
        P1B1Dict = pickle.load(open(P1B1DictPath,'rb'))
        p = P1B1(condition = condition)
        p.__dict__.update(P1B1Dict)
        print 'The group sizes are, ', p.groupSizes
        print 'The IMDB sizes are, ', p.IMDBSizes
        if manyDicts:
            p.joinDicts(P1B1Dict=P1B1DictPath)
        p.computeTimes(accTh = 0.8)
        p.plotResults()
        p.plotResults2()

    elif plotFlag:
        p = P1B1(job = int(sys.argv[2]), IMDBPath = sys.argv[3], repList = sys.argv[4], groupSizesCNN = sys.argv[5], condition = sys.argv[6])
        p.compute()
