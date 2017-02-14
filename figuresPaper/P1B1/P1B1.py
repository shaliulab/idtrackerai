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
sys.path.append('../utils')
sys.path.append('../CNN')

from idTrainer import *
from input_data_cnn import *

def runRepetition(P1B1Dict, trainDict,  imSize, images, labels, numIndivImdb, numImagesPerIndiv, numIndiv, i, rep):

    # Get individual indices for this repetition
    permIndiv = permuter(numIndivImdb,'individualsTrain',[])
    indivIndices = permIndiv[:numIndiv]
    P1B1Dict['IndivIndices'][str(numIndiv)].append(indivIndices)

    # Get images and labels of the current individuals
    imagesS, labelsS = sliceDatabase(images, labels, indivIndices)

    # Get images permutation for this set of images
    permImages = permuter(numImagesPerIndiv*numIndiv,'imagesTrain',[])
    P1B1Dict['ImagesIndices'][str(numIndiv)].append(permImages)

    # Permute images
    imagesS = imagesS[permImages]
    labelsS = labelsS[permImages]

    # Split in train and validation
    X_train, Y_train, X_val, Y_val = splitter(imagesS, labelsS, P1B1Dict['numImagesToUse'], numIndiv, imSize)

    # check train's dimensions
    cardTrain = int(np.ceil(np.true_divide(np.multiply(P1B1Dict['numImagesToUse'],9),10)))*numIndiv
    dimTrainL = (cardTrain, numIndiv)
    dimTrainI = (cardTrain, images.shape[2]*images.shape[3])
    dimensionChecker(X_train.shape, dimTrainI)
    dimensionChecker(Y_train.shape, dimTrainL)

    # check val's dimensions
    cardVal = int(np.ceil(np.true_divide(P1B1Dict['numImagesToUse'],10)))*numIndiv
    dimValL = (cardVal, numIndiv)
    dimValI = (cardVal, images.shape[2] * images.shape[3])
    dimensionChecker(X_val.shape, dimValI)
    dimensionChecker(Y_val.shape, dimValL)

    # Update ckpt_dir and loadCkpt_folder
    ckpt_dir = './P1B1/numIndiv_%s_rep_%s' %(numIndiv, rep)
    loadCkpt_folder = ''

    # Compute index batches
    numImagesT = Y_train.shape[0]
    numImagesV = Y_val.shape[0]
    Tindices, Titer_per_epoch = get_batch_indices(numImagesT,trainDict['batchSize'])
    Vindices, Viter_per_epoch = get_batch_indices(numImagesV,trainDict['batchSize'])

    # Run training
    lossAccDict = run_training(X_train, Y_train, X_val, Y_val,
                                trainDict['width'], trainDict['height'], trainDict['channels'], numIndiv, trainDict['resolution'],
                                ckpt_dir, loadCkpt_folder,
                                trainDict['batchSize'], trainDict['numEpochs'],
                                Tindices, Titer_per_epoch,
                                Vindices, Viter_per_epoch,
                                trainDict['keep_prob'],trainDict['lr'])

    P1B1Dict['LossAccDicts'][str(numIndiv)].append(lossAccDict)
    P1B1Dict['trainAccs'][rep,i] = lossAccDict['acc'][-1]
    P1B1Dict['valAccs'][rep,i] = lossAccDict['valAcc'][-1]

    return P1B1Dict

def computeDataP1B1(P1B1Dict):
    IMDBname = P1B1Dict['IMDBname']
    trainDict = P1B1Dict['trainDict']

    P1B1Dict['IndivIndices'] = {}
    P1B1Dict['ImagesIndices'] = {}
    P1B1Dict['LossAccDicts'] = {}
    P1B1Dict['trainAccs'] = np.ones((P1B1Dict['numRepetitions'],len(P1B1Dict['numIndivList']))) * np.nan
    P1B1Dict['valAccs'] = np.ones((P1B1Dict['numRepetitions'],len(P1B1Dict['numIndivList']))) * np.nan


    # Load IMDB
    databaseInfo, images, labels, imSize, numIndivImdb, numImagesPerIndiv = loadIMDB(IMDBname)
    # P1B1Dict['IMDBInfo'] = databaseInfo

    # Training parameters
    channels, width, height = imSize
    resolution = np.prod(imSize)
    trainDict['channels'] = channels
    trainDict['width'] = width
    trainDict['height'] = height
    trainDict['resolution'] = resolution

    # Main loop
    for i, numIndiv in enumerate(P1B1Dict['numIndivList']):
        P1B1Dict['IndivIndices'][str(numIndiv)] = []
        P1B1Dict['ImagesIndices'][str(numIndiv)] = []
        P1B1Dict['LossAccDicts'][str(numIndiv)] = []

        for rep in range(P1B1Dict['numRepetitions']):

            P1B1Dict = runRepetition(P1B1Dict, trainDict, imSize, images, labels, numIndivImdb, numImagesPerIndiv, numIndiv, i, rep)

            print '\nSaving dictionary...'
            pickle.dump(P1B1Dict,open('./CNN_models/P1B1Dict.pkl','wb'))

    return P1B1Dict

def plotP1B1(P1B1Dict):
    import seaborn as sns
    sns.set(style="white")
    trainAccs = P1B1Dict['trainAccs']
    valAccs = P1B1Dict['valAccs']
    numIndivList = P1B1Dict['numIndivList']

    sns.tsplot(data=trainAccs,time = numIndivList, err_style="unit_points", color="r",value='accuracy',estimator=np.median)
    sns.tsplot(data=valAccs, time = numIndivList ,err_style="unit_points", color="b",value='accuracy',estimator=np.median)
    sns.plt.xlabel('numer of individuals')
    sns.plt.ylim((0.5,1.02))
    sns.plt.xlim((1,51))
    plt.legend(['training','validation'],loc='lower left')
    sns.despine()
    sns.plt.show()


if __name__ == '__main__':

    trainDict = {
            'batchSize': 250,
            'numEpochs': 300,
            'lr': 0.01,
            'keep_prob': 1.,
            }

    P1B1Dict = {
            'IMDBname': '36dpf_60indiv_29754ImPerInd_curvaturePortrait', # the IMDB is assumed to be in the folder data
            'numIndivList': [2, 3, 5, 10, 25, 50],
            'numRepetitions': 5,
            'numImagesToUse': 29000, # 10% will be used for validation
            'trainDict': trainDict,
                }

    compute = False
    if compute:
        P1B1Dict = computeDataP1B1(P1B1Dict)
    print '\nPlotting data...'
    P1B1Dict = pickle.load(open('./P1B1/P1B1Dict.pkl','rb'))
    plotP1B1(P1B1Dict)
