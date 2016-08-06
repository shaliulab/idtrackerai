import pandas as pd
import numpy as np
from tf_utils import *


def loadData(numTrain, numVal, numTest):

    ''' load data '''
    df = pd.read_csv('./data/pnas.1601827113.sd01.csv')

    ''' take parameters '''
    cases = df['case'].unique()
    numCases = len(cases)
    doctors = df['diagnostician'].unique()
    numDoctors = len(doctors)

    ''' build truth vector '''
    truth = np.array(df['melanoma'][:numCases])
    truth = dense_to_one_hot(truth, n_classes=2)


    ''' build decisions matrix '''
    decisions = []
    for case in cases:
        decisions.append(np.array(df['decision'].loc[df['case']==case]))

    decisions = np.array(decisions)
    decisions[ decisions == 0] = -1


    """ permute data and labels """

    # perm = np.random.permutation(numCases)
    # truth = truth[perm]
    # decisions = decisions[perm]

    ''' split data in train, validation and test '''
    labelsTrain = truth[:numTrain]
    dataTrain = decisions[:numTrain]
    labelsVal = truth[numTrain:numTrain + numVal]
    dataVal = decisions[numTrain:numTrain + numVal]
    labelsTest = truth[numTrain+numVal:numTrain + numVal + numTest]
    dataTest = decisions[numTrain+numVal:numTrain + numVal + numTest]

    return labelsTrain, dataTrain, labelsVal, dataVal, labelsTest, dataTest
