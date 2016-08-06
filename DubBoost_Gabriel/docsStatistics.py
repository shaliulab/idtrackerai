import os
import sys
sys.path.append('../utils')
from loadData import *
import argparse
import numpy as np
from tf_utils import *

if __name__ == '__main__':

    # prep for args
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', default='matlabexports/imdb_5indiv_3000_1000_s1.mat', type = str)
    parser.add_argument('--num_train', default = 54, type = int)
    parser.add_argument('--num_val', default = 0, type = int)
    parser.add_argument('--num_test', default = 0, type = int)
    args = parser.parse_args()

    # path = args.dataset
    num_train = args.num_train
    num_test = args.num_test
    num_valid = args.num_val

    Y_train, X_train, Y_valid, X_valid, Y_test, X_test = loadData(num_train, num_valid, num_test)
    Y_train = Y_train[:,1]
    Y_valid = Y_valid[:,1]
    Y_test = Y_test[:,1]

    Y_comp = np.tile(Y_train, [40, 1])

    X_train[X_train == -1] = 0
    X_train = X_train.T

    error = (Y_comp == X_train)*1
    accPerDoc = np.sum(error, axis =1)
    # print X_train[0], len(X_train[0])
    # print Y_train == 0

    PTruth = Y_train == 1
    NTruth = Y_train == 0

    TP = [np.sum(np.logical_and(X_t == 1, Y_train == 1)) for X_t in X_train]
    FP = [np.sum(np.logical_and(X_t == 1, Y_train == 0)) for X_t in X_train]
    FN = [np.sum(np.logical_and(X_t == 0, Y_train == 1)) for X_t in X_train]
    TN = [np.sum(np.logical_and(X_t == 0, Y_train == 0)) for X_t in X_train]

    ''' print all the stats '''
    print 'number of cases: ' + str(len(Y_train))
    print 'number of doctors: ' + str(X_train.shape[0])
    print 'number of negative diagnosis (no cancer): ' + str(sum(NTruth))  # check!
    print 'number of positive diagnosis (cancer): ' + str(sum(PTruth))  # check!

    print 'individual accuracy (%):'
    percAcc = np.true_divide(accPerDoc, float(len(Y_train)))*100
    print percAcc
    print 'average accuracy: ' + str(np.mean(percAcc))

    print 'true positive per Doc'
    truePos = np.true_divide(TP, np.sum(PTruth))*100
    print truePos
    print 'average true positive: ' + str(np.mean(truePos))

    print 'true negative per Doc'
    trueNeg = np.true_divide(TN, np.sum(NTruth))*100
    print trueNeg
    print 'average true negative: ' + str(np.mean(trueNeg))

    # print 'false positive per Doc'
    # falsePos =  FP#np.true_divide(FP, np.sum(NTruth))*100
    # print falsePos
    # print 'average false positive: ' + str(np.mean(falsePos))
    #
    # print 'false negative per Doc'
    # falseNeg =  FN#np.true_divide(FN, np.sum(PTruth))*100
    # print falseNeg
    # print 'average false negative: ' + str(np.mean(falseNeg))
    # # print TPperDoc
