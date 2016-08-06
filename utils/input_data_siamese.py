import numpy as np
import h5py
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
import itertools
from py_utils import *

def dataHelper(path, num_train, num_test, num_valid):
    Fish = {}
    with h5py.File(path, 'r') as f:
        Fish['data'] = f['images']['data'][()]
        Fish['labels'] = f['images']['labels'][()]
        print 'database loaded'

    # Get data and reversed to generate pairs
    data1 = Fish['data'].astype(np.float32)
    data2 = data1[::-1]
    # print 'data 1 *****************'
    # print data1[0]
    # print 'data 2 *****************'
    # print data2[len(data2)-1]
    # get image dimensions
    imsize = data1[0].shape
    resolution = np.prod(imsize)
    # ditto for the labels
    label = Fish['labels'].astype(np.int32)
    label1 = np.squeeze(label)
    label2 = label1[::-1]
    # print label1
    # print "-------------------->8"
    # print label2

    # compute number of individuals (classes)
    numIndiv = len(list(set(label1)))
    # compute match/mismatch label for each pair
    label = np.array(label1 == label2, dtype=np.int32)
    # print "-------------------->8"
    # print label

    print 'splitting data in train, test and validation'
    N = np.round(num_train*numIndiv/2) # of training data
    N_test = np.round(num_test*numIndiv/2) # of test data
    N_val = np.round(num_valid*numIndiv/2) # validation data

    # print N, N_test, N_val

    X1_train = data1[:N]
    X1_test = data1[N:N+N_test]
    X1_valid = data1[N+N_test:N+N_test+N_val]
    X2_train = data2[:N]
    X2_test = data2[N:N+N_test]
    X2_valid = data2[N+N_test:N+N_test+N_val]
    Y_train = label[:N]
    Y_test = label[N:N+N_test]
    Y_valid = label[N+N_test:N+N_test+N_val]

    # reshape images
    X1_train = np.reshape(X1_train, [N, resolution])
    X1_test = np.reshape(X1_test, [N_test, resolution])
    X1_valid = np.reshape(X1_valid, [N_val, resolution])
    X2_train = np.reshape(X2_train, [N, resolution])
    X2_test = np.reshape(X2_test, [N_test, resolution])
    X2_valid = np.reshape(X2_valid, [N_val, resolution])
    # dense to one hot, i.e. [i]-->[0,0,...0,1 (ith position),0,..,0]
    # Y_train = dense_to_one_hot(y_train, n_classes=2)
    # Y_valid = dense_to_one_hot(y_valid, n_classes=2)
    # Y_test = dense_to_one_hot(y_test, n_classes=2)

    return numIndiv, imsize, X1_train, X1_valid, X1_test, X2_train, X2_valid, X2_test, Y_train, Y_valid, Y_test

def dataHelperPlot(path, num_train, num_test, num_valid):
    Fish = {}
    with h5py.File(path, 'r') as f:
        Fish['data'] = f['images']['data'][()]
        Fish['labels'] = f['images']['labels'][()]
        print 'database loaded'

    # Get data and reversed to generate pairs
    data1 = Fish['data'].astype(np.float32)
    data2 = data1[::-1]
    # print 'data 1 *****************'
    # print data1[0]
    # print 'data 2 *****************'
    # print data2[len(data2)-1]
    # get image dimensions
    imsize = data1[0].shape
    resolution = np.prod(imsize)
    # ditto for the labels
    label = Fish['labels'].astype(np.int32)
    label1 = np.subtract(np.squeeze(label),1)
    label2 = label1[::-1]
    # print label1
    # print "-------------------->8"
    # print label2

    # compute number of individuals (classes)
    numIndiv = len(list(set(label1)))
    # compute match/mismatch label for each pair
    label = np.array(label1 == label2, dtype=np.int32)
    # print "-------------------->8"
    # print label

    print 'splitting data in train, test and validation'
    N = np.round(num_train*numIndiv/2) # of training data
    N_test = np.round(num_test*numIndiv/2) # of test data
    N_val = np.round(num_valid*numIndiv/2) # validation data

    # print N, N_test, N_val

    X1_train = data1[:N]
    X1_test = data1[N:N+N_test]
    X1_valid = data1[N+N_test:N+N_test+N_val]
    X2_train = data2[:N]
    X2_test = data2[N:N+N_test]
    X2_valid = data2[N+N_test:N+N_test+N_val]
    Y_train = label[:N]
    Y_test = label[N:N+N_test]
    Y_valid = label[N+N_test:N+N_test+N_val]
    Y1_train = label1[:N]
    Y1_test = label1[N:N+N_test]
    Y1_valid = label1[N+N_test:N+N_test+N_val]
    Y2_train = label2[:N]
    Y2_test = label2[N:N+N_test]
    Y2_valid = label2[N+N_test:N+N_test+N_val]

    # reshape images
    X1_train = np.reshape(X1_train, [N, resolution])
    X1_test = np.reshape(X1_test, [N_test, resolution])
    X1_valid = np.reshape(X1_valid, [N_val, resolution])
    X2_train = np.reshape(X2_train, [N, resolution])
    X2_test = np.reshape(X2_test, [N_test, resolution])
    X2_valid = np.reshape(X2_valid, [N_val, resolution])
    # dense to one hot, i.e. [i]-->[0,0,...0,1 (ith position),0,..,0]
    # Y_train = dense_to_one_hot(y_train, n_classes=2)
    # Y_valid = dense_to_one_hot(y_valid, n_classes=2)
    # Y_test = dense_to_one_hot(y_test, n_classes=2)

    train_size = np.round(num_train/2)*numIndiv

    return numIndiv, imsize, X1_train, X1_valid, X1_test, X2_train, X2_valid, X2_test, Y_train, Y_valid, Y_test, Y1_train, Y1_valid, Y1_test, Y2_train, Y2_valid, Y2_test, train_size


# path = '../matlabexports/imdb_5indiv_15000_1000_102_rot_25dpf_s1.mat'
# num_train = 20
# num_test = 10
# num_valid = 5
# numIndiv, imsize, X1_train, X1_valid, X1_test, X2_train, X2_valid, X2_test, Y_train, Y_valid, Y_test = dataHelperPlot(path, num_train, num_test, num_valid)

def dataHelperPlot2(path, num_train, num_test, num_valid):
    Fish = {}
    with h5py.File(path, 'r') as f:
        Fish['data'] = f['images']['data'][()] # [numImages, c, h, w]
        Fish['labels'] = f['images']['labels'][()]
        print 'database loaded'

    # Initialize images and labels
    data = Fish['data'].astype(np.float32)
    labels = Fish['labels'].astype(np.int32)
    labels =np.subtract(np.squeeze(labels),1)
    numImages = len(labels)
    imsize = data[0].shape
    resolution = np.prod(imsize)
    # print("numImages " + str(numImages))
    numIndiv = len(list(set(labels)))
    # print("numIndiv " + str(numIndiv))
    # Permutation of the images and labels.
    perm = np.random.permutation(numImages)
    data = data[perm, :, :, :]
    labels = labels[perm]

    # Split data to generate genuine and impostor pairs
    numG = int(np.ceil(numImages*(numIndiv - 1)/numIndiv))
    numI = int(numImages-numG)
    # print("numG " + str(numG))
    # print("numI " + str(numI))

    dataG = data[:numG, :, :, :]
    dataI = data[numG:, :, :, :]
    labelsG = labels[:numG]
    labelsI = labels[numG:]

    # Init
    data1 = []
    data2 = []
    labels1 = []
    labels2 = []

    totalNumG = 0
    totalNumI = 0
    RevDataI = dataI[::-1]
    RevLabelsI = labelsI[::-1]
    # Generate pairs
    for i in range(numIndiv):
        # Generate genuine pairs
        indicesG = labelsG == i
        numCurGPairs = int(np.floor(sum(indicesG)/2))
        # print("Indiv " + str(i) + " numCurGPairs " + str(numCurGPairs))
        totalNumG += numCurGPairs
        curData = dataG[indicesG, :, :, :][:numCurGPairs]
        curRevData = curData[::-1][:numCurGPairs]

        data1.append(curData)
        data2.append(curRevData)
        labels1.append(np.ones(numCurGPairs)*i)
        labels2.append(np.ones(numCurGPairs)*i)

        # Generate impostor pairs
        indices = Ncycle(range(len(dataI)),i)
        # print("indices " + str(indices[:5]))
        # print("Len indices " + str(len(indices)))
        dataIshifted = dataI[indices]
        labelsIshifted = labelsI[indices]
        # print("shape dataIshifted " + str(dataIshifted.shape))
        # print("shape RevDataI " + str(RevDataI.shape))
        numAdPairs = np.round(len(labelsI)/2)

        dataIhalf = dataIshifted[:numAdPairs]
        # print("shape dataIhalf " + str(dataIhalf.shape))
        RevDataIhalf = RevDataI[:numAdPairs]
        # print("shape RevDataIhalf " + str(RevDataIhalf.shape))
        labelsIhalf = labelsIshifted[:numAdPairs]
        RevLabelsIhalf = RevLabelsI[:numAdPairs]

        indicesI = RevLabelsIhalf != labelsIhalf
        # print("Len indicesI " + str(len(indicesI)))
        RevIndicesI = indicesI[::-1]

        numCurIPairs = np.sum(indicesI)
        totalNumI += numCurIPairs
        # print("numCurIPairs " + str(numCurIPairs))
        data1.append(dataIhalf[indicesI])
        data2.append(RevDataIhalf[RevIndicesI])
        labels1.append(labelsIhalf[indicesI])
        labels2.append(RevLabelsIhalf[indicesI])

        # print "---------------------------------------"

    # labelsGI = np.array(labels1 == labels2, dtype=np.int32)
    # totalNumG2 = sum(labelsGI == 1)
    # totalNumI2 = sum(labelsGI != 0)
    # print("totalNumG " + str(totalNumG))
    # print("totalNumI " + str(totalNumI))
    # print("totalNumG2 " + str(totalNumG2))
    # print("totalNumI2 " + str(totalNumI2))

    # print "data1 ******"
    # print data1[:2]
    # print "data2 ******"
    # print data2[:2]
    # print "labels1 ******"
    # print labels1
    # print "labels2 ******"
    # print labels2

    data1 = np.asarray(flatten(flatten(data1)))
    data2 = np.asarray(flatten(flatten(data2)))
    labels1 = np.asarray(flatten(labels1))
    labels2 = np.asarray(flatten(labels2))

    # print "labels1 ******"
    # print labels1
    # print "labels2 ******"
    # print labels2

    labelsGI = np.asarray(labels1 == labels2,'float32')
    # print labelsGI
    # totalNumG2 = sum(labelsGI == 1)
    # totalNumI2 = sum(labelsGI == 0)
    # print("totalNumG " + str(totalNumG))
    # print("totalNumI " + str(totalNumI))
    # print("totalNumG2 " + str(totalNumG2))
    # print("totalNumI2 " + str(totalNumI2))

    # print "data1 ******"
    # print data1[:2]
    # print "data2 ******"
    # print data2[:2]
    # print "labels1 ******"
    # print labels1
    # print "labels2 ******"
    # print labels2
    # print "labelsGI ******"
    # print labelsGI


    # # Second permutation
    perm = np.random.permutation(len(labelsGI))
    data1 = data1[perm]
    data2 = data2[perm]
    labels1 = labels1[perm]
    labels2 = labels2[perm]
    labelsGI = labelsGI[perm]

    print 'splitting data in train, test and validation'
    N = num_train*numIndiv # of training data
    N_test = num_test*numIndiv # of test data
    N_val = num_valid*numIndiv # validation data

    # print N, N_test, N_val

    X1_train = data1[:N]
    X1_test = data1[N:N+N_test]
    X1_valid = data1[N+N_test:N+N_test+N_val]
    X2_train = data2[:N]
    X2_test = data2[N:N+N_test]
    X2_valid = data2[N+N_test:N+N_test+N_val]
    Y_train = labelsGI[:N]
    Y_test = labelsGI[N:N+N_test]
    Y_valid = labelsGI[N+N_test:N+N_test+N_val]
    Y1_train = labels1[:N]
    Y1_test = labels1[N:N+N_test]
    Y1_valid = labels1[N+N_test:N+N_test+N_val]
    Y2_train = labels2[:N]
    Y2_test = labels2[N:N+N_test]
    Y2_valid = labels2[N+N_test:N+N_test+N_val]

    # reshape images
    X1_train = np.reshape(X1_train, [N, resolution])
    X1_test = np.reshape(X1_test, [N_test, resolution])
    X1_valid = np.reshape(X1_valid, [N_val, resolution])
    X2_train = np.reshape(X2_train, [N, resolution])
    X2_test = np.reshape(X2_test, [N_test, resolution])
    X2_valid = np.reshape(X2_valid, [N_val, resolution])
    # dense to one hot, i.e. [i]-->[0,0,...0,1 (ith position),0,..,0]
    # Y_train = dense_to_one_hot(y_train, n_classes=2)
    # Y_valid = dense_to_one_hot(y_valid, n_classes=2)
    # Y_test = dense_to_one_hot(y_test, n_classes=2)

    train_size = len(Y_train)

    return numIndiv, imsize, X1_train, X1_valid, X1_test, X2_train, X2_valid, X2_test, Y_train, Y_valid, Y_test, Y1_train, Y1_valid, Y1_test, Y2_train, Y2_valid, Y2_test, train_size, totalNumG, totalNumI

# path = '../matlabexports/imdb_5indiv_15000_1000_32_rotateAndCrop_25dpf_s1.mat'
# num_train = 20
# num_test = 10
# num_valid = 5
# numIndiv, imsize, X1_train, X1_valid, X1_test, X2_train, X2_valid, X2_test, Y_train, Y_valid, Y_test, Y1_train, Y1_valid, Y1_test, Y2_train, Y2_valid, Y2_test, train_size, totalNumG, totalNumI = dataHelperPlot2(path, num_train, num_test, num_valid)

def dataHelperPlot3(path, num_train, num_test, num_valid, G_ratio):
    """

    This function generates a dataset for the training of a siamese network,
    in which genuine and impostor pairs (defined following [1]) are genereated
    according to the ratio specified by the user. The main idea consists in
    generating pairs by folding an array of indices and taking its first half
    to minimize repetitions in the images passed to the network as pairs.

    [1] Chopra, Sumit, Raia Hadsell, and Yann LeCun. "Learning a similarity
    metric discriminatively, with application to face verification."
    2005 IEEE Computer Society Conference on Computer Vision and Pattern
    Recognition (CVPR'05). Vol. 1. IEEE, 2005.


    """

    # load a Matlab dataset in which data are divided in images and labels
    Fish = {}
    with h5py.File(path, 'r') as f:
        Fish['data'] = f['images']['data'][()] # [numImages, c, h, w]
        Fish['labels'] = f['images']['labels'][()]
        print 'database loaded'

    # Initialize images and labels
    # cast images to float32 and labels to int32
    # data is a four dimensional tensor [numOfImages, channel, width, height]
    data = Fish['data'].astype(np.float32)
    labels = Fish['labels'].astype(np.int32)
    # being a matlab dataset we retrieve labels from 0 to n-1 by subtracting 1.
    # np.squeeze assure to deal with an array of labels
    labels =np.subtract(np.squeeze(labels),1)
    imsize = data[0].shape
    resolution = np.prod(imsize)
    numIndiv = len(list(set(labels)))
    numImages = len(labels) # number of images available
    numPairsRequiredPerIndiv = num_train + num_valid + num_test #according to inputs

    print "-----------------------------"
    print("numIndiv " + str(numIndiv))
    print("numImages " + str(numImages))
    print("numPairsRequiredPerIndiv " + str(numPairsRequiredPerIndiv))

    # A first permutation of the entire dataset (images and labels) will allow
    # us to generate genuine and impostor pairs.
    perm = np.random.permutation(numImages)
    data = data[perm, :, :, :]
    labels = labels[perm]

    # Compute number of pairs available in one shift per individual
    numGPerShiftPerIndiv = numImages/(numIndiv*2)
    numIPershiftPerIndiv = numImages*(numIndiv-1)/(numIndiv*2)

    print "-----------------------------"
    print("numGPerShiftPerIndiv " + str(numGPerShiftPerIndiv))
    print("numIPershiftPerIndiv " + str(numIPershiftPerIndiv))

    # Compute number of pairs needed per individual as a function of the number
    # of images required by the user and the ratio |G|/|I|.
    numGRequiredPerIndiv = np.ceil(numPairsRequiredPerIndiv*G_ratio)
    numIRequiredPerIndiv = np.ceil(numPairsRequiredPerIndiv*(1-G_ratio))

    print "-----------------------------"
    print("numGRequiredPerIndiv " + str(numGRequiredPerIndiv))
    print("numIRequiredPerIndiv " + str(numIRequiredPerIndiv))

    # Folding the array of indices and taking the first half of pairs generated
    # by the folding is not always enough. Hence we generate more pairs
    # (if needed) by permuting one of the two halves, we call this operation shift.
    # Here we compute how many shifts are needed.
    numShiftG = np.ceil(numGRequiredPerIndiv/numGPerShiftPerIndiv) # Per individual
    numShiftI = np.ceil(numIRequiredPerIndiv/numIPershiftPerIndiv) # Among individuals

    print "-----------------------------"
    print("numShiftG " + str(numShiftG))
    print("numShiftI " + str(numShiftI))

    if numShiftG > 1:
        print('WARNING: the number of genuine pairs required is bigger than the number of genuine pairs available to feed the CNN with unique images.')
    # if numShiftI > 1:
    #     print('WARNING: the number of impostor pairs required is bigger than the number of impostor pairs available to feed the CNN with unique images.')

    # Initialize variables
    dataG1 = []
    dataG2 = []
    labelsG1 = []
    labelsG2 = []
    dataI1 = []
    dataI2 = []
    labelsI1 = []
    labelsI2 = []
    totalNumG = 0
    totalNumI = 0

    # Generate genuine pairs for each individual
    for i in range(numIndiv):
        # check for images associated to a certain individual
        indicesG = labels == i
        # number of available genuine pairs for individual i
        curNumGPairs = sum(indicesG)/2
        print("curNumGPairs " + str(curNumGPairs))
        curData = data[indicesG][:curNumGPairs]
        # reverse the images
        curRevData = curData[::-1][:curNumGPairs]
        # apply shift operation (if necessary)
        for shift in range(int(numShiftG)):
            print("shift " + str(shift))
            indices = Ncycle(range(curNumGPairs),shift-1) # list from 0 to sum(indicesG)/2
            dataG1.append(curData[indices])
            dataG2.append(curRevData)
            labelsG1.append(np.ones(curNumGPairs)*i)
            labelsG2.append(np.ones(curNumGPairs)*i)
            # compute the number of genuine pairs generated with shift
            totalNumG += curNumGPairs
    # flatten the arrays generated by appending genuine pairs in the loops above
    dataG1 = np.asarray(flatten(dataG1))
    dataG2 = np.asarray(flatten(dataG2))
    labelsG1 = np.asarray(flatten(labelsG1))
    labelsG2 = np.asarray(flatten(labelsG2))

    print("shape dataG1 " + str(dataG1.shape))
    print("shape dataG2 " + str(dataG2.shape))

    # slice the arrays in order to consider the number of pairs required in input
    numRequiredG = int(numGRequiredPerIndiv*numIndiv)
    # and permute them
    perm = np.random.permutation(len(labelsG1))
    dataG1 = dataG1[perm][:numRequiredG]
    dataG2 = dataG2[perm][:numRequiredG]
    labelsG1 = labelsG1[perm][:numRequiredG]
    labelsG2 = labelsG2[perm][:numRequiredG]

    # Generage impostor pairs
    numAdIPairs = numImages*(numIndiv-1)/(numIndiv*2) #number of admisible impostor pairs (minimal repetition)
    print "-----------------------------"
    print("numAdIPairs " + str(numAdIPairs))
    # generate both original and reversed images (symmetrically labels)
    dataHalf = data[:numAdIPairs]
    revDataHalf = data[::-1][:numAdIPairs]
    labelsHalf = labels[:numAdIPairs]
    revLabelHalf = labels[::-1][:numAdIPairs]
    # gather the indices corresponding to impostor pairs
    indicesI = revLabelHalf != labelsHalf
    # generate data according to these indices
    dataHalf = dataHalf[indicesI]
    revDataHalf = revDataHalf[indicesI]
    labelsHalf = labelsHalf[indicesI]
    revLabelHalf = revLabelHalf[indicesI]
    # number of impostor pairs generated so far
    numIPairs = len(labelsHalf)
    print "-----------------------------"
    print("numIPairs " + str(numIPairs))
    # if it is necessary use the shift permutation to generate more impostors,
    # the following loop is symmetrical with respect to the genuine's.
    for shift in range(int(numShiftI)):
        indices = Ncycle(range(numIPairs),shift-1) # list from 0 to sum(indicesG)/2

        dataI1.append(dataHalf[indices])
        dataI2.append(revDataHalf)
        labelsI1.append(labelsHalf[indices])
        labelsI2.append(revLabelHalf)

        totalNumI += numIPairs

    dataI1 = np.asarray(flatten(dataI1))
    dataI2 = np.asarray(flatten(dataI2))
    labelsI1 = np.asarray(flatten(labelsI1))
    labelsI2 = np.asarray(flatten(labelsI2))

    print("shape dataG1 " + str(dataI1.shape))
    print("shape dataG2 " + str(dataI2.shape))

    perm = np.random.permutation(len(labelsI1))
    numRequiredI = int(numIRequiredPerIndiv*numIndiv)
    dataI1 = dataI1[perm][:numRequiredI]
    dataI2 = dataI2[perm][:numRequiredI]
    labelsI1 = labelsI1[perm][:numRequiredI]
    labelsI2 = labelsI2[perm][:numRequiredI]

    # print "---------------------------------------"

    # labelsGI = np.array(labels1 == labels2, dtype=np.int32)
    # totalNumG2 = sum(labelsGI == 1)
    # totalNumI2 = sum(labelsGI != 0)
    # print("totalNumG " + str(totalNumG))
    # print("totalNumI " + str(totalNumI))
    # print("totalNumG2 " + str(totalNumG2))
    # print("totalNumI2 " + str(totalNumI2))

    # print "data1 ******"
    # print data1[:2]
    # print "data2 ******"
    # print data2[:2]
    # print "labels1 ******"
    # print labels1
    # print "labels2 ******"
    # print labels2

    data1 = [dataG1,dataI1]
    data2 = [dataG2,dataI2]
    labels1 = [labelsG1,labelsI1]
    labels2 = [labelsG2,labelsI2]
    # print "========================"
    # print "data1" + str(data1[0])
    # print "========================"
    data1 = np.asarray(flatten(flatten(data1)))
    data2 = np.asarray(flatten(flatten(data2)))
    labels1 = np.asarray(flatten(labels1))
    labels2 = np.asarray(flatten(labels2))
    # print "========================"
    # print "data1" + str(data1[0])
    # print "========================"
    print("shape data1 " + str(data1.shape))
    print("shape data2 " + str(data2.shape))

    # dataI1 = np.asarray(data1)
    # dataI2 = np.asarray(data2)
    # labels1 = np.asarray(labels1)
    # labels2 = np.asarray(labels2)

    # print "labels1 ******"
    # print labels1
    # print "labels2 ******"
    # print labels2

    # get the labels of genuine (loabelled as 1) and impostor (0) pairs
    labelsGI = np.asarray(labels1 == labels2,'float32')
    print labelsGI
    totalNumG2 = sum(labelsGI == 1)
    totalNumI2 = sum(labelsGI == 0)
    print("totalNumG2 " + str(totalNumG2))
    print("totalNumI2 " + str(totalNumI2))

    # print "data1 ******"
    # print data1[:2]
    # print "data2 ******"
    # print data2[:2]
    # print "labels1 ******"
    # print labels1
    # print "labels2 ******"
    # print labels2
    # print "labelsGI ******"
    # print labelsGI

    # Second permutation
    perm = np.random.permutation(len(labelsGI))
    data1 = data1[perm]
    data2 = data2[perm]
    labels1 = labels1[perm]
    labels2 = labels2[perm]
    labelsGI = labelsGI[perm]

    print 'splitting data in train, test and validation'
    N = num_train*numIndiv # of training data
    N_test = num_test*numIndiv # of test data
    N_val = num_valid*numIndiv # validation data

    # print N, N_test, N_val

    X1_train = data1[:N]
    X1_test = data1[N:N+N_test]
    X1_valid = data1[N+N_test:N+N_test+N_val]
    X2_train = data2[:N]
    X2_test = data2[N:N+N_test]
    X2_valid = data2[N+N_test:N+N_test+N_val]
    Y_train = labelsGI[:N]
    Y_test = labelsGI[N:N+N_test]
    Y_valid = labelsGI[N+N_test:N+N_test+N_val]
    Y1_train = labels1[:N]
    Y1_test = labels1[N:N+N_test]
    Y1_valid = labels1[N+N_test:N+N_test+N_val]
    Y2_train = labels2[:N]
    Y2_test = labels2[N:N+N_test]
    Y2_valid = labels2[N+N_test:N+N_test+N_val]

    print '-------------'
    print X1_train.shape
    print N
    print resolution
    print '-------------'
    # reshape images
    X1_train = np.reshape(X1_train, [N, resolution])
    X1_test = np.reshape(X1_test, [N_test, resolution])
    X1_valid = np.reshape(X1_valid, [N_val, resolution])
    X2_train = np.reshape(X2_train, [N, resolution])
    X2_test = np.reshape(X2_test, [N_test, resolution])
    X2_valid = np.reshape(X2_valid, [N_val, resolution])

    numGenuine = sum(Y_train == 1)
    numImpostor = sum(Y_train == 0)
    print("numGenuine " + str(numGenuine))
    print("numImpostor " + str(numImpostor))

    train_size = len(Y_train)

    return numIndiv, imsize, X1_train, X1_valid, X1_test, X2_train, X2_valid, X2_test, Y_train, Y_valid, Y_test, Y1_train, Y1_valid, Y1_test, Y2_train, Y2_valid, Y2_test, train_size, totalNumG, totalNumI

# path = '../matlabexports/imdb_5indiv_15000_1000_32_rotateAndCrop_25dpf_s1.mat'
# num_train = 40
# num_test = 10
# num_valid = 5
# G_ratio = .5
# numIndiv, imsize, X1_train, X1_valid, X1_test, X2_train, X2_valid, X2_test, Y_train, Y_valid, Y_test, Y1_train, Y1_valid, Y1_test, Y2_train, Y2_valid, Y2_test, train_size, totalNumG, totalNumI = dataHelperPlot3(path, num_train, num_test, num_valid, G_ratio)
