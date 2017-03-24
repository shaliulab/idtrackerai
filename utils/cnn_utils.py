# Import standard libraries
import os

# Import third party libraries
import numpy as np
import numpy.matlib as npm
import tensorflow as tf
import itertools

# Import application/library specifics
from tf_utils import *
from py_utils import *

'''
****************************************************************************
Tensorboard
*****************************************************************************
'''

def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % 'tower', '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _add_loss_summary(loss):
    tf.summary.scalar(loss.op.name, loss)

def put_kernels_on_grid(kernel, (grid_Y, grid_X), pad=1):
    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
    '''
    # pad X and Y
    x1 = tf.pad(kernel, tf.constant( [[pad,pad],[pad,pad],[0,0],[0,0]] ))

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + 2*pad
    X = kernel.get_shape()[1] + 2*pad
    NumChannels = kernel.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, NumChannels]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, NumChannels]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.summary.image order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 1]
    x_min = tf.reduce_min(x7)
    x_max = tf.reduce_max(x7)
    x8 = (x7 - x_min) / (x_max - x_min)

    return x8

'''
****************************************************************************
CNN wrappers
*****************************************************************************
'''

def computeVolume(width, height, strides):
    c1 = float(strides[1])
    c2 = float(strides[2])
    widthS = int(np.ceil(width/c1))
    heightS = int(np.ceil(height/c2))
    return widthS, heightS

def buildConv2D(scopeName, inputWidth, inputHeight, inputDepth, inputConv ,filter_size, n_filters, stride, pad):
    w,h = computeVolume(inputWidth, inputHeight, stride)
    with tf.variable_scope(scopeName) as scope:
        W = tf.get_variable(
            'weights',
            [filter_size, filter_size, inputDepth, n_filters],
            # initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1)
            initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0)
            )
        b = tf.get_variable(
            'biases',
            [n_filters],
            # initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1)
            initializer=tf.constant_initializer(0.0)
            )
        conv = tf.nn.conv2d(
                    input=inputConv,
                     filter=W,
                     strides=stride,
                     padding=pad)
        convb = tf.nn.bias_add(conv, b, name = scope.name)

        _activation_summary(convb)

        with tf.variable_scope('visualization') :

            grid_x = int(np.sqrt(n_filters))
            grid_y = grid_x  # to get a square grid for 64 conv1 features
            WtoPlot = tf.slice(W, [0, 0, 0, 0], [filter_size, filter_size, 1, n_filters])
            grid = put_kernels_on_grid(WtoPlot, (grid_y, grid_x))
            tf.summary.image(scopeName + '/features', grid, max_outputs=1)

            # x_min = tf.reduce_min(convb)
            # x_max = tf.reduce_max(convb)
            # convb_0_to_1 = (convb - x_min) / (x_max - x_min)

            convbToPlot = tf.slice(convb, [0, 0, 0, 0], [-1, w, h, 1])

            # this will display random images
            tf.summary.image(scopeName + '/output', convbToPlot, max_outputs=10)


    return convb,w,h,grid


def maxpool2d(name,inputWidth, inputHeight, inputPool, pool=2 , stride=[1,2,2,1] ,pad='VALID'):
    # MaxPool2D wrapper
    max_pool = tf.nn.max_pool(inputPool,
        ksize=[1, pool, pool, 1],
        strides=stride,
        padding=pad,
        name = name
        )
    w, h = computeVolume(inputWidth, inputHeight, stride)
    return max_pool, w, h

def buildFc(scopeName, inputFc, height, width, n_filters, n_fc, keep_prob):
    with tf.variable_scope(scopeName) as scope:
        W = tf.get_variable(
            'weights',
            [height * width * n_filters, n_fc],
            # initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1)
            initializer=tf.contrib.layers.xavier_initializer(seed=0)
            )
        b = tf.get_variable(
            'biases',
            [n_fc],
            # initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1)
            initializer=tf.constant_initializer(0.0)
            )
        fc = tf.add(tf.matmul(inputFc, W), b)
        fc_drop = tf.nn.dropout(fc, keep_prob, name = scope.name)
        _activation_summary(fc_drop)

    return fc_drop

def reLU(scopeName, inputRelu):
    with tf.variable_scope(scopeName) as scope:
        relu = tf.nn.relu(inputRelu, name = scope.name)
        _activation_summary(relu)
    return relu

def buildSoftMax(scopeName, inputSoftMax, n_fc, classes):
    with tf.variable_scope(scopeName) as scope:
        W = tf.get_variable(
            'weights',
            [n_fc, classes],
            initializer=tf.contrib.layers.xavier_initializer(seed=0)
            )
        b = tf.get_variable(
            'biases',
            [classes],
            # initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01)
            initializer=tf.constant_initializer(0.0)
            )
        logits = tf.add(tf.matmul(inputSoftMax, W), b, name = scope.name)
        _activation_summary(logits)
    return logits, [W,b]

# def buildSoftMaxWeights(inputSoftMax,n_fc,classes):
#     # the same as build softmax, but outputs the weights for visualization
#     W_fc = weight_variable([n_fc, classes])
#     b_fc = bias_variable([classes])
#     y_logits = tf.matmul(inputSoftMax, W_fc) + b_fc
#     return y_logits, W_fc, b_fc

'''
****************************************************************************
Manage checkpoit folders, saving and restore
*****************************************************************************
'''

def createSaver(varName, include, name):
    '''
    varName: string (name of part of the name of the variable)
    include: boolean (save or discard)

    Scans all the variables and save the ones that has 'varName' as part of their
    name, if include is True or the contrary if include is False
    '''
    if include:
        saver = tf.train.Saver([v for v in tf.global_variables() if varName in v.name], name = name)
    elif not include:
        saver = tf.train.Saver([v for v in tf.global_variables() if varName not in v.name], name = name)
    else:
        raise ValueError('The second argument has to be a boolean')

    return saver


def createCkptFolder(folderName, subfoldersNameList):
    '''
    create if it does not exist the folder folderName in CNN and
    the same for the subfolders in the subfoldersNameList
    '''
    if not os.path.exists(folderName): # folder does not exist
        os.makedirs(folderName) # create a folder
        print folderName + ' has been created'
    else:
        print folderName + ' already exists'

    subPaths = []
    for name in subfoldersNameList:
        subPath = folderName + '/' + name
        if not os.path.exists(subPath):
            os.makedirs(subPath)
            print subPath + ' has been created'
        else:
            print subPath + ' already exists'
        subPaths.append(subPath)
    return subPaths

def restoreFromFolder(pathToCkpt, saver, session):
    '''
    Restores variables stored in pathToCkpt with a certain saver,
    for a particular (TF) session
    '''
    ckpt = tf.train.get_checkpoint_state(pathToCkpt)
    print "************************************************************"
    print pathToCkpt
    print ckpt
    print "************************************************************"
    if ckpt and ckpt.model_checkpoint_path:
        print "restoring from " + ckpt.model_checkpoint_path
        saver.restore(session, ckpt.model_checkpoint_path) # restore model variables

''' ****************************************************************************
CNN statistics and cluster analysis
*****************************************************************************'''

def computeROCAccuracy(y, y_logits, name=None):
    # sess = tf.Session()
    y_logits = tf.argmax(y_logits,1)
    TP = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y[:,1],True),tf.equal(y_logits,True)), tf.float32), 0)
    FP = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y[:,1],False),tf.equal(y_logits,True)), tf.float32), 0)
    FN = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y[:,1],True),tf.equal(y_logits,False)), tf.float32), 0)
    TN = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y[:,1],False),tf.equal(y_logits,False)), tf.float32), 0)

    # print sess.run(TP)
    # print sess.run(FP)
    # print sess.run(FN)
    # print sess.run(TN)

    sensitivity = tf.div(TP,tf.add(TP,FN))
    specificity = tf.div(TN,tf.add(TN,FP))

    # print sess.run(sensitivity)
    # print sess.run(specificity)

    return sensitivity, specificity, TP, FP, FN, TN

def computeDerivatives(array):

    if len(array) > 1:
        speed = np.diff(array)
    else:
        # warnings.warn('Not enough points to compute the speed')
        speed = []

    if len(array) > 2:
        accel = np.diff(speed)
    else:
        # warnings.warn('Not enough points to compute the acceleration')
        accel = []

    return speed, accel

def computeRefAccuracy(Rfeat,Y_ref,Vfeat,Y_val):
    Y_ref = one_hot_to_dense(Y_ref)
    Y_val = one_hot_to_dense(Y_val)
    numIndiv = len(list(set(Y_ref)))
    # print 'numIndiv', str(numIndiv)
    numRef = len(Y_ref)
    # print 'numRef', str(numRef)
    numRefPerIndiv = numRef/numIndiv
    # print 'numRefPerIndiv', str(numRefPerIndiv)
    numProbImages = len(Y_val)
    idP = []
    for i, (Y_P, featP) in enumerate(zip(Y_val,Vfeat)):
        # print 'Y_P', str(Y_P)
        # print 'featP', str(featP)
        featP_rep = npm.repmat(featP,numRef,1)
        # print 'featP_rep', str(featP_rep)
        dist = np.sum(np.abs(np.subtract(featP_rep,Rfeat)),1)
        # print 'dist', str(dist)
        bestRef = np.where(dist==np.min(dist))
        # print 'bestRef', str(bestRef)
        ids = Y_ref[bestRef]
        # print ids

        if len(list(set(ids))) == 1:
            idP.append(ids[0])
        else:
            idP.append(np.float('nan'))

    overallAcc = np.true_divide(np.sum(np.subtract(Y_val,idP) == 0),numProbImages)
    indivAcc = [np.true_divide(np.sum(np.logical_and(np.equal(idP, i), np.equal(Y_val, i)), axis=0), np.sum(np.equal(Y_val, i))) for i in range(numIndiv)]

    return idP, overallAcc, indivAcc

# Rfeat = np.asarray([[1.,2.,3.1],[1.,2.,3.],[1.,2.,3.],[2.,3.,4.1],[3.,4.,5.],[3.,4.,5.1]])
# Y_ref = np.asarray([0,0,1,1,2,2])
# Vfeat = np.asarray([[1.,2.,3.],[1.,2.,3.1],[1.,2.,3.1],[3.,4.,5.],[2.,3.,4.1],[3.,4.,5.1]])
# Y_val = np.asarray([0,1,0,2,1,2])
# computeRefAccuracy(Rfeat,Y_ref,Vfeat,Y_val)

# y = np.array([[1,0],[0,1],[1,0],[0,1]])
# y_logits = np.array([[1,0],[0,1],[1,0],[1,0]])
# computeROCAccuracy(y, y_logits, name=None)

def individualAccuracy(labels,logits,classes):
    # We add 1 to the labels and predictions to avoid having a 0 label
    labels = tf.cast(tf.add(tf.where(tf.equal(labels,1))[:,1],1),tf.float32)
    predictions = tf.cast(tf.add(tf.argmax(logits,1),1),tf.float32)
    labelsRep = tf.reshape(tf.tile(labels, [classes]), [classes,tf.shape(labels)[0]])

    correct = tf.cast(tf.equal(labels,predictions),tf.float32)
    indivCorrect = tf.multiply(predictions,correct)

    indivRep = tf.cast(tf.transpose(tf.reshape(tf.tile(tf.range(1,classes+1), [tf.shape(labels)[0]]), [tf.shape(labels)[0],classes])),tf.float32)
    indivCorrectRep = tf.reshape(tf.tile(indivCorrect, [classes]), [classes,tf.shape(labels)[0]])
    correctPerIndiv = tf.cast(tf.equal(indivRep,indivCorrectRep),tf.float32)

    countCorrect = tf.reduce_sum(correctPerIndiv,1)
    numImagesPerIndiv = tf.reduce_sum(tf.cast(tf.equal(labelsRep,indivRep),tf.float32),1)

    indivAcc = tf.div(countCorrect,numImagesPerIndiv)

    correct_prediction = tf.equal(predictions, labels, name='correctPrediction')
    acc = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name='overallAccuracy')
    # acc = tf.reduce_mean(indivAcc)

    return acc,indivAcc

# labels = tf.constant([[1,0,0],[0,1,0],[1,0,0],[0,0,1]])
# logits = tf.constant([[1,0,0],[0,1,0],[1,0,0],[0,1,0]])
# individualAccuracy(labels,logits,tf.constant(3))

''' ****************************************************************************
Data Augmentation and image processing
*****************************************************************************'''
def getCorrelatedImages(images,labels,numImages, minNumImages):
    '''
    This functions assumes that images and labels have not been permuted
    and they are temporarly ordered for each animals
    :images: all images of a particular list of individuals ordered by individuals
    :labels: all labels of a particular list of individuals ordered by individuals
    :numImages: number of images for training for each individual
    '''
    imagesTrain = []
    labelsTrain = []
    imagesVal = []
    labelsVal = []
    imagesTest = []
    labelsTest = []

    numImagesVal = int(numImages * 0.1)
    # Select randomly the frame where the fragment of correlated images starts
    print 'minNumImages, ', minNumImages
    framePos = np.random.randint(0,minNumImages - (numImages + numImagesVal))
    print 'the fragment will start at frame, ', framePos
    for i in np.unique(labels):
        print 'individual, ', i
        # Get images of this individual
        thisIndivImages = images[labels==i]
        thisIndivLabels = labels[labels==i]
        print 'num images of this individual, ', thisIndivImages.shape[0]

        # Get train and validation images and labels
        # first we select a set of correlated images
        imTrainVal = thisIndivImages[framePos:framePos+numImages+numImagesVal]
        labTrainVal = thisIndivLabels[framePos:framePos+numImages+numImagesVal]
        print 'num images for train and val for this indiv, ', imTrainVal.shape[0]
        # we permute the iamges
        imTrainVal = imTrainVal[np.random.permutation(len(imTrainVal))]
        # we select images for training and validation from the permuted images
        imagesTrain.append(imTrainVal[:numImages])
        labelsTrain.append(labTrainVal[:numImages])
        imagesVal.append(imTrainVal[numImages:])
        labelsVal.append(labTrainVal[numImages:])
        print 'num images for train, ', imagesTrain[i].shape[0]
        print 'num images for val, ', imagesVal[i].shape[0]

        # Get test images and labels
        # all the rest of images are the test images
        imTest  = flatten([thisIndivImages[:framePos], thisIndivImages[framePos+numImages+numImagesVal:]])
        imTest = np.asarray(imTest)
        labTest = flatten([thisIndivLabels[:framePos], thisIndivLabels[framePos+numImages+numImagesVal:]])
        labTest = np.asarray(labTest)
        imagesTest.append(imTest) #before the fragment
        labelsTest.append(labTest)
        print 'num images for test, ', imagesTest[i].shape[0]

    # we flatten the arrays
    imagesTrain = flatten(imagesTrain)
    imagesTrain = np.asarray(imagesTrain)
    labelsTrain = flatten(labelsTrain)
    labelsTrain = np.asarray(labelsTrain)
    perm = np.random.permutation(len(labelsTrain))
    imagesTrain = imagesTrain[perm]
    labelsTrain = labelsTrain[perm]

    imagesVal = flatten(imagesVal)
    imagesVal = np.asarray(imagesVal)
    labelsVal = flatten(labelsVal)
    labelsVal = np.asarray(labelsVal)

    imagesTest = flatten(imagesTest)
    imagesTest = np.asarray(imagesTest)
    labelsTest = flatten(labelsTest)
    labelsTest = np.asarray(labelsTest)

    return imagesTrain, labelsTrain, imagesVal, labelsVal, imagesTest, labelsTest, framePos

def cropImages(images,imageSize,shift=(0,0)):
    """ Given batch of images it crops thme in a shape (imageSize,imageSize)
    with a shift in the rows and columns given by the variable shifts. The
    size of the portait must be bigger than
    :param images: batch of images of the shape (numImages, channels, width, height)
    :param imageSize: size of the new portrait, usually 32, since the network accepts images of 32x32  pixels
    :param shift: (x,y) displacement when cropping, it can only go from -maxShift to +maxShift
    :return
    """
    currentSize = images.shape[2]
    if currentSize < imageSize:
        raise ValueError('The size of the input portrait must be bigger than imageSize')
    elif currentSize == imageSize:
        return images
    elif currentSize > imageSize:
        maxShift = np.divide(currentSize - imageSize,2)
        if np.max(shift) > maxShift:
            raise ValueError('The shift when cropping the portrait cannot be bigger than (currentSize - imageSize)/2')
        croppedImages = images[:,maxShift+shift[1]:currentSize-maxShift+shift[1],maxShift+shift[0]:currentSize-maxShift+shift[0],:]
        # print 'Portrait cropped'
        return croppedImages

def dataAugment(images,labels,flag = False):

    def getPossibleShifts():
        possibleShifts = []
        possibleShifts.append(list(itertools.combinations_with_replacement(range(-2,3),2)))
        possibleShifts.append(list(itertools.permutations(range(-2,3),2)))
        possibleShifts.append(list(itertools.combinations(range(-2,3),2)))
        possibleShifts = [shift for l in possibleShifts for shift in l]
        possibleShifts = set(possibleShifts)
        return possibleShifts

    if flag:
        print 'Performing data augmentation...'
        possibleShifts = getPossibleShifts() #(0,0) is included
        augmentedImages = []
        augmentedLabels = []
        for shift in possibleShifts:
            newImages = cropImages(images,32,shift=shift)
            augmentedImages.append(newImages)
            augmentedLabels.append(labels)
        images = np.vstack(augmentedImages)
        labels = flatten(augmentedLabels)
    else:
        print 'No data augmentation...'
        print 'Cropping images to 32x32...'
        images = cropImages(images,32,(0,0))

    return images, np.asarray(labels)
