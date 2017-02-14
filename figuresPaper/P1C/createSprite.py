from __future__ import division
import numpy as np
np.random.seed(0)
import pandas as pd
import sys
sys.path.append('../utils')
sys.path.append('../CNN')

import matplotlib.pyplot as plt
import cPickle as pickle
import cv2

from input_data_cnn import *
from cnn_model_summaries import *
from cnn_architectures import *
from cnn_utils import *

def sliceDatabaseForSprite(images, labels, indicesIndiv):
    ''' Select images and labels relative to a subset of individuals'''
    print 'Slicing database...'
    images = np.array([images[labels==ind] for ind in indicesIndiv])
    labels = np.array([i*np.ones(sum(labels==ind)).astype(int) for i,ind in enumerate(indicesIndiv)])
    return images, labels


def getImagesAndLabels(imdbTrain, indIndices):
    '''
    Gets images and labels from library. Returns them ordered according to their
    identities. The number of individuals to consider is set by the input parameter
    numIndivToRepresent.
    '''

    databaseTrainInfo, imagesTrain, labelsTrain, imsizeTrain, numIndivImdbTrain, numImagesPerIndivTrain = loadIMDB(imdbTrain)
    print 'num Images per indiv, ', numImagesPerIndivTrain
    print 'number of Images', len(labelsTrain)

    imagesTrainS, labelsTrainS = sliceDatabaseForSprite(imagesTrain, labelsTrain, indIndices)
    print 'images dimension after slicing ', imagesTrainS.shape
    print 'labels dimension after slicing ', labelsTrainS.shape

    return imagesTrainS, labelsTrainS

def prepareTSNEImages(images, labels, numImagesPerIndiv):
    croppedImages = []
    croppedLabels = []
    print 'images per indiv, while preparing TSNE', numImagesPerIndiv
    for i, indImages in enumerate(images):
        perm = np.random.permutation(len(indImages))
        indImages = indImages[perm]
        indLabels = labels[i]
        indLabels = indLabels[perm]

        croppedImages.append(indImages[:numImagesPerIndiv])
        croppedLabels.append(indLabels[:numImagesPerIndiv])

    images = np.asarray(flatten(croppedImages))
    labels = np.asarray(flatten(croppedLabels))

    perm2 = np.random.permutation(len(labels))
    images = images[perm2]
    labels = labels[perm2]

    return images, labels

def spritify(spriteWidth = 8192, spriteHeight = 8192, numIndivToRepresent = 8):
    """
    Generates sprite image and associated labels
    """
    imdbTrain = selectFile()
    # imdbTrain = ''.join(imdbTrain.split('_')[:-1])
    imdbTrain = imdbTrain.split('/')[-1]
    imdbTrain = imdbTrain[:-7]

    indIndices = np.random.permutation(60)
    indIndices = indIndices[:numIndivToRepresent]
    images, labels = getImagesAndLabels(imdbTrain, indIndices)
    imH, imW = images[0][0].shape
    numColumns = spriteWidth / imW
    numRows = spriteHeight / imH
    numImages = int(numColumns * numRows)
    numImagesPerIndiv = int(np.floor(numImages / numIndivToRepresent))

    images, labels = prepareTSNEImages(images, labels,numImagesPerIndiv)
    linearImages = np.reshape(images, [numImages, imH*imW])

    imagesTSNE = []

    for ind in range(numIndivToRepresent):
        imagesTSNE.append(images[ind][:numImagesPerIndiv])

    rowSprite = []
    sprite = []
    i = 0

    while i < numImages:
        rowSprite.append(images[i])

        if (i+1) % numColumns == 0:
            sprite.append(np.hstack(rowSprite))
            rowSprite = []
        i += 1

    sprite = np.vstack(sprite)
    spriteName = str(numIndivToRepresent) + '_fish_'+ str(numImages)+'imgs_sprite.png'
    cv2.imwrite(spriteName, uint8caster(sprite))

    imageName = str(numIndivToRepresent) + '_fish_'+ str(numImages)+'images.pkl'
    pickle.dump(linearImages, open(imageName, "wb"))

    labelName = str(numIndivToRepresent) + '_fish_'+ str(numImages)+'labels.tsv'
    df = pd.DataFrame(labels)
    df.to_csv(labelName, sep='\t')

    return images, labels

def generateFeaturesFromCNN(labels, images, restorePath = '../CNN/ckpt_dir_new3_xavierSGD_maxImages_300epoch_lr01', numIndivToRepresent = 4):

    batch_size = 1000

    numImages = len(images)
    height, width = images[0].shape
    channels = 1
    resolution = width * height
    classes = len(np.unique(labels))
    n_epochs = 100
    X_t = np.reshape(images, [numImages, height*width])
    Y_t = dense_to_one_hot(labels, classes)
    keep_prob = 1.
    print 'the resolution of the images is ', resolution

    Tindices, Titer_per_epoch = get_batch_indices(len(labels),250)
    Vindices, Viter_per_epoch = get_batch_indices(len(labels),batch_size)

    images_pl, labels_pl = placeholder_inputs(batch_size, resolution, classes)
    keep_prob_pl = tf.placeholder(tf.float32, name = 'keep_prob')

    logits, relu,(W1,W3,W5) = inference1(images_pl, width, height, channels, classes, keep_prob_pl)

    cross_entropy = loss(labels_pl,logits)

    train_op, global_step =  optimize(cross_entropy,0.01)

    accuracy, indivAcc = evaluation(labels_pl,logits,classes)

    summary_op = tf.summary.merge_all()

    saver_model = createSaver('soft', False, 'saver_model')
    saver_softmax = createSaver('soft', True, 'saver_softmax')

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        opListTrain = [train_op, cross_entropy, accuracy, indivAcc, relu, W1,W3,W5]
        opListVal = [cross_entropy, accuracy, indivAcc, relu]

        ckpt_dir_model = restorePath + '/model'



        restoreFromFolder(ckpt_dir_model, saver_model, sess)

        opListVal = [cross_entropy, accuracy, indivAcc, relu]


        epoch_counter = 0
        continueFlag = True

        while continueFlag:
            lossEpoch = []
            accEpoch = []
            indivAccEpoch = []
            epochFeat = []
            for iter_i in range(Titer_per_epoch):

                _, batchLoss, batchAcc, indivBatchAcc, batchFeat, WConv1, WConv3, WConv5, feed_dict = run_batch(
                    sess, opListTrain, Tindices, iter_i, Titer_per_epoch,
                    images_pl, labels_pl, keep_prob_pl,
                    X_t, Y_t, keep_prob = keep_prob)

                lossEpoch.append(batchLoss)
                accEpoch.append(batchAcc)
                indivAccEpoch.append(indivBatchAcc)
                epochFeat.append(batchFeat)
                # Print per batch loss and accuracies



            trainFeatLabels = Y_t[Tindices[iter_i]:Tindices[iter_i+1]]

            trainLoss = np.mean(lossEpoch)
            trainAcc = np.mean(accEpoch)
            trainIndivAcc = np.nanmean(indivAccEpoch, axis=0) # nanmean because in minibatches some individuals could not appear...

            print('Train (epoch %d): ' % epoch_counter + \
                " Loss=" + "{:.6f}".format(trainLoss) + \
                ", Accuracy=" + "{:.5f}".format(trainAcc) + \
                ", Individual Accuracy=")
            print(trainIndivAcc)

            epoch_counter += 1
            if np.mean(trainIndivAcc) > 0.99999:
                continueFlag = False

        #
        # features = []
        # for iter_i in range(Viter_per_epoch):
        #
        #     batchLoss, batchAcc, indivBatchAcc, batchFeat, feed_dict = run_batch(
        #         sess, opListVal, Vindices, iter_i, Viter_per_epoch,
        #         images_pl, labels_pl, keep_prob_pl,
        #         X_v, Y_v, keep_prob = keep_prob)
        #
        #     features.append(batchFeat)

        trainFeat = np.asarray(flatten(epochFeat))
        featName = str(numIndivToRepresent) + '_fish_'+ str(numImages)+'features.pkl'
        pickle.dump(trainFeat, open(featName, "wb"))
    return trainFeat

if __name__ == '__main__':
    images, labels = spritify(spriteWidth = 8192/8, spriteHeight = 8192/8, numIndivToRepresent = 4)

    features = generateFeaturesFromCNN(labels, images, numIndivToRepresent = 4)
