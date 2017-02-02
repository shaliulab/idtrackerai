import os
import sys
if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('GtkAgg')

from tf_utils import *
from py_utils import *

import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
import numpy as np
import numpy.matlib as npm
from pprint import *
from tensorflow.python.platform import gfile
import cPickle as pickle
import re
import pyautogui
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
    return logits

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
        saver = tf.train.Saver([v for v in tf.all_variables() if varName in v.name], name = name)
    elif not include:
        saver = tf.train.Saver([v for v in tf.all_variables() if varName not in v.name], name = name)
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
    if ckpt and ckpt.model_checkpoint_path:
        print "restoring from " + ckpt.model_checkpoint_path
        saver.restore(session, ckpt.model_checkpoint_path) # restore model variables

'''
****************************************************************************
Plotting utilities
*****************************************************************************
'''

def get_spaced_colors(n):
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    rgbcolorslist = [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]
    rgbcolorslist = np.true_divide(rgbcolorslist,255)
    hexcolorslist = [matplotlib.colors.rgb2hex(c) for c in rgbcolorslist]
    return hexcolorslist

def get_legend_str(n):

    return [str(i+1) for i in range(n)]




def CNNplotterFast2(lossAccDict,weightsDict,show=False):

    # get variables
    lossPlot, valLossPlot, lossSpeed,valLossSpeed, lossAccel, valLossAccel, \
    accPlot, valAccPlot, accSpeed,valAccSpeed, accAccel, valAccAccel, \
    indivAcc,indivValAcc, \
    features, labels = getVarFromDict(lossAccDict,[
        'loss', 'valLoss', 'lossSpeed', 'valLossSpeed', 'lossAccel', 'valLossAccel',
        'acc', 'valAcc', 'accSpeed', 'valAccSpeed', 'accAccel', 'valAccAccel',
        'indivAcc', 'indivValAcc',
        'features', 'labels'])

    WConv1, WConv3, WConv5  = getVarFromDict(weightsDict,['W1','W3','W5'])

    # 'Weights': [WConv1,WConv3,WConv5,WFc]

    meanIndivAcc = indivAcc[-1]
    meanValIndiviAcc = indivValAcc[-1]
    numIndiv = len(meanIndivAcc)
    features = features[:30]
    features = np.reshape(features, [features.shape[0],int(np.sqrt(features.shape[1])),int(np.sqrt(features.shape[1]))])
    labels = labels[:30]



    # plt.switch_backend('TkAgg')
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())
    w, h = pyautogui.size()
    print w,h
    fig = plt.figure("fine-tuning", figsize=(w/(2*96),h/96))
    plt.clf()

    # loss
    ax1 = fig.add_subplot(241)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
    ax1.set_axis_bgcolor('none')

    ax1.plot(lossPlot,'or-', label='training')
    ax1.plot(valLossPlot, 'ob--', label='validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss function')
    ax1.legend(fancybox=True, framealpha=0.05)
    ax1.set_xlim((0,1000))
    ax1.set_ylim((0,2.))

    # accuracy
    ax2 = fig.add_subplot(242)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.get_xaxis().tick_bottom()
    ax2.get_yaxis().tick_left()
    ax2.set_axis_bgcolor('none')

    ax2.plot(accPlot, 'or-')
    ax2.plot(valAccPlot, 'ob--')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuray')
    ax2.set_xlim((0,1000))
    ax2.set_ylim((0,1))


    # Individual accuracies
    ax3 = fig.add_subplot(2, 2, 2)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.get_xaxis().tick_bottom()
    ax3.get_yaxis().tick_left()
    ax3.set_axis_bgcolor('none')

    individuals = [str(j) for j in range(1,numIndiv+1)]
    ind = np.arange(numIndiv)+1
    # width = 0.25
    width = 0.35
    rects1 = ax3.bar(ind-width, meanIndivAcc, width, color='red', alpha=0.4,label='training')
    rects2 = ax3.bar(ind, meanValIndiviAcc, width, color='blue', alpha=0.4,label='validation')
    ax3.set_ylim((0,1))
    ax3.set_xlim((0,numIndiv+1))
    ax3.set_xlabel('individual')
    ax3.set_ylabel('Individual accuracy')
    # ax3.legend(fancybox=True, framealpha=0.05)

    # W1
    ax4 = fig.add_subplot(2,3,4)
    ax4.imshow(np.squeeze(WConv1),interpolation='none',cmap='gray',vmin=0, vmax=1)
    ax4.set_title('Conv1 filters')
    ax4.xaxis.set_ticklabels([])
    ax4.yaxis.set_ticklabels([])

    # W3
    ax5 = fig.add_subplot(2,3,5)
    ax5.imshow(np.squeeze(WConv3),interpolation='none',cmap='gray',vmin=0, vmax=1)
    ax5.set_title('Conv2 filters')
    ax5.xaxis.set_ticklabels([])
    ax5.yaxis.set_ticklabels([])

    # W5
    ax6 = fig.add_subplot(2,3,6)
    ax6.imshow(np.squeeze(WConv5),interpolation='none',cmap='gray',vmin=0, vmax=1)
    ax6.set_title('Conv3 filters')
    ax6.xaxis.set_ticklabels([])
    ax6.yaxis.set_ticklabels([])

    # plt.subplots_adjust(bottom=0.1, right=.9, left=0.1, top=.9, wspace = 0.25, hspace=0.25)

    plt.draw()
    plt.pause(0.00000001)


def CNNplotterFast(lossAccDict):

    # get variables
    lossPlot, valLossPlot, lossSpeed,valLossSpeed, lossAccel, valLossAccel, \
    accPlot, valAccPlot, accSpeed,valAccSpeed, accAccel, valAccAccel, \
    indivAcc,indivValAcc, \
    features, labels, weights = getVarFromDict(lossAccDict,[
        'loss', 'valLoss', 'lossSpeed', 'valLossSpeed', 'lossAccel', 'valLossAccel',
        'acc', 'valAcc', 'accSpeed', 'valAccSpeed', 'accAccel', 'valAccAccel',
        'indivAcc', 'indivValAcc',
        'features', 'labels','Weights'])

    # 'Weights': [WConv1,WConv3,WConv5,WFc]

    meanIndivAcc = indivAcc[-1]
    meanValIndiviAcc = indivValAcc[-1]
    numIndiv = len(meanIndivAcc)
    features = features[:30]
    features = np.reshape(features, [features.shape[0],int(np.sqrt(features.shape[1])),int(np.sqrt(features.shape[1]))])
    labels = labels[:30]


    plt.close()
    # fig, axes = plt.subplots(nrows=10, ncols=12)
    # fig = plt.figure()
    plt.switch_backend('TkAgg')
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())


    # loss
    ax1 = plt.subplot(261)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
    ax1.set_axis_bgcolor('none')

    ax1.plot(lossPlot,'or-', label='training')
    ax1.plot(valLossPlot, 'ob--', label='validation')
    ax1.set_ylabel('Loss function')
    ax1.legend(fancybox=True, framealpha=0.05)

    ax2 = plt.subplot(262)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.get_xaxis().tick_bottom()
    ax2.get_yaxis().tick_left()
    ax2.set_axis_bgcolor('none')

    ax2.plot(lossSpeed,'ro-',label='training')
    plt.plot(valLossSpeed,'bo--',label='validation')
    ax2.set_ylabel('Loss function speed')

    ax3 = plt.subplot(263)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.get_xaxis().tick_bottom()
    ax3.get_yaxis().tick_left()
    ax3.set_axis_bgcolor('none')

    ax3.plot(lossAccel,'ro-',label='training')
    plt.plot(valLossAccel,'bo--',label='validation')
    ax3.set_ylabel('Loss function accel.')


    # accuracy
    ax4 = plt.subplot(267)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.get_xaxis().tick_bottom()
    ax4.get_yaxis().tick_left()
    ax4.set_axis_bgcolor('none')

    ax4.plot(accPlot, 'or-')
    ax4.plot(valAccPlot, 'ob--')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuray')

    ax5 = plt.subplot(268)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    ax5.get_xaxis().tick_bottom()
    ax5.get_yaxis().tick_left()
    ax5.set_axis_bgcolor('none')

    ax5.plot(accSpeed,'ro-',label='training')
    plt.plot(valAccSpeed,'bo--',label='validation')
    ax5.set_ylabel('Accuray speed')

    ax6 = plt.subplot(2,6,9)
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)
    ax6.get_xaxis().tick_bottom()
    ax6.get_yaxis().tick_left()
    ax6.set_axis_bgcolor('none')

    ax6.plot(accAccel,'ro-',label='training')
    plt.plot(valAccAccel,'bo--',label='validation')
    ax6.set_ylabel('Accuray accel.')

    # Individual accuracies
    ax7 = plt.subplot(1, 4, 3)
    ax7.spines["top"].set_visible(False)
    ax7.spines["right"].set_visible(False)
    ax7.get_xaxis().tick_bottom()
    ax7.get_yaxis().tick_left()
    ax7.set_axis_bgcolor('none')

    individuals = [str(j) for j in range(1,numIndiv+1)]
    ind = np.arange(numIndiv)+1
    # width = 0.25
    width = 0.35
    rects1 = ax7.barh(ind, meanIndivAcc, width, color='red', alpha=0.4,label='training')
    rects2 = ax7.barh(ind+width, meanValIndiviAcc, width, color='blue', alpha=0.4,label='validation')

    # rects3 = ax7.barh(ind+width*2, indivAccRef, width, color='green', alpha=0.4,label='validation')

    # ax7.set_yticks((ind+width), individuals)
    ax7.set_xlim((0,1))
    ax7.set_ylim((0,numIndiv+1))
    ax7.set_ylabel('individual')
    ax7.set_title('Individual accuracy')
    ax7.legend(fancybox=True, framealpha=0.05)

    # k=0
    # ax_feats = []
    # for i in range(30):
    #     ax8 = plt.subplot(10,12,(i % 3)+10+12*k)
    #     ax_feats.append(ax8)
    #     if i % 3 == 2:
    #         k+=1
    #     ax8.imshow(features[i], interpolation='none', cmap='gray')
    #     ax8.set_ylabel('Indiv' + str(labels[i]))
    # print fig.get_children()
    # plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, right=.9, left=0.1, top=.9, wspace = 0.5, hspace=0.5)
    plt.draw()
    plt.pause(1)

def CNNplotterFastNoses(lossDict):
    # get variables
    lossPlot, valLossPlot, lossSpeed,valLossSpeed, lossAccel, valLossAccel, \
    coord, coord_hat, miniframes = getVarFromDict(lossDict,[
        'loss', 'valLoss', 'lossSpeed', 'valLossSpeed', 'lossAccel', 'valLossAccel',
        'coordinate', 'coordinate_hat', 'miniframes'])
    try:
        fig = plt.gcf()
    except:
        fig = plt.figure()
    fig.clear()

    plt.switch_backend('TkAgg')
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())


    # loss
    ax1 = plt.subplot(231)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
    ax1.set_axis_bgcolor('none')

    ax1.plot(lossPlot[1:],'or-', label='training')
    ax1.plot(valLossPlot[1:], 'ob--', label='validation')
    ax1.set_ylabel('Loss function')
    ax1.legend(fancybox=True, framealpha=0.05)

    ax2 = plt.subplot(234)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.get_xaxis().tick_bottom()
    ax2.get_yaxis().tick_left()
    ax2.set_axis_bgcolor('none')

    ax2.plot(lossSpeed[1:],'ro-',label='training')
    plt.plot(valLossSpeed[1:],'bo--',label='validation')
    ax2.set_ylabel('Loss function speed')

    k=2
    ax_feats = []
    ax8 = plt.subplot(232)
    ax8.imshow(miniframes[0], cmap='gray', interpolation='none')
    ax8.scatter(coord[0,0],coord[0,1], c='r') # remark: the nose of the fish is red since we are writing this piece of
    ax8.scatter(coord[0,2],coord[0,3], c='b') #code during christmas time!

    ax8.scatter(coord_hat[0,0],coord_hat[0,1], c='r', marker='v')
    ax8.scatter(coord_hat[0,2],coord_hat[0,3], c='b', marker='v')

    # for i in range(1,2):
    #     ax8 = plt.subplot(4,6,k+i)
    #     ax_feats.append(ax8)
    #     if (k + i) % 6 == 0:
    #         k+= 2
    #     minif = miniframes[i]
    #     minif[minif == 0] = 255
    #     ax8.imshow(minif, cmap='gray', interpolation='none')
    #     ax8.scatter(coord[i,0],coord[i,1], c='r') # remark: the nose of the fish is red since we are writing this piece of
    #     ax8.scatter(coord[i,2],coord[i,3], c='b') #code during christmas time!
    #
    #     ax8.scatter(coord_hat[i,0],coord_hat[i,1], c='r', marker='v')
    #     ax8.scatter(coord_hat[i,2],coord_hat[i,3], c='b', marker='v')
    #
    #     # ax8.set_xlim((0,1))
    #     # ax8.set_ylim((0,1))
    print coord
    print coord_hat

    plt.subplots_adjust(bottom=0.1, right=.9, left=0.1, top=.9, wspace = 0.5, hspace=0.5)
    plt.draw()
    plt.pause(1)


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
