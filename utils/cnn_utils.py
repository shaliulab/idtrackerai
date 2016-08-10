import tensorflow as tf
import numpy as np
from tf_utils import *
import sys
if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('GtkAgg')
import matplotlib.pyplot as plt
import warnings
from py_utils import *

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

def buildConv2D(Wname, Bname, inputWidth, inputHeight, inputDepth, inputConv ,filter_size, n_filters, stride, pad):
    # WConv = weight_variable([filter_size, filter_size, inputDepth, n_filters])
    # bConv = bias_variable([n_filters])
    WConv = tf.get_variable(
        Wname,
        [filter_size, filter_size, inputDepth, n_filters],
        initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01)
        )
    bConv = tf.get_variable(
        Bname,
        [n_filters],
        initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01)
        )
    hConv = tf.nn.conv2d(input=inputConv,
                 filter=WConv,
                 strides=stride,
                 padding=pad) + bConv
    w,h = computeVolume(inputWidth, inputHeight, stride)
    return hConv,w,h


def maxpool2d(inputWidth, inputHeight, inputPool, pool=2 , stride=[1,2,2,1] ,pad='VALID'):
    # MaxPool2D wrapper
    max_pool = tf.nn.max_pool(
        inputPool,
        ksize=[1, pool, pool, 1],
        strides=stride,
        padding=pad)
    w, h = computeVolume(inputWidth, inputHeight, stride)
    return max_pool, w, h

def buildFc(Wname, Bname,inputFc,height,width,n_filters,n_fc,keep_prob):
    W_fc = tf.get_variable(
        Wname,
        [height * width * n_filters, n_fc],
        initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01)
        )
    b_fc = tf.get_variable(
        Bname,
        [n_fc],
        initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01)
        )
    h_fc = tf.matmul(inputFc, W_fc) + b_fc
    h_fc_drop = tf.nn.dropout(h_fc, keep_prob)
    return h_fc_drop

def buildSoftMax(Wname, Bname,inputSoftMax,n_fc,classes):
    W_fc = tf.get_variable(
        Wname,
        [n_fc, classes],
        initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01)
        )
    b_fc = tf.get_variable(
        Bname,
        [classes],
        initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01)
        )
    y_logits = tf.matmul(inputSoftMax, W_fc) + b_fc
    return y_logits

def buildSoftMaxWeights(inputSoftMax,n_fc,classes):
    # the same as build softmax, but outputs the weights for visualization
    W_fc = weight_variable([n_fc, classes])
    b_fc = bias_variable([classes])
    y_logits = tf.matmul(inputSoftMax, W_fc) + b_fc
    return y_logits, W_fc, b_fc

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

def CNNplotterFast(lossAccDict):

    # get variables
    lossPlot, valLossPlot, lossSpeed,valLossSpeed, lossAccel, valLossAccel, \
    accPlot, valAccPlot, accSpeed,valAccSpeed, accAccel, valAccAccel, \
    indivAcc,indivValAcc,\
    features, labels = getVarFromDict(lossAccDict,[
        'loss', 'valLoss', 'lossSpeed', 'valLossSpeed', 'lossAccel', 'valLossAccel',
        'acc', 'valAcc', 'accSpeed', 'valAccSpeed', 'accAccel', 'valAccAccel',
        'indivAcc', 'indivValAcc',
        'features', 'labels'])

    meanIndivAcc = indivAcc[-1]
    meanValIndiviAcc = indivValAcc[-1]
    numIndiv = len(meanIndivAcc)
    features = features[:30]
    features = np.reshape(features, [features.shape[0],10,10])
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
    # plt.plot(valLossSpeed,'bo--',label='validation')
    ax2.set_ylabel('Loss function speed')

    ax3 = plt.subplot(263)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.get_xaxis().tick_bottom()
    ax3.get_yaxis().tick_left()
    ax3.set_axis_bgcolor('none')

    ax3.plot(lossAccel,'ro-',label='training')
    # plt.plot(valLossAccel,'bo--',label='validation')
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
    # plt.plot(valAccSpeed,'bo--',label='validation')
    ax5.set_ylabel('Accuray speed')

    ax6 = plt.subplot(2,6,9)
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)
    ax6.get_xaxis().tick_bottom()
    ax6.get_yaxis().tick_left()
    ax6.set_axis_bgcolor('none')

    ax6.plot(accAccel,'ro-',label='training')
    # plt.plot(valAccAccel,'bo--',label='validation')
    ax6.set_ylabel('Accuray accel.')

    # Individual accuracies
    ax7 = plt.subplot(1, 4, 3)
    ax7.spines["top"].set_visible(False)
    ax7.spines["right"].set_visible(False)
    ax7.get_xaxis().tick_bottom()
    ax7.get_yaxis().tick_left()
    ax7.set_axis_bgcolor('none')

    individuals = [str(j) for j in range(1,numIndiv+1)]
    ind = np.arange(numIndiv)
    width = 0.35
    rects1 = ax7.barh(ind, meanIndivAcc, width, color='red', alpha=0.4,label='training')

    rects2 = ax7.barh(ind+width, meanValIndiviAcc, width, color='blue', alpha=0.4,label='validation')

    ax7.set_yticks((ind+width), individuals)
    ax7.set_xlim((0,1))
    ax7.set_ylabel('individual')
    ax7.set_title('Individual accuracy')
    ax7.legend(fancybox=True, framealpha=0.05)

    k=0
    ax_feats = []
    for i in range(30):
        ax8 = plt.subplot(10,12,(i % 3)+10+12*k)
        ax_feats.append(ax8)
        if i % 3 == 2:
            k+=1
        ax8.imshow(features[i], interpolation='none', cmap='gray')
        ax8.set_ylabel('Indiv' + str(labels[i]))
    # print fig.get_children()
    # plt.tight_layout()
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
        warnings.warn('Not enough points to compute the speed')
        speed = []

    if len(array) > 2:
        accel = np.diff(speed)
    else:
        warnings.warn('Not enough points to compute the acceleration')
        accel = []

    return speed, accel

# y = np.array([[1,0],[0,1],[1,0],[0,1]])
# y_logits = np.array([[1,0],[0,1],[1,0],[1,0]])
# computeROCAccuracy(y, y_logits, name=None)
