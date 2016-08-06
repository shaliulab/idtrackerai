import tensorflow as tf
import numpy as np
from tf_utils import *
import sys
if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('GtkAgg')
import matplotlib.pyplot as plt
# Create some wrappers for simplicity
def computeVolume(width, height, strides):
    c1 = float(strides[1])
    c2 = float(strides[2])
    widthS = int(np.ceil(width/c1))
    heightS = int(np.ceil(height/c2))
    return widthS, heightS

def buildConv2D(varInit,inputVol,params):
    '''
    varInit .
        Wname: is a string
        Winit:
        Bname:
        Binit:
    inputVol
        inW
        inH
        inD
        inVol
    params
        filtSize
        numFilt
        stride
        pad
    '''
    # WConv = weight_variable([filter_size, filter_size, inputDepth, n_filters])
    # bConv = bias_variable([n_filters])
    shapeW = [params['filtSize'], params['filtSize'],inputVol['inD'], params['numFilt']]
    WConv = tf.get_variable(
        varInit['Wname'],
        shapeW,
        initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01,seed=0))
    bConv = tf.get_variable(
        varInit['Bname'],
        params['numFilt'],
        initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01,seed=0))
    hConv = tf.nn.conv2d(input=inputVol['inVol'],
                 filter=WConv,
                 strides=params['stride'],
                 padding=params['pad']) + bConv
    w,h = computeVolume(inputVol['inW'], inputVol['inH'], params['stride'])
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

def buildFc(varInit,inputVol,params):
    '''
    varInit .
        Wname:
        Winit:
        Bname:
        Binit:
    inputVol
        inW
        inH
        inD
        inVol
    params
        numNeurons
        keep_prob
    '''
    shapeW = [inputVol['inH']*inputVol['inW']*inputVol['inD'], params['numNeurons']]
    W = tf.get_variable(
        varInit['Wname'],
        shapeW,
        initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01,seed=0))
    b = tf.get_variable(
        varInit['Bname'],
        params['numNeurons'],
        initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01,seed=0))

    h_fc = tf.matmul(inputVol['inVol'], W) + b
    h_fc_drop = tf.nn.dropout(h_fc, params['keep_prob'])
    return h_fc_drop

def buildSoftMax(inputSoftMax,n_fc,classes):
    W_fc = weight_variable([n_fc, classes])
    b_fc = bias_variable([classes])
    y_logits = tf.matmul(inputSoftMax, W_fc) + b_fc
    return y_logits

def CNNplotter(lossPlot,accPlot,features, labels,numIndiv):
    features = [inner for outer in features for inner in outer]
    features = np.asarray(features)

    labels = one_hot_to_dense(labels)
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999',
         '#505050', '#ffffff', '#ff9966', '#cccccc', '#ff6699']
    leg = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']


    plt.close()
    plt.figure
    plt.switch_backend('TkAgg')
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.subplot(131)
    plt.plot(lossPlot,'o-')

    plt.subplot(132)
    plt.plot(accPlot, 'or-')

    plt.subplot(133)

    for i in range(numIndiv):
        # print len(features)
        # print len(labels)
        # print np.where(labels == i)[0]
        featThisIndiv = features[np.where(labels == i)[0]]

        # ax.plot(feat[:, 0], feat[:, 1],feat[:, 2], '.', c=c[i])
        plt.plot(featThisIndiv[:, 0], featThisIndiv[:, 1], '.', c=c[i])
    plt.legend(leg[:numIndiv])

    plt.draw()
    plt.pause(1)
