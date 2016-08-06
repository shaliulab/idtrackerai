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

# plotting utilities

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


def CNNplotter(lossPlot,accPlot,features, labels,numIndiv):
    features = [inner for outer in features for inner in outer]
    features = np.asarray(features)

    labels = one_hot_to_dense(labels)

    c = get_spaced_colors(numIndiv)
    leg = get_legend_str(numIndiv)


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

def CNNplotterFast(lossPlot,accPlot,valAccPlot,valLossPlot):
    plt.close()
    plt.figure
    plt.switch_backend('TkAgg')
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.subplot(121)
    plt.plot(lossPlot,'or-', label='training')
    plt.plot(valLossPlot, 'ob--', label='validation')
    plt.legend()

    plt.subplot(122)
    plt.plot(accPlot, 'or-')
    plt.plot(valAccPlot, 'ob--')
    plt.draw()
    plt.pause(1)

def CNNplotterROCFast(lossPlot,accPlot,sensPlot,specPlot, valAccPlot, valLossPlot, valSensPlot, valSpecPlot):
    plt.close()
    plt.figure
    plt.switch_backend('TkAgg')
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.subplot(221)
    plt.plot(lossPlot,'or-', label='training')
    plt.plot(valLossPlot, 'ob--', label='validation')
    plt.legend()
    plt.ylabel('Loss')

    plt.subplot(222)
    plt.plot(sensPlot,'or-')
    plt.plot(valSensPlot, 'ob--')
    plt.ylabel('Sensitivity')


    plt.subplot(223)
    plt.plot(accPlot, 'or-')
    plt.plot(valAccPlot, 'ob--')
    plt.ylabel('Accuracy')

    plt.subplot(224)
    plt.plot(specPlot, 'or-')
    plt.plot(valSpecPlot, 'ob--')
    plt.ylabel('Specificity')

    plt.draw()
    plt.pause(1)

def CNNplotterROCFastWeights(weightsTrain, bDocs, weightsVal, bDocsVal, lossPlot,accPlot,sensPlot,specPlot, valAccPlot, valLossPlot, valSensPlot, valSpecPlot):
    plt.close()
    plt.figure
    plt.switch_backend('TkAgg')
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.subplot(231)
    plt.plot(lossPlot,'or-', label='training')
    plt.plot(valLossPlot, 'ob--', label='validation')
    plt.legend()
    plt.ylabel('Loss')

    plt.subplot(232)
    plt.plot(sensPlot,'or-')
    plt.plot(valSensPlot, 'ob--')
    plt.ylabel('Sensitivity')


    plt.subplot(234)
    plt.plot(accPlot, 'or-')
    plt.plot(valAccPlot, 'ob--')
    plt.ylabel('Accuracy')

    plt.subplot(235)
    plt.plot(specPlot, 'or-')
    plt.plot(valSpecPlot, 'ob--')
    plt.ylabel('Specificity')

    plt.subplot(1, 12, 9)
    plt.imshow(weightsTrain, cmap = 'gray', interpolation='none')
    plt.colorbar()

    plt.subplot(1 ,12, 10)
    plt.imshow([bDocs, [0,0]], cmap = 'gray', interpolation='none')

    plt.subplot(1, 12, 11)
    plt.imshow(weightsVal, cmap = 'gray', interpolation='none')
    # plt.colorbar()

    plt.subplot(1, 12, 12)
    plt.imshow([bDocsVal,[0,0]], cmap = 'gray', interpolation='none')

    plt.draw()
    plt.pause(1)


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

# y = np.array([[1,0],[0,1],[1,0],[0,1]])
# y_logits = np.array([[1,0],[0,1],[1,0],[1,0]])
# computeROCAccuracy(y, y_logits, name=None)
