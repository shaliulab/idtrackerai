import tensorflow as tf
import numpy as np
import sys
if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('GtkAgg')
    '''TkAgg WX QTAgg QT4Agg '''
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
sys.path.append('../utils')
from py_utils import *

def contrastive_loss1(features1, features2, targets, name=None):
    sess = tf.Session()
    # diff[i] = |f1[i] - f2[i]|
    diff = tf.abs(tf.sub(features1,features2)) # N x 2
    # E_w = \sum_i diff[i]
    l1dist = tf.reduce_sum(diff, 1)  # N
    # Find sup(E_w) [REMARK: this computation is correct iff units in the last
    # layer of the CNN are relu6]
    numFeat = tf.shape(features1)[1]
    Q = tf.multiply(tf.cast(numFeat, tf.float32), 6.)
    def Lg(Q,l1dist):
        # 2/Q
        coeff = tf.div(2.,Q)
        # E_w^2
        l1dist2 = tf.square(l1dist)
        # 2/Q * E_w
        return tf.multiply(coeff,l1dist2)

    def Li(Q,l1dist):
        # 2 * Q
        coeff1 = tf.multiply(2.,Q)
        # -2.77 / Q
        coeff2 = tf.div(-2.77,Q)
        # (-2.77 / Q) * E_W
        exponent = tf.multiply(coeff2,l1dist)
        #  exp((-2.77 / Q) * E_W)
        exp = tf.exp(exponent)
        # 2 * Q * exp((-2.77 / Q) * E_W)
        return tf.multiply(coeff1,exp)

    LG = Lg(Q,l1dist)# 2/Q * E_w^2
    LI = Li(Q,l1dist) # 2 * Q * exp((-2.77 / Q) * E_W)
    # y * (2 * Q * exp((-2.77 / Q) * E_W))
    firstadd = tf.multiply(targets,LG)
    # (1 - y) * (2/Q * E_w^2)
    secondadd = tf.multiply(tf.sub(1.,targets), LI)
    loss = tf.add(firstadd,secondadd)
    return loss

def contrastive_loss2(features1, features2, targets, name=None):
    diff = tf.sub(features1,features2)  # N x 2
    dist2 = tf.reduce_sum(tf.square(diff), 1)  # N
    l2dist = tf.sqrt(dist2)  # N
    mdist = tf.sub(1.,l2dist)
    # reduce before computeing maximum
    # maxdist = tf.reduce_max(mdist,0)
    zerosTensor = tf.zeros_like(mdist)
    dist = tf.maximum(mdist, zerosTensor)
    # changed 0 in 1
    firstadd = tf.multiply(targets, dist2)
    secondadd = tf.multiply(tf.sub(1.,targets),tf.square(dist))
    loss = tf.add(firstadd, secondadd)

    return loss

def computeROCAccuracy(features1, features2, targets, name=None):
    # sess = tf.Session()
    diff = tf.abs(tf.sub(features1,features2)) # N x 2
    l1dist = tf.reduce_sum(diff, 1)  # N

    # set the number of thresholds to be considered
    numThreshold = tf.constant(100)
    ths = tf.linspace(tf.reduce_min(l1dist),tf.reduce_max(l1dist),numThreshold)
    # print 'ths *******'
    # print sess.run(ths)
    l1distRep = tf.reshape(tf.tile(l1dist, [numThreshold]), [numThreshold, tf.shape(features1)[0]])
    # print 'l1distRep *******'
    # print sess.run(l1distRep)
    # reshape the targets to be a tensor of the right size
    targetsRep = tf.reshape(tf.tile(targets, [numThreshold]), [numThreshold, tf.shape(features1)[0]])
    # print 'targetsRep *******'
    # print sess.run(targetsRep)
    thsRep = tf.transpose(tf.reshape(tf.tile(ths, [tf.shape(features1)[0]]), [tf.shape(features1)[0],numThreshold]))
    # thsRep = tf.transpose(ths)
    # print 'thsRep *******'
    # print sess.run(thsRep)
    thTargets = tf.less_equal(tf.sub(l1distRep,thsRep),0.)
    # print 'thTargets *******'
    # print sess.run(tf.cast(thTargets,tf.float32))


    TP = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(targetsRep,True),tf.equal(thTargets,True)), tf.float32), 1)
    FP = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(targetsRep,False),tf.equal(thTargets,True)), tf.float32), 1)
    # print 'TP *******'
    # print sess.run(TP)
    # print 'FP *******'
    # print sess.run(FP)

    # thTargets = tf.stack([l1dist <= ths[i] for i in range(numThreshold)])
    #
    # TP = tf.stack([tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(targets,True),tf.equal(thTargets[:,i],True)),tf.float32)) for i in range(numThreshold)])
    # FP = tf.stack([tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(targets,False),tf.equal(thTargets[:,i],True)),tf.float32)) for i in range(numThreshold)])
    # # FN = [tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(targets,True),tf.equal(thTarget,False)),tf.float32)) for thTarget in thTargets]
    # # TN = [tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(targets,False),tf.equal(thTarget,False)),tf.float32)) for thTarget in thTargets]

    TPR = tf.div(TP,tf.reduce_sum(tf.cast(tf.equal(targets,1),tf.float32)))
    FPR = tf.div(FP,tf.reduce_sum(tf.cast(tf.equal(targets,0),tf.float32)))
    # print 'TPR *******'
    # print sess.run(TPR)
    # print 'FPR *******'
    # print sess.run(FPR)

    optimalThInd = tf.cast(tf.argmax(tf.add(TPR,tf.sub(1.,FPR)),0),tf.int32)

    TPRshift1 = TPR[1:100]
    FPRshift1 = FPR[1:100]
    factor1 = tf.sub(FPRshift1, FPR[:100-1])
    factor2 = tf.div(tf.add(TPRshift1, TPR[:100-1]),2.)
    trapzROC = tf.multiply(factor1, factor2)
    acc = tf.reduce_sum(trapzROC)
    # print sess.run(acc)
    return FPR, TPR, acc, optimalThInd, ths
########### Test ROC
# features1 = tf.constant([[1.,5.],[2.,5.],[1.,4.],[2.,4.],[3.,3.],[4.,3.],[5.,2.],[5.,1.],[6.,2.],[6.,1.]])
# features2 = tf.constant([[3.,3.],[2.,4.],[1.,4.],[5.,1.],[4.,3.],[5.,2.],[3.,3.],[2.,5.],[6.,1.],[1.,5.]])
# targets = tf.constant([1.,1.,1.,1.,1.,0.,0.,0.,0.,0.])
# computeROCAccuracy(features1, features2, targets, name=None)
# print accuracy(features1, features2, targets, name=None)
# plt.show()
############# Test areaROC
# features1 = tf.constant([[1.,5.],[2.,5.],[6.,2.],[6.,1.]])
# features2 = tf.constant([[3.,3.],[2.,4.],[6.,1.],[1.,5.]])
# targets = tf.constant([1.,1.,0.,0.])
# computeROCAccuracy(features1, features2, targets, name=None)
# ths *******
# [ 1.  3.  5.  7.  9.]
# l1distRep *******
# [[ 4.  1.  1.  9.]
#  [ 4.  1.  1.  9.]
#  [ 4.  1.  1.  9.]
#  [ 4.  1.  1.  9.]
#  [ 4.  1.  1.  9.]]
# targetsRep *******
# [[ 1.  1.  0.  0.]
#  [ 1.  1.  0.  0.]
#  [ 1.  1.  0.  0.]
#  [ 1.  1.  0.  0.]
#  [ 1.  1.  0.  0.]]
# thsRep *******
# [[ 1.  1.  1.  1.]
#  [ 3.  3.  3.  3.]
#  [ 5.  5.  5.  5.]
#  [ 7.  7.  7.  7.]
#  [ 9.  9.  9.  9.]]
# thTargets *******
# [[ 0.  1.  1.  0.]
#  [ 0.  1.  1.  0.]
#  [ 1.  1.  1.  0.]
#  [ 1.  1.  1.  0.]
#  [ 1.  1.  1.  1.]]
# TP *******
# [ 1.  1.  2.  2.  2.]
# FP *******
# [ 1.  1.  1.  1.  2.]
# TPR *******
# [ 0.5  0.5  1.   1.   1. ]
# # FPR *******
# [ 0.5  0.5  0.5  0.5  1. ]
# Accuracy
# 0.5

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


def plotterSiameseTest(features, labels, numIndiv):

    c = get_spaced_colors(numIndiv)
    leg = get_legend_str(numIndiv)

    plt.figure
    # plt.switch_backend('TkAgg')
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())

    for i in range(numIndiv):
        featThisIndiv = features[np.where(labels-1 == i)]

        # ax.plot(feat[:, 0], feat[:, 1],feat[:, 2], '.', c=c[i])
        plt.plot(featThisIndiv[:, 0], featThisIndiv[:, 1], '.', c=c[i],alpha=0.5)
    plt.legend(leg[:numIndiv])
    plt.ioff()
    plt.show()

def plotterSiameseAccuracy(lossTrain,lossVal,accTrain,accVal,TPRval,FPRval,featuresTrain,featuresVal,lab1train,lab1val,lab2train,lab2val,numIndiv):
    '''plotterSiameseAccuracy(
        lossTrainPlot, lossValPlot,
        npTPRtrain, npTPRval,
        npFPRval, npFPRtrain,
        clusterPlotTrain, clusterPlotVal,
        Y1_train, Y1_valid,
        Y2_train, Y2_valid,
        numIndiv)'''

    c = get_spaced_colors(numIndiv)
    legLabels = get_legend_str(numIndiv)
    print len(legLabels)

    labelsTrain = np.hstack((lab1train,lab2train))
    labelsVal = np.hstack((lab1val,lab2val))
    # print np.shape(features['h1'])
    # print np.shape(features['h2'])
    featTrain = np.vstack((featuresTrain['h1'],featuresTrain['h2']))
    featVal = np.vstack((featuresVal['h1'],featuresVal['h2']))
    # print np.shape(labels)
    # print np.shape(feat)
    featTrainF1HD = featuresTrain['f1'][1]
    feat1HDlen = len(featTrainF1HD)/5
    featTrainF1HD = np.reshape(featTrainF1HD,[5,feat1HDlen])

    plt.close()
    plt.figure
    plt.switch_backend('TkAgg')
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    plt.subplot(341)
    plt.plot(lossTrain,'ro-',label='training')
    plt.plot(lossVal,'bo-',label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss function')

    plt.legend()

    plt.subplot(342)
    plt.semilogy(lossTrain,'ro-',label='training')
    plt.semilogy(lossVal,'bo-',label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss function')

    # plt.legend()

    plt.subplot(345)
    plt.plot(accTrain,'ro-')
    plt.plot(accVal,'bo-')
    plt.ylabel('Area under Roc curve')

    plt.subplot(346)
    plt.semilogy(accTrain,'ro-')
    plt.semilogy(accVal,'bo-')
    plt.ylabel('Log Area under Roc curve')

    plt.subplot(3,4,9)
    plt.plot(FPRval,TPRval,'b-')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.title('ROC curve for validation set')
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    # plt.subplot(3,4,10)
    # plt.imshow(featTrainF1HD, interpolation='none',cmap='gray')#, vmin=0, vmax=6)
    # plt.colorbar()
    # plt.title('Last layer feature')
    numPairs = 10
    for i in range(numPairs):

        featTrainF1HD = featuresTrain['f1'][i]
        feat1HDlen = len(featTrainF1HD)/10
        featTrainF1HD = np.reshape(featTrainF1HD,[feat1HDlen,10])
        featTrainF2HD = featuresTrain['f2'][i]
        feat2HDlen = len(featTrainF2HD)/10
        featTrainF2HD = np.reshape(featTrainF2HD,[feat2HDlen,10])

        plt.subplot(numPairs,4,3+i*4)
        plt.imshow(featTrainF1HD, interpolation='none',cmap='gray')#, vmin=0, vmax=6)
        plt.axis('off')
        # plt.title('Last layer feature')

        plt.subplot(numPairs,4,4+i*4)
        plt.imshow(featTrainF2HD, interpolation='none',cmap='gray')#, vmin=0, vmax=6)
        plt.axis('off')
        # plt.title('Last layer feature')


    # plt.subplot(222)
    # print featTrain.shape
    # print '..............'
    # print labelsTrain.shape
    # for i in range(numIndiv):
    #     featThisIndiv = featTrain[np.where(labelsTrain == i)]
    #
    #     # ax.plot(feat[:, 0], feat[:, 1],feat[:, 2], '.', c=c[i])
    #     plt.plot(featThisIndiv[:, 0], featThisIndiv[:, 1], '.', c=c[i], alpha=0.01, label=legLabels[i])
    #
    # leg=plt.legend()
    #
    # for l in leg.get_lines():
    #     l.set_alpha(1)
    #     l.set_marker('.')

    plt.subplot(3,4,10)

    for i in range(numIndiv):
        featThisIndiv = featVal[np.where(labelsVal == i)]

        # ax.plot(feat[:, 0], feat[:, 1],feat[:, 2], '.', c=c[i])
        plt.plot(featThisIndiv[:, 0], featThisIndiv[:, 1], '.', c=c[i], alpha=0.2,label=legLabels[i])
    leg=plt.legend()

    for l in leg.get_lines():
        l.set_alpha(1)
        l.set_marker('.')

    plt.draw()

def PrepareDataTest(numRef, X1_test, X2_test, Y1_test, Y2_test):
    # split test in references and problem images according to the labelling
    X = np.vstack((X1_test,X2_test)) #all images
    Y = np.hstack((Y1_test,Y2_test)) #all labels
    numIndivTest = len(list(set(Y))) #count the individuals in test
    X_ref = X[:numRef*numIndivTest]
    Y_ref = Y[:numRef*numIndivTest]

    X_prob = X[numRef*numIndivTest:]
    Y_prob = Y[numRef*numIndivTest:]
    refImages = []
    probImages = []
    refLabels = []
    probLabels = []
    for i in range(numIndivTest):
        # organize references per individuals
        refImages.append(X_ref[Y_ref == i])
        probImages.append(X_prob[Y_prob == i])
        refLabels.append(i*np.ones(sum(Y_ref == i)))
        probLabels.append(i*np.ones(sum(Y_prob == i)))
    # here I modified the script in order to consider the same number of
    # references for every individual
    trueNumRef = min([len(ref) for ref in refImages])
    refImages = [refI[:trueNumRef] for refI in refImages]
    refLabels = [refL[:trueNumRef] for refL in refLabels]
    refImages = flatten(refImages)
    probImages = flatten(probImages)
    refLabels = flatten(refLabels)
    probLabels = flatten(probLabels)
    return refImages, probImages, refLabels, probLabels

def getIds(featProb,featRef,yProb,optimal_threshold,numIndivTest):
    # sess = tf.Session()
    numRef = tf.shape(featRef)[0]
    numProb = tf.shape(featProb)[0]
    numFeat = tf.shape(featRef)[1]
    # print "featRef"
    # print sess.run(featRef)
    # print "optth"
    # print sess.run(optimal_threshold)
    #
    # print "featRef tile with shape featProb 0"
    # print sess.run(tf.reshape(tf.tile(featRef,[1,numProb]),[numRef*numProb,numFeat]))
    # print "featProb tile with shape featRef 0"
    # print sess.run(tf.tile(featProb,[numRef,1]))


    featRefRep = tf.reshape(tf.tile(featRef,[1,numProb]),[numRef*numProb,numFeat])
    featProbRep = tf.tile(featProb,[numRef,1])

    # print "feat Ref Rep"
    # print sess.run(featRefRep)
    # print "feat Ref Rep"
    # print sess.run(featProbRep)
    subtraction = tf.abs(tf.sub(featRefRep,featProbRep))
    dist = tf.reduce_sum(subtraction,1)

    # print "dist"
    # print sess.run(dist)

    match = tf.less_equal(dist, 47.4545)

    # print "match"
    # print sess.run(match)

    reshapeMatch = tf.cast(tf.reshape(match,[numIndivTest,tf.div(numRef,numIndivTest),numProb]),tf.float32)

    # print "reshapeMatch"
    # print sess.run(reshapeMatch)

    counter = tf.reduce_sum(reshapeMatch,1)

    # print "counter"
    # print sess.run(counter)

    Id = tf.cast(tf.argmax(counter,0),tf.float32)

    # print "Id"
    # print sess.run(Id)


    acc = tf.div(tf.reduce_sum(tf.cast(tf.equal(Id,yProb),tf.float32)),tf.cast(numProb,tf.float32))
    #
    # print "acc"
    # print sess.run(acc)
    # return featProbRep, featRefRep, subtraction, dist, match
    return featProb, featRef
    # return counter, Id, acc
# ref = tf.constant([[1.,1.,1.],[1.1,1.1,1.1],[2.,2.,2.],[2.1,2.1,2.1],[3.,3.,3.],[3.1,3.1,3.1]])
# prob = tf.constant([[0.,1.,2.],[1.,2.,3.],[2.,3.,4.],[3.,4.,5.]])
# optTh = 3.
# numIndivTest = 3
# yProb = tf.constant([0,2,1,2])
# getIds(prob,ref,yProb,optTh,numIndivTest)

def plotFailsTraining(plotFails, numIndiv):
    # plotFails is an array organised as follows
    # [sorted_IdDistrib, predictedID, labP_]
    sorted_IdDistrib = np.asarray([plotFail[0] for plotFail in plotFails])
    predictedID = np.asarray([plotFail[1] for plotFail in plotFails])
    groundTruth = np.asarray([plotFail[2] for plotFail in plotFails])
    # divide the groundTruth according to its labelling
    IndivInFails = list(set(groundTruth))
    numIndivFails = len(IndivInFails)
    plt.switch_backend('TkAgg')
    plt.ioff()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    c = 1
    for i in IndivInFails:
        #
        avDistFails = np.zeros(numIndiv)
        individuals = [str(j) for j in range(numIndiv)]
        pos = np.arange(numIndiv)
        IndDistrib = sorted_IdDistrib[np.where(groundTruth == i)]
        for dist in IndDistrib:
            for d in dist:
                avDistFails[int(d[0])] += d[1]
        ax = plt.subplot(np.ceil(numIndivFails/5), 5, c)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        plt.barh(pos, np.true_divide(avDistFails, np.max(avDistFails)), align='center', alpha=0.4)
        plt.yticks(pos, individuals)
        plt.ylabel(str(int(i)))
        c += 1
    plt.show()
#
#
#
# plotFailsTraining([[[(1.0, 20),(2.0, 10)], 1.0, 4.0],\
#                 [[(1.0, 20),(2.0, 10)], 1.0, 4.0],\
#                 [[(1.0, 20),(5.0, 23)], 5.0, 2.0],\
#                 [[(1.0, 20),(5.0, 23)], 5.0, 2.0],\
#                 [[(6.0, 20),(2.0, 10)], 6.0, 2.0]],10)
