import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('../../utils')
from py_utils import *

def contrastive_loss1(features1, features2, targets):

    # diff[i] = |f1[i] - f2[i]|
    diff = np.abs(np.subtract(features1,features2)) # N x 2
    # diff = tf.abs(tf.sub(features1,features2)) # N x 2
    # print 'diff **********'
    # print(sess.run(diff))

    # E_w = \sum_i diff[i]
    l1dist = np.sum(diff, axis=1)  # N
    print l1dist
    def Lg(Q,l1dist):
        # 2/Q
        coeff = np.true_divide(2.,Q)
        # E_w^2
        l1dist2 = np.power(l1dist,2)
        # 2/Q * E_w
        return np.multiply(coeff,l1dist2)

    Q = 6000.
    LG = Lg(Q,l1dist)# 2/Q * E_w^2

    def Li(Q,l1dist):
        # 2 * Q
        coeff1 = np.multiply(2.,Q)
        # -2.77 / Q
        coeff2 = np.true_divide(-2.77,Q)
        # (-2.77 / Q) * E_W
        exponent = np.multiply(coeff2,l1dist)
        #  exp((-2.77 / Q) * E_W)
        exp = np.exp(exponent)
        # 2 * Q * exp((-2.77 / Q) * E_W)
        return np.multiply(coeff1,exp)

    LI = Li(Q,l1dist) # 2 * Q * exp((-2.77 / Q) * E_W)
    # (1 - y) * (2/Q * E_w^2)
    firstadd = np.multiply(targets,LG)
    # y * (2 * Q * exp((-2.77 / Q) * E_W))
    # REMARK: in our case genuine pairs are labelled with ones and
    # impostors correspond to zero labels
    secondadd = np.multiply(np.subtract(1.,targets), LI)
    loss = np.add(firstadd,secondadd)

    return LG, LI, loss

def generateFeaturesAndLabels(length, batch):
    # a feature is an array whose dimensionality depends on
    # the size of the last layer of the network
    # label takes values in {0,1}, 0 to generate impostor and 1
    # for genuine pairs, respectively.
    features1 = []
    features2 = []
    labels = []
    for b in range(batch) :
        features1.append(np.random.rand(length))
        features2.append(np.random.rand(length))
        labels.append(np.random.randint(2))
    return features1, features2, labels


def plot3DLoss(length):
    features1,features2, targets = generateFeaturesAndLabels(length,100)
    # print targets
    LG, LI, loss = contrastive_loss1(features1, features2, targets)
    # print '------------------------'
    # print LG
    # print '------------------------'
    # print LI
    # print '------------------------'
    # print loss
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(LG, LI, loss, c=loss, marker='.')
    ax.set_xlabel('LG')
    ax.set_ylabel('LI')
    ax.set_zlabel('Loss')
    plt.show()

plot3DLoss(10000)
