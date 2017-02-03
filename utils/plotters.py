import os
import sys
sys.path.append('../utils')

from py_utils import *

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import cPickle as pickle
import pyautogui

''' plot and save fragment selected '''
def orderVideo(matrixToOrder,permutations,maxNumBlobs):
    matrixOrdered = np.zeros_like(matrixToOrder)

    for frame in range(len(permutations)):
        for i in range(maxNumBlobs):
            index = list(np.where(permutations[frame]==i)[0])
            # print index
            if len(index) == 1:
                matrixOrdered[frame,i] = matrixToOrder[frame,index]
            else:
                matrixOrdered[frame,i] = -1

    return matrixOrdered

def fragmentAccumPlotter(fragmentsDict,portraits,accumDict,figurePath):

    fragsForTrain = accumDict['fragsForTrain']
    accumCounter = accumDict['counter']

    fragments = fragmentsDict['fragments']
    permutations = np.asarray(portraits.loc[:,'permutations'].tolist())
    maxNumBlobs = len(permutations[0])
    permOrdered =  orderVideo(permutations,permutations,maxNumBlobs)
    permOrdered = permOrdered.T.astype('float32')

    ''' Fragments '''
    fig = plt.figure("fragments-accum",figsize=(25, 5))
    plt.clf()
    ax = fig.add_subplot(111)
    permOrdered[permOrdered >= 0] = .5
    im = plt.imshow(permOrdered,cmap=plt.cm.gray,interpolation='none',vmin=0.,vmax=1.)
    im.cmap.set_under('r')

    for i in fragsForTrain:
        ax.add_patch(
            patches.Rectangle(
                (fragments[i,0], -0.5),   # (x,y)
                fragments[i,1]-fragments[i,0],  # width
                maxNumBlobs,          # height
                fill=True,
                edgecolor=None,
                facecolor='b',
                alpha = 0.5
            )
        )

    plt.axis('tight')
    plt.xlabel('Frame number')
    plt.ylabel('Blob index')
    plt.gca().set_yticks(range(0,maxNumBlobs,4))
    plt.gca().set_yticklabels(range(1,maxNumBlobs+1,4))
    plt.gca().invert_yaxis()
    plt.tight_layout()

    print 'Saving figure...'
    figname = figurePath + '/fragments_' + str(accumCounter) + '.pdf'
    fig.savefig(figname)

def P2AccumPlotter(fragmentsDict,portraits,accumDict,figurePath,ckpt_dir):

    fragsForTrain = accumDict['fragsForTrain']
    accumCounter = accumDict['counter']

    fragments = fragmentsDict['fragments']
    permutations = np.asarray(portraits.loc[:,'permutations'].tolist())
    maxNumBlobs = len(permutations[0])
    permOrdered =  orderVideo(permutations,permutations,maxNumBlobs)
    permOrdered = permOrdered.T.astype('float32')

    ''' P2 '''
    statistics = pickle.load( open( ckpt_dir + "/statistics.pkl", "rb" ) )
    P2 = statistics['P2FragAllVideo']
    P2Ordered =  orderVideo(P2,permutations,maxNumBlobs)
    P2good = np.max(P2Ordered,axis=2).T

    fig = plt.figure("P2-accum",figsize=(25, 5))
    plt.clf()
    ax = fig.add_subplot(111)
    im2 = plt.imshow(P2good,cmap=plt.cm.gray,interpolation='none')
    im2.cmap.set_under('r')
    im2.set_clim(0, 1)
    cb = plt.colorbar(im2)
    # fig.colorbar(im, ax=ax)
    plt.axis('tight')
    plt.xlabel('Frame number')
    plt.ylabel('Blob index')
    plt.gca().set_yticks(range(0,maxNumBlobs,4))
    plt.gca().set_yticklabels(range(1,maxNumBlobs+1,4))
    plt.gca().invert_yaxis()
    plt.tight_layout()

    print 'Saving figure...'
    figname = figurePath + '/P2_' + str(accumCounter) + '.pdf'
    fig.savefig(figname)

def CNNplotterFast22(lossAccDict,weightsDict,accumDict,fragmentsDict,portraits,sessionPath,show=False):

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
    # print w,h
    fig = plt.figure("fine-tuning", figsize=(w/(2*96),h/96))
    plt.clf()

    if len(lossPlot) < 100:
        xlim = 100
    elif len(lossPlot) < 250:
        xlim = 250
    elif len(lossPlot) < 500:
        xlim = 500
    elif len(lossPlot) < 750:
        xlim = 750
    else:
        xlim = 1000

    # loss
    ax1 = fig.add_subplot(611)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
    ax1.set_axis_bgcolor('none')

    ax1.plot(lossPlot,'r-', label='training')
    ax1.plot(valLossPlot, 'b-', label='validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss function')
    ax1.legend(fancybox=True, framealpha=0.05)
    ax1.set_xlim((0,xlim))
    ax1.set_ylim((0,2.))

    # accuracy
    ax2 = fig.add_subplot(612)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.get_xaxis().tick_bottom()
    ax2.get_yaxis().tick_left()
    ax2.set_axis_bgcolor('none')

    ax2.plot(accPlot, 'r-')
    ax2.plot(valAccPlot, 'b-')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuray')
    ax2.set_xlim((0,xlim))
    ax2.set_ylim((0,1))


    # Individual accuracies
    ax3 = fig.add_subplot(613)
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
    ax4 = fig.add_subplot(6,3,10)
    ax4.imshow(np.squeeze(WConv1),interpolation='none',cmap='gray',vmin=0, vmax=1)
    ax4.set_title('Conv1 filters')
    ax4.xaxis.set_ticklabels([])
    ax4.yaxis.set_ticklabels([])

    # W3
    ax5 = fig.add_subplot(6,3,11)
    ax5.imshow(np.squeeze(WConv3),interpolation='none',cmap='gray',vmin=0, vmax=1)
    ax5.set_title('Conv2 filters')
    ax5.xaxis.set_ticklabels([])
    ax5.yaxis.set_ticklabels([])

    # W5
    ax6 = fig.add_subplot(6,3,12)
    ax6.imshow(np.squeeze(WConv5),interpolation='none',cmap='gray',vmin=0, vmax=1)
    ax6.set_title('Conv3 filters')
    ax6.xaxis.set_ticklabels([])
    ax6.yaxis.set_ticklabels([])

    # Fragments accumulation
    fragsForTrain = accumDict['fragsForTrain']
    accumCounter = accumDict['counter']

    fragments = fragmentsDict['fragments']
    permutations = np.asarray(portraits.loc[:,'permutations'].tolist())
    maxNumBlobs = len(permutations[0])
    permOrdered =  orderVideo(permutations,permutations,maxNumBlobs)
    permOrdered = permOrdered.T.astype('float32')

    ax7 = fig.add_subplot(615)
    permOrdered[permOrdered >= 0] = .5
    im = plt.imshow(permOrdered,cmap=plt.cm.gray,interpolation='none',vmin=0.,vmax=1.)
    im.cmap.set_under('r')

    for i in fragsForTrain:
        ax7.add_patch(
            patches.Rectangle(
                (fragments[i,0], -0.5),   # (x,y)
                fragments[i,1]-fragments[i,0],  # width
                maxNumBlobs,          # height
                fill=True,
                edgecolor=None,
                facecolor='b',
                alpha = 0.5
            )
        )

    ax7.axis('tight')
    ax7.set_xlabel('Frame number')
    ax7.set_ylabel('Blob index')
    ax7.set_yticks(range(0,maxNumBlobs,4))
    ax7.set_yticklabels(range(1,maxNumBlobs+1,4))
    ax7.invert_yaxis()

    # P2
    P2 = accumDict['overallP2']

    ax8 = fig.add_subplot(616)
    ax8.spines["top"].set_visible(False)
    ax8.spines["right"].set_visible(False)
    ax8.plot(range(len(P2)),P2,'or-')
    ax8.set_xlabel('Accumulation step')
    ax8.set_ylabel('Overall P2')




    plt.subplots_adjust(bottom=0.05, right=.95, left=0.05, top=.95, wspace = 0.25, hspace=0.25)

    plt.draw()
    plt.pause(0.00000001)
