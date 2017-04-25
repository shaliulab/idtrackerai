import os
import sys
sys.path.append('../utils')

from py_utils import *

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
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

def CNNplotterFast22(epoch_counter, epoch_i, handlesDict,lossAccDict, idUsedIndivIntervals, accumDict,fragmentsDict,portraits,sessionPath, plotFlag=True):

    # get variables
    lossPlot, valLossPlot, accPlot, valAccPlot,indivAcc,indivValAcc = getVarFromDict(lossAccDict,['loss', 'valLoss','acc', 'valAcc', 'indivAcc', 'indivValAcc'])
    meanIndivAcc = indivAcc[-1]
    meanValIndiviAcc = indivValAcc[-1]
    numAnimals = len(meanIndivAcc)

    w, h = pyautogui.size()
    # plt.clf()
    if epoch_counter == 0 or handlesDict['restoring']:
        fig, axarr = plt.subplots(5,1,num = "fine-tuning", figsize = (w/(2*96),h/96))
    else:
        fig = plt.gcf()
        axarr = fig.axes

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

    if epoch_counter == 0 or handlesDict['restoring']:
        handlesDict['restoring'] = False
        # loss
        ax1 = axarr[0]
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.get_xaxis().tick_bottom()
        ax1.get_yaxis().tick_left()
        ax1.set_axis_bgcolor('none')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss function')
        ax1.legend(fancybox=True, framealpha=0.05)
        ax1.set_xlim((0,xlim))
        ax1.set_ylim((0,2.))
        handlesDict['TrainLoss'], = ax1.plot(lossPlot,'r-', label='training')
        handlesDict['ValLoss'], = ax1.plot(valLossPlot, 'b-', label='validation')

        # accuracy
        ax2 = axarr[1]
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.get_xaxis().tick_bottom()
        ax2.get_yaxis().tick_left()
        ax2.set_axis_bgcolor('none')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuray')
        ax2.set_xlim((0,xlim))
        ax2.set_ylim((0,1))
        handlesDict['TrainAcc'], = ax2.plot(accPlot, 'r-')
        handlesDict['ValAcc'], = ax2.plot(valAccPlot, 'b-')

        # Individual accuracies
        ax3 = axarr[2]
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.get_xaxis().tick_bottom()
        ax3.get_yaxis().tick_left()
        ax3.set_axis_bgcolor('none')
        ax3.set_ylim((0,1))
        ax3.set_xlim((0,numAnimals+1))
        ax3.set_xlabel('individual')
        ax3.set_ylabel('Individual accuracy')

        width = 0.35
        individuals = [str(j) for j in range(1,numAnimals+1)]
        ind = np.arange(numAnimals)+1
        handlesDict['TrainIndivAcc'] = ax3.bar(ind-width, meanIndivAcc, width, color='red', alpha=0.4,label='training')
        handlesDict['ValIndivAcc'] = ax3.bar(ind, meanValIndiviAcc, width, color='blue', alpha=0.4,label='validation')

    else:

        # loss
        handlesDict['TrainLoss'].set_ydata(lossPlot), handlesDict['TrainLoss'].set_xdata(range(len(lossPlot)))
        handlesDict['ValLoss'].set_ydata(valLossPlot),  handlesDict['ValLoss'].set_xdata(range(len(valLossPlot)))
        ax1 = axarr[0]
        ax1.set_xlim((0,xlim))

        # accuracy
        handlesDict['TrainAcc'].set_ydata(accPlot), handlesDict['TrainAcc'].set_xdata(range(len(accPlot)))
        handlesDict['ValAcc'].set_ydata(valAccPlot),  handlesDict['ValAcc'].set_xdata(range(len(valAccPlot)))
        ax1 = axarr[1]
        ax1.set_xlim((0,xlim))

        # Individual accuracies
        for i in range(numAnimals):
            handlesDict['TrainIndivAcc'][i].set_height(meanIndivAcc[i])
            handlesDict['ValIndivAcc'][i].set_height(meanValIndiviAcc[i])


    if epoch_i == 0:
        # Fragments accumulation
        fragsForTrain = accumDict['fragsForTrain']
        accumCounter = accumDict['counter']

        fragments = fragmentsDict['fragments']
        permutations = np.asarray(portraits.loc[:,'permutations'].tolist())
        maxNumBlobs = len(permutations[0])
        permOrdered =  orderVideo(permutations,permutations,maxNumBlobs)
        permOrdered = permOrdered.T.astype('float32')

        ax4 = axarr[3]
        ax4.cla()
        permOrdered[permOrdered >= 0] = 1.
        im = ax4.imshow(permOrdered,cmap=plt.cm.gray, interpolation='none',vmin=0.,vmax=1.)
        im.cmap.set_under('k')

        colors = get_spaced_colors_util(numAnimals,norm=True)
        # print numAnimals
        # print colors
        for (frag,ID) in idUsedIndivIntervals:
            # print ID
            blobIndex = frag[0]
            start = frag[2][0]
            end = frag[2][1]
            ax4.add_patch(
                patches.Rectangle(
                    (start, blobIndex-0.5),   # (x,y)
                    end-start,  # width
                    1.,          # height
                    fill=True,
                    edgecolor=None,
                    facecolor=colors[ID+1],
                    alpha = 1.
                )
            )

        ax4.axis('tight')
        ax4.set_xlabel('Frame number')
        ax4.set_ylabel('Blob index')
        ax4.set_yticks(range(0,maxNumBlobs,4))
        ax4.set_yticklabels(range(1,maxNumBlobs+1,4))
        ax4.invert_yaxis()

        # P2
        P2 = accumDict['overallP2']

        ax5 = axarr[4]
        ax5.cla()
        ax5.spines["top"].set_visible(False)
        ax5.spines["right"].set_visible(False)
        ax5.plot(range(len(P2)),P2,'or-')
        for i,p2 in enumerate(P2):
            ax5.text(i,p2,str(p2))
        ax5.set_xlabel('Accumulation step')
        ax5.set_ylabel('Overall P2')

        plt.subplots_adjust(bottom=0.05, right=.95, left=0.05, top=.95, wspace = 0.25, hspace=0.25)

    if plotFlag:
        plt.draw()

    plt.pause(0.00000001)

    return handlesDict

def cnn_weights_plotter(weightsDict, epoch_counter):

    # get variables
    W1, W3, W5, W_fc, W_softmax = getVarFromDict(weightsDict,['W1', 'W3','W5', 'W_fc', 'W_softmax'])

    w, h = pyautogui.size()
    # plt.clf()
    if epoch_counter == 0:
        fig = plt.figure(figsize = (w/(96),h/96))
        ax1 = plt.subplot2grid((2,4),(0,0))
        ax2 = plt.subplot2grid((2,4),(0,1))
        ax3 = plt.subplot2grid((2,4),(0,2))
        ax4 = plt.subplot2grid((2,4),(0,3))
        ax5 = plt.subplot2grid((2,4),(1,0),colspan = 4)
        axarr = [ax1,ax2,ax3,ax4,ax5]
    else:
        fig = plt.gcf()
        axarr = fig.axes

    # W1
    ax4 = axarr[0]
    ax4.imshow(np.squeeze(W1),interpolation='none',cmap='gray',vmin=0, vmax=1)
    ax4.set_title('Conv1 filters')
    ax4.xaxis.set_ticklabels([])
    ax4.yaxis.set_ticklabels([])

    # W3
    ax5 = axarr[1]
    ax5.imshow(np.squeeze(W3),interpolation='none',cmap='gray',vmin=0, vmax=1)
    ax5.set_title('Conv2 filters')
    ax5.xaxis.set_ticklabels([])
    ax5.yaxis.set_ticklabels([])

    # W5
    ax6 = axarr[2]
    ax6.imshow(np.squeeze(W5),interpolation='none',cmap='gray',vmin=0, vmax=1)
    ax6.set_title('Conv3 filters')
    ax6.xaxis.set_ticklabels([])
    ax6.yaxis.set_ticklabels([])

    # W_fc
    ax6 = axarr[3]
    ax6.imshow(np.reshape(W_fc,[80,80]),interpolation='none',cmap='gray',vmin=0, vmax=1)
    ax6.set_title('Fc weights')
    ax6.xaxis.set_ticklabels([])
    ax6.yaxis.set_ticklabels([])

    # W_fc
    ax6 = axarr[4]
    ax6.imshow(W_softmax.T,interpolation='none',cmap='gray',vmin=0, vmax=1)
    ax6.set_title('Softmax weights')
    ax6.xaxis.set_ticklabels([])
    ax6.yaxis.set_ticklabels([])

    plt.draw()

    plt.pause(0.00000001)
