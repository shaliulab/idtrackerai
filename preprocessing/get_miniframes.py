import cv2
import sys
sys.path.append('../utils')

from py_utils import *
from video_utils import *

import time
import numpy as np
from matplotlib import pyplot as plt
from Tkinter import *
import tkMessageBox
import argparse
import os
import glob
import pandas as pd
import re
from joblib import Parallel, delayed
import multiprocessing
import cPickle as pickle
import math

def smooth(x,window_len=20,window='hanning'):
   """smooth the data using a window with requested size."""

   if x.ndim != 1:
       raise ValueError, "smooth only accepts 1 dimension arrays."

   if x.size < window_len:
       raise ValueError, "Input vector needs to be bigger than window size."


   if window_len<3:
       return x

   if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
       raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

   s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
   # s = x
   #print(len(s))
   if window == 'flat': #moving average
       w=np.ones(window_len,'d')
   else:
       w=eval('np.'+window+'(window_len)')

   y=np.convolve(s,w/w.sum(),mode='valid')
   return y

def smoother(contour):
    """
    smooth contour by convolution
    """
    window = 11
    X = contour[:,0]
    X = np.append(X,X[:window])
    Y = contour[:,1]
    Y = np.append(Y,Y[:window])

    bkg = np.ones((np.max(Y),np.max(X)))

    G = cv2.transpose(cv2.getGaussianKernel(window, 32, cv2.CV_64FC1))

    X_smooth = np.convolve(X,G[0],mode='same')
    Y_smooth = np.convolve(Y,G[0],mode='same')
    X_smooth = X_smooth[int(np.floor(window/2)):-int(np.ceil(window/2))]
    Y_smooth = Y_smooth[int(np.floor(window/2)):-int(np.ceil(window/2))]
    return X_smooth, Y_smooth

def smooth_resample(contour,smoothFlag = False):
    # print 'smoothing-resampling arclength...'
    if smoothFlag:
        x,y  = smoother(contour)
        # x = smooth(contour[:,0],8)
        # y = smooth(contour[:,1],8)
    else:
        x = contour[:,0]
        y = contour[:,1]

    x_new = x
    y_new = y

    # M = 1000
    M = 1500 ### NOTE we change it to 1500 otherwise was getting trapped inside of the while loop
    t = np.linspace(0, len(x_new), M)
    x = np.interp(t, np.arange(len(x_new)), x_new)
    y = np.interp(t, np.arange(len(y_new)), y_new)
    # tol = .1
    tol = .1
    i, idx = 0, [0]
    # print 'inside the loop'
    while i < len(x):
        total_dist = 0
        for j in range(i+1, len(x)):
            total_dist += np.sqrt((x[j]-x[j-1])**2 + (y[j]-y[j-1])**2)
            if total_dist > tol:
                idx.append(j)
                break

        i = j+1

    xn = x[idx]
    yn = y[idx]
    sContour = [[x,y] for (x,y) in zip(xn,yn)]

    return sContour

def phi(ind, cnt): # FIXME we need to understand the arctan2 and return the correct angle to compute the curvature and do not have artifacts
    cnt = np.squeeze(cnt)
    points = [cnt[(ind-1) % (len(cnt)-1)], cnt[(ind+1) % (len(cnt)-1)]]
    return math.atan2((points[1][1] - points[0][1]), (points[1][0] - points[0][0]))
    # atan = np.arctan2((points[1][1] - points[0][1]), (points[1][0] - points[0][0]))
    # if atan > 0:
    #     atan = np.pi*2 - atan
    # # else:
    # #     atan = -atan
    # return atan


def curv(cnt,i,n,orientation):
    left = (i + n) % (len(cnt)-1)
    right = (i - n) % (len(cnt)-1)
    phi_l = phi(left, cnt) #% 360
    phi_r = phi(right, cnt) #% 360

    if cnt[left+1][0] <= cnt[right+1][0]:
        if phi_l <= 0:
            phi_l += 2*np.pi
        if phi_r <= 0:
            phi_r += 2*np.pi

    return orientation*(phi_l - phi_r) /(2*n+1) # good one

def get_extrema(curvature):
    gradients = np.diff(curvature)
    maxima_num=0
    minima_num=0
    max_locations=[]
    min_locations=[]
    count=0
    for i in gradients[:-1]:
        count+=1

        if ((cmp(i,0)>0) & (cmp(gradients[count],0)<0) & (i != gradients[count])):
            maxima_num+=1
            max_locations.append(count)

        if ((cmp(i,0)<0) & (cmp(gradients[count],0)>0) & (i != gradients[count])):
            minima_num+=1
            min_locations.append(count)

    return np.asarray(max_locations), np.asarray(min_locations)

def getMiddle(points):
    l = points.shape[0]
    b = np.sum(points, axis=0)
    b = np.divide(b,l)
    return b

def full2miniframe(point, boundingBox):
    """
    Push a point in the fullframe to miniframe coordinate system.
    Here it is use for centroids
    """
    return tuple(np.asarray(point) - np.asarray([boundingBox[0][0],boundingBox[0][1]]))

def getEncompassingIndices(frameIndices, num_segmnent, goodIndices):
    """
    frameIndices = dataframe containing the list of frame per segment
    num_segment
    goodIndices = indices in which the permutation is defined (non-crossing and overlapping)
    """
    frameSegment = frameIndices.loc[frameIndices.loc[:,'segment']==num_segmnent]
    goodFrameIndices = frameSegment.iloc[goodIndices].index.tolist()

    return goodFrameIndices, frameSegment.index.tolist()

def getMFandC(videoPath, frameIndices):
    """
    videoPath: videoPath to dataframe
    generate a list of arrays containing miniframes and centroids detected in
    videoPath at this point we can already discard miniframes that does not belong
    to a specific fragments
    """
    # get number of segment
    # video = os.videoPath.basename(videoPath)
    # filename, extension = os.path.splitext(video)
    # numSegment = int(filename.split('_')[-1])
    #load dataframe
    # df = pd.read_pickle(videoPath)
    df, numSegment = loadFile(videoPath, 'segmentation', time=0)
    # print 'you loaded it!'
    # print df
    # check if permutations are NaN (i.e. the frame is not included in a fragment)
    permutationsBool = np.asarray(df['permutation'].notnull())

    #generate a lists of "admissible" miniframes and centroids
    permutations = np.asarray(df['permutation'])
    boundingBoxes = np.asarray(df.loc[:, 'boundingBoxes'])
    miniframes = np.asarray(df.loc[:, 'miniFrames'])
    pixels = np.asarray(df.loc[:, 'pixels'])
    contours = np.asarray(df.loc[:, 'contours'])
    bkgSamples = np.asarray(df.loc[:,'bkgSamples'])
    areas = np.asarray(df.loc[:,'areas'])

    goodIndices = np.where(permutationsBool==True)[0]
    goodFrameIndices, segmentIndices = getEncompassingIndices(frameIndices, int(numSegment), goodIndices)
    goodFrameIndices = segmentIndices
    # boundingBoxes = boundingBoxes[goodIndices]
    # miniframes = miniframes[goodIndices]
    # centroids = centroids[goodIndices]
    # bkgSamples = bkgSamples[goodIndices]
    # permutations = permutations[goodIndices]
    # #
    return boundingBoxes.tolist(), miniframes.tolist(), contours.tolist(), bkgSamples.tolist(), goodFrameIndices, segmentIndices, permutations.tolist(), areas.tolist(), pixels.tolist()

def fillSquareFrame(square_frame,bkgSamps):
    bkgSamps = bkgSamps[bkgSamps > 150]
    numSamples = len(bkgSamps)

    numSamplesRequested = np.sum(square_frame == 0)
    indicesSamples = np.random.randint(0,numSamples,numSamplesRequested)
    square_frame[square_frame == 0] = bkgSamps[indicesSamples]
    return square_frame

def getMiniframes(pixels, miniframe,cnt,bb,bkgSamp,counter = None, path = ''):
    height, width = miniframe.shape
    if path != '':
      widthFrame, heightFrame = getVideoInfo([path])

    orientation = np.sign(cv2.contourArea(cnt,oriented=True)) ### TODO this can probably be optimized

    # Pass contour to bb coord, resample, smooth, and duplicate
    cnt = full2miniframe(cnt, bb)
    cnt = np.asarray(cnt)
    cnt = np.squeeze(cnt)
    cnt = smooth_resample(cnt,smoothFlag=True)
    cnt = np.vstack([cnt,cnt])

    # Compute curvature
    curvature = [curv(cnt,i,3,orientation) for i in range(len(cnt))]
    curvature = np.asarray(curvature)


    # Smooth curvature
    window = 51
    curvature = smooth(curvature, window)
    index = (window-1)/2
    curvature = curvature[index:-index]

    # Crop contour and curvature from the first zero curvature point
    zero = np.argmin(np.abs(curvature[:len(curvature)/2]))
    i1 = zero
    i2 = zero+len(cnt)/2
    cnt = cnt[i1:i2]
    curvature = curvature[i1:i2]

    ### Uncomment to plot
    # plt.close("all")
    # plt.ion()
    # plt.figure()
    # plt.plot(curvature)
    # plt.figure()
    # plt.plot(cnt[:,0],cnt[:,1],'o')
    # plt.show()
    # plt.pause(.5)

    # Find first two maxima (tail and nose), get middle point, and angle of rotation
    max_locations, min_locations = get_extrema(curvature)
    sorted_locations = max_locations[np.argsort(curvature[max_locations])][::-1]
    maxCoord = [cnt[max_loc,:] for max_loc in sorted_locations]
    nose = maxCoord[1]
    cntL = cnt.shape[0]
    steps = 50
    support = np.asarray([cnt[(sorted_locations[1] + steps) % cntL,:], cnt[(sorted_locations[1] - steps) % cntL,:]])
    m = getMiddle(support)
    rot_ang = -(-np.arctan2(nose[1]-m[1],nose[0]-m[0])-np.pi/2)*180/np.pi

    # Copy miniframe in a bigger frame to rotate
    rowsMin, colsMin = miniframe.shape
    # print 'shape of the miniframe, ', miniframe.shape
    diag = np.round(np.sqrt(rowsMin**2 + colsMin**2)).astype('int')
    diag = 128
    new_frame = np.ones((diag,diag)).astype('uint8')*255
    x_offset = np.ceil((diag-colsMin)/2).astype('int')
    y_offset = np.ceil((diag-rowsMin)/2).astype('int')

    new_pixels = np.unravel_index(pixels,(heightFrame,widthFrame))
    new_pixels = zip(new_pixels[1], new_pixels[0])
    new_pixels = [tuple(npx) for npx in new_pixels]
    new_pixels = full2miniframe(new_pixels, bb)
    new_pixels = np.asarray(new_pixels)
    black_miniframe = np.zeros_like(miniframe)
    black_miniframe[new_pixels[:,1], new_pixels[:,0]] = miniframe[new_pixels[:,1], new_pixels[:,0]]
    new_frame[y_offset:y_offset + rowsMin, x_offset:x_offset+colsMin] = black_miniframe

    # Translate and rotate nose and middle point to the new frame
    new_nose = tuple(np.asarray([nose[0]+x_offset, nose[1]+y_offset]).astype('int'))
    new_m = tuple(np.asarray([m[0]+x_offset, m[1]+y_offset]).astype('int'))

    # Get roto-translation matrix and rotate nose and miniframe
    # M = cv2.getRotationMatrix2D(new_m, rot_ang,1)
    # R = np.matrix([M[0][0:2], M[1][0:2]])
    # T = np.matrix([M[0][2],M[1][2]])
    # nose_rt = np.asarray(np.add(np.squeeze(np.asarray(np.dot(R, np.asmatrix(new_nose).T))),T).astype('int'))[0]
    # minif_rot = cv2.warpAffine(new_frame, M, (diag,diag),flags = cv2.INTER_NEAREST)
    # nose_pixels = [int(nose_rt[0]),int(nose_rt[1])]
    # cols,rows = minif_rot.shape
    # M1 = np.float32([[1,0,-nose_pixels[0]+cols/2],[0,1,-nose_pixels[1]]])
    # minif_rot = cv2.warpAffine(minif_rot,M1,(cols,rows))
    # # Crop the image in 32x32 frame around the nose
    #
    new_frame[new_frame == 255] = 0

    return new_frame, new_nose, new_m

def reaper(videoPath, frameIndices):
    # print 'segment number ', i
    print 'reaping', videoPath
    df, numSegment = loadFile(videoPath, 'segmentation', time=0)
    boundingboxes, miniframes, contours, bkgSamples, goodFrameIndices, segmentIndices, permutations, areas, pixels = getMFandC(videoPath,frameIndices)

    segmentIndices = frameIndices.loc[frameIndices.loc[:,'segment']==int(numSegment)]
    segmentIndices = segmentIndices.index.tolist()

    """ Visualise """
    miniframes = np.asarray(miniframes)
    pixels = np.asarray(pixels)
    AllNewMiniframes = pd.DataFrame(index = segmentIndices, columns= ['images', 'noses', 'middleP'])

    counter = 0
    while counter < len(miniframes):
        # print counter
        newMiniframes = []
        newNoses = []
        newMiddleP = []
        bbs = boundingboxes[counter]
        minif = miniframes[counter]
        cnts = contours[counter]
        pxs = pixels[counter]
        bkgSamps = bkgSamples[counter]
        for j, miniframe in enumerate(minif):
            # print '----------------', j, counter, videoPath
            # print miniframe
            ### Uncomment to plot
            # cv2.imshow('frame', miniframe)
            # cv2.waitKey()
            new_frame, new_nose, new_m = getMiniframes(pxs[j], miniframe,cnts[j],bbs[j],bkgSamps[j],counter = None, path = videoPath)
            # portrait = getPortrait(miniframe,cnts[j],bbs[j],bkgSamps[j])

            # get all the heads in a single list
            newMiniframes.append(new_frame)
            newNoses.append(new_nose)
            newMiddleP.append(new_m)

        ### UNCOMMENT TO PLOT ##################################################
        #     cv2.circle(new_frame,new_nose,2,255)
        #     cv2.circle(new_frame,new_m,2,255)
        #     cv2.imshow(str(j),new_frame)
        #
        #
        # k = cv2.waitKey(100) & 0xFF
        # if k == 27: #pres esc to quit
        #     break
        ########################################################################

        AllNewMiniframes.set_value(segmentIndices[counter], 'images', np.asarray(newMiniframes))
        AllNewMiniframes.set_value(segmentIndices[counter], 'noses', newNoses)
        AllNewMiniframes.set_value(segmentIndices[counter], 'middleP', newMiddleP)
      #   AllNewMiniframes.set_value(segmentIndices, 'areas', areas)
        # print counter
        counter += 1
    print 'you just reaped', videoPath
    return AllNewMiniframes

def modelDiffArea(fragments,areas):
    """
    fragment: fragment where to stract the areas to cumpute the mean and std of the diffArea
    areas: areas of all the blobs of the video
    """
    goodFrames = flatten([list(range(fragment[0],fragment[1])) for fragment in fragments])
    individualAreas = np.asarray(flatten(areas[goodFrames].tolist()))
    meanArea = np.mean(individualAreas)
    stdArea = np.std(individualAreas)
    return meanArea, stdArea

def newMiniframesBuilder(videoPaths):
    frameIndices = loadFile(videoPaths[0], 'frameIndices', time=0)

    num_cores = multiprocessing.cpu_count()
   #  num_cores = 1
    allNewMiniframes = Parallel(n_jobs=num_cores)(delayed(reaper)(videoPath,frameIndices) for videoPath in videoPaths)
    allNewMiniframes = pd.concat(allNewMiniframes)
    allNewMiniframes = allNewMiniframes.sort_index(axis=0,ascending=True)
    videoPath = videoPaths[0]
    # folder = os.videoPath.dirname(videoPath)
    # video = os.videoPath.basename(videoPath)
    # filename, extension = os.videoPath.splitext(video)
    # filename = filename.split('_')[0]
    # allNewMiniframes.to_pickle(folder +'/'+ filename + '_portraits' + '.pkl')

    saveFile(videoPath, allNewMiniframes, 'newMiniframes', time = 0)


   #  fragments = loadFile(videoPath, 'fragments', time=0)
   #  fragments = np.asarray(fragments)
   #  meanArea, stdArea = modelDiffArea(fragments, allPortraits.areas)
   #  videoInfo = loadFile(videoPath, 'videoInfo', time = 0)
   #  videoInfo = videoInfo.to_dict()[0]
   #  videoInfo['meanIndivArea'] = meanArea
   #  videoInfo['stdIndivArea'] = stdArea
   #  saveFile(videoPath,videoInfo,'videoInfo',time=0)
# 
# videoPaths = scanFolder('../Conflict8/conflict3and4_20120316T155032_1.avi')
# frameIndices = loadFile(videoPaths[0], 'frameIndices', time=0)
# # AllNewMiniframes = reaper(videoPaths[0], frameIndices)
# newMiniframesBuilder(videoPaths)
