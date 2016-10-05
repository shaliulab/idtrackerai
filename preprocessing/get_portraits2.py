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
   #print(len(s))
   if window == 'flat': #moving average
       w=np.ones(window_len,'d')
   else:
       w=eval('np.'+window+'(window_len)')

   y=np.convolve(w/w.sum(),s,mode='valid')
   return y

def smoother(contour):
    """
    smooth contour by convolution
    """
    X = contour[:,0]
    X = np.append(X,X[:5])
    Y = contour[:,1]
    Y = np.append(Y,Y[:5])

    bkg = np.ones((np.max(Y),np.max(X)))

    G = cv2.transpose(cv2.getGaussianKernel(5, 16, cv2.CV_64FC1))

    X_smooth = np.convolve(X,G[0],mode='same')
    Y_smooth = np.convolve(Y,G[0],mode='same')
    X_smooth = X_smooth[2:-3]
    Y_smooth = Y_smooth[2:-3]
    return X_smooth, Y_smooth

def smooth_resample(contour,smooth = False):
    if smooth:
        x,y  = smoother(contour)
    else:
        x = contour[:,0]
        y = contour[:,1]

    x_new = x
    y_new = y

    M = 1000 # FIXME The number of points for the resampling has to be consistent with the resolution for the artifact of the arctan2 not to appear
    t = np.linspace(0, len(x_new), M)
    x = np.interp(t, np.arange(len(x_new)), x_new)
    y = np.interp(t, np.arange(len(y_new)), y_new)
    tol = .1
    i, idx = 0, [0]
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


def curv(cnt,i,n):
    left = (i + n) % (len(cnt)-1)
    right = (i - n) % (len(cnt)-1)
    phi_l = phi(left, cnt) #% 360
    phi_r = phi(right, cnt) #% 360

    if cnt[left+1][0] <= cnt[right+1][0]:
        if phi_l <= 0:
            phi_l += 2*np.pi
        if phi_r <= 0:
            phi_r += 2*np.pi

    return (phi_l - phi_r) /(2*n+1)

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

def getMFandC(path, frameIndices):
    """
    path: path to dataframe
    generate a list of arrays containing miniframes and centroids detected in
    path at this point we can already discard miniframes that does not belong
    to a specific fragments
    """
    # get number of segment
    # video = os.path.basename(path)
    # filename, extension = os.path.splitext(video)
    # numSegment = int(filename.split('_')[-1])
    #load dataframe
    # df = pd.read_pickle(path)
    df, numSegment = loadFile(path, 'segmentation', time=0)
    # print df
    # check if permutations are NaN (i.e. the frame is not included in a fragment)
    permutationsBool = np.asarray(df['permutation'].notnull())

    #generate a lists of "admissible" miniframes and centroids
    permutations = np.asarray(df['permutation'])
    boundingBoxes = np.asarray(df.loc[:, 'boundingBoxes'])
    miniframes = np.asarray(df.loc[:, 'miniFrames'])
    contours = np.asarray(df.loc[:, 'contours'])
    bkgSamples = np.asarray(df.loc[:,'bkgSamples'])

    goodIndices = np.where(permutationsBool==True)[0]
    goodFrameIndices, segmentIndices = getEncompassingIndices(frameIndices, int(numSegment), goodIndices)
    goodFrameIndices = segmentIndices
    # boundingBoxes = boundingBoxes[goodIndices]
    # miniframes = miniframes[goodIndices]
    # centroids = centroids[goodIndices]
    # bkgSamples = bkgSamples[goodIndices]
    # permutations = permutations[goodIndices]
    # #
    return boundingBoxes.tolist(), miniframes.tolist(), contours.tolist(), bkgSamples.tolist(), goodFrameIndices, segmentIndices, permutations.tolist()

def fillSquareFrame(square_frame,bkgSamps):
    bkgSamps = bkgSamps[bkgSamps > 150]
    numSamples = len(bkgSamps)

    numSamplesRequested = np.sum(square_frame == 0)
    indicesSamples = np.random.randint(0,numSamples,numSamplesRequested)
    square_frame[square_frame == 0] = bkgSamps[indicesSamples]
    return square_frame

def getPortrait(miniframe,cnt,bb,bkgSamp):
    height, width = miniframe.shape

    # Pass contour to bb coord, resample, smooth, and duplicate
    cnt = full2miniframe(cnt, bb)
    cnt = np.asarray(cnt)
    cnt = np.squeeze(cnt)
    cnt = smooth_resample(cnt,smooth=True)
    cnt = np.vstack([cnt,cnt])

    # Compute curvature
    curvature = [curv(cnt,i,2) for i in range(len(cnt))]
    curvature = np.asarray(curvature)

    # Smooth curvature
    window = 101
    curvature = smooth(curvature, window)
    index = (window-1)/2
    curvature = curvature[index:-index]

    # Crop contour and curvature from the first zero curvature point
    zero = np.argmin(np.abs(curvature[:len(curvature)/2]))
    i1 = zero
    i2 = zero+len(cnt)/2
    cnt = cnt[i1:i2]
    curvature = curvature[i1:i2]

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
    diag = np.round(np.sqrt(rowsMin**2 + colsMin**2)).astype('int')
    new_frame = np.zeros((diag,diag)).astype('uint8')
    x_offset = np.ceil((diag-colsMin)/2).astype('int')
    y_offset = np.ceil((diag-rowsMin)/2).astype('int')
    new_frame[y_offset:y_offset + rowsMin, x_offset:x_offset+colsMin] = miniframe

    # Translate and rotate nose and middle point to the new frame
    new_nose = tuple(np.asarray([nose[0]+x_offset, nose[1]+y_offset]).astype('int'))
    new_m = tuple(np.asarray([m[0]+x_offset, m[1]+y_offset]).astype('int'))

    # Get roto-translation matrix and rotate nose and miniframe
    M = cv2.getRotationMatrix2D(new_m, rot_ang,1)
    R = np.matrix([M[0][0:2], M[1][0:2]])
    T = np.matrix([M[0][2],M[1][2]])
    nose_rt = np.asarray(np.add(np.squeeze(np.asarray(np.dot(R, np.asmatrix(new_nose).T))),T).astype('int'))[0]
    minif_rot = cv2.warpAffine(new_frame, M, (diag,diag),flags = cv2.INTER_NEAREST)

    # Crop the image in 32x32 frame around the nose
    nose_pixels = [int(nose_rt[0]),int(nose_rt[1])]
    minif_cropped = minif_rot[nose_pixels[1]-7:nose_pixels[1]+25,nose_pixels[0]-16:nose_pixels[0]+16]

    # Fill black parts of the portrait with random background
    portrait = fillSquareFrame(minif_cropped,bkgSamp)
    return portrait

def reaper(path, frameIndices):
    # print 'segment number ', i
    print 'reaping', path
    boundingboxes, miniframes, contours, bkgSamples, goodFrameIndices, segmentIndices, permutations = getMFandC(path,frameIndices)
    miniframes = np.asarray(miniframes)
    """ Visualise """
    AllPortraits = pd.DataFrame(index = segmentIndices, columns= ['images', 'permutations'])
    # print goodFrameIndices
    counter = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    while counter < len(miniframes):
        portraits = []
        bbs = boundingboxes[counter]
        minif = miniframes[counter]
        cnts = contours[counter]
        bkgSamps = bkgSamples[counter]
        for j, miniframe in enumerate(minif):
            # print '----------------', j, counter, path
            portrait = getPortrait(miniframe,cnts[j],bbs[j],bkgSamps[j])

            # get all the heads in a single list
            portraits.append(portrait)

        ### UNCOMMENT TO PLOT ##################################################
        #     cv2.imshow(str(j),portrait)
        #
        # k = cv2.waitKey(1) & 0xFF
        # if k == 27: #pres esc to quit
        #     break
        ########################################################################

        AllPortraits.set_value(goodFrameIndices[counter], 'images', np.asarray(portraits))
        AllPortraits.set_value(goodFrameIndices[counter], 'permutations', permutations[counter])
        counter += 1
    return AllPortraits

if __name__ == '__main__':
    # frameIndices = pd.read_pickle('../Conflict8/conflict3and4_frameIndices.pkl')
    # frameIndices = pd.read_pickle('../Cafeina5peces/Caffeine5fish_frameIndices.pkl')
    # paths = scanFolder('../Cafeina5peces/Caffeine5fish_20140206T122428_1.avi')
    paths = scanFolder('../Conflict8/conflict3and4_20120316T155032_1.avi')
    frameIndices = loadFile(paths[0], 'frameIndices', time=0)

    num_cores = multiprocessing.cpu_count()

    num_cores = 1
    allPortraits = Parallel(n_jobs=num_cores)(delayed(reaper)(path,frameIndices) for path in paths)
    allPortraits = pd.concat(allPortraits)
    allPortraits = allPortraits.sort_index(axis=0,ascending=True)
    path = paths[0]
    # folder = os.path.dirname(path)
    # video = os.path.basename(path)
    # filename, extension = os.path.splitext(video)
    # filename = filename.split('_')[0]
    # allPortraits.to_pickle(folder +'/'+ filename + '_portraits' + '.pkl')

    saveFile(path, allPortraits, 'portraits', time = 0)
