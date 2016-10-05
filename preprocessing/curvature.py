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

def full2miniframe(cnt, boundingBox):
    """
    Push a point in the fullframe to miniframe coordinate system.
    Here it is use for centroids
    """
    cnt = np.squeeze(cnt)
    return np.asarray(cnt) - np.asarray([boundingBox[0][0],boundingBox[0][1]])

def rotate(p, deg):
    # deg = deg - 90
    theta = deg/180 * np.pi
    p1 = []
    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)
    if abs(cosTheta) < 0.0001:
        cosTheta = 0.
    if abs(sinTheta) < 0.0001:
        sinTheta = 0.

    p1.append( np.round(p[0]*cosTheta - p[1]*sinTheta).astype('int'))
    p1.append(np.round(p[0]*sinTheta + p[1]*cosTheta).astype('int'))
    return tuple(p1)

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

def subsampleCirc(r,n=20):
    return [(np.cos(2*np.pi/n*x)*r,np.sin(2*np.pi/n*x)*r) for x in xrange(0,n+1)]

def subsampleEllipse(a,b,n=20):
    return [(np.cos((2*np.pi)/n*x+np.pi/7)*a,np.sin((2*np.pi)/n*x)*b) for x in xrange(0,n+1)]

def subsampleParabola(a,b,n=20):
    return [(x+100,a*x**2+b) for x in xrange(-n-1,n+1)]

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
#
# def curvature_smoother(curvature):
#     """
#     smooth contour by convolution
#     """
#     # curvature = np.append(curvature,curvature[:5])
#     G = cv2.transpose(cv2.getGaussianKernel(16, 64, cv2.CV_64FC1))
#
#     curvature_smooth = np.convolve(curvature,G[0],mode='same')
#     # curvature_smooth = curvature_smooth[2:-3]
#     return curvature_smooth

def fillSquareFrame(square_frame,bkgSamps):
    bkgSamps = bkgSamps[bkgSamps > 150]
    numSamples = len(bkgSamps)

    numSamplesRequested = np.sum(square_frame == 0)
    indicesSamples = np.random.randint(0,numSamples,numSamplesRequested)
    square_frame[square_frame == 0] = bkgSamps[indicesSamples]
    return square_frame

# paths = scanFolder('../Conflict8/conflict3and4_20120316T155032_10.avi')
path = '../Conflict8/_20161003155023/segmentation/conflict3and4_10.pkl'
# df, numSegment = loadFile(paths[0], 'segmentation', time=0)
df = pd.read_pickle(path)
# print path


numFrames = 14
for j in range(10,numFrames):
    for numFish in range(8):
        print '**************************************************'
        # numFish = fishIndex[i]
        # numFish = 5
        contour = df.loc[j,'contours'][numFish]
        minif = df.loc[j, 'miniFrames'][numFish]
        bb = df.loc[j, 'boundingBoxes'][numFish]
        bkgSamp = df.loc[j, 'bkgSamples'][numFish]
        height, width = minif.shape

        plt.ion()
        frame = np.zeros_like((500,500)).astype('uint8')
        frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
        contour = full2miniframe(contour, bb)
        contour = smooth_resample(contour,smooth=True)
        contour = np.vstack([contour,contour])

        curvature = [curv(contour,i,2) for i in range(len(contour))]
        curvature = np.asarray(curvature)
        window = 101
        curvature = smooth(curvature, window)
        index = (window-1)/2
        curvature = curvature[index:-index]

        zero = np.argmin(np.abs(curvature[:len(curvature)/2]))
        i1 = zero
        i2 = zero+len(contour)/2
        contour = contour[i1:i2]
        curvature = curvature[i1:i2]

        max_locations, min_locations = get_extrema(curvature)
        sorted_locations = max_locations[np.argsort(curvature[max_locations])][::-1]
        maxCoord = [contour[max_loc,:] for max_loc in sorted_locations]
        nose = maxCoord[1]
        cntL = contour.shape[0]
        steps = 50
        support = np.asarray([contour[(sorted_locations[1] + steps) % cntL,:], contour[(sorted_locations[1] - steps) % cntL,:]])
        m = getMiddle(support)

        rot_ang = -(-np.arctan2(nose[1]-m[1],nose[0]-m[0])-np.pi/2)*180/np.pi
        # print 'Angle ,', rot_ang*180/np.pi

        rowsMin, colsMin = minif.shape
        diag = np.round(np.sqrt(rowsMin**2 + colsMin**2)).astype('int')
        new_frame = np.zeros((diag,diag)).astype('uint8')

        x_offset = np.ceil((diag-colsMin)/2).astype('int')
        y_offset = np.ceil((diag-rowsMin)/2).astype('int')

        new_frame[y_offset:y_offset + rowsMin, x_offset:x_offset+colsMin] = minif
        new_nose = tuple(np.asarray([nose[0]+x_offset, nose[1]+y_offset]).astype('int'))
        new_m = tuple(np.asarray([m[0]+x_offset, m[1]+y_offset]).astype('int'))

        M = cv2.getRotationMatrix2D(new_m, rot_ang,1)
        R = np.matrix([M[0][0:2], M[1][0:2]])
        T = np.matrix([M[0][2],M[1][2]])
        nose_rt = np.asarray(np.add(np.squeeze(np.asarray(np.dot(R, np.asmatrix(new_nose).T))),T).astype('int'))[0]



        minif_rot = cv2.warpAffine(new_frame, M, (diag,diag),flags = cv2.INTER_NEAREST)
        cv2.circle(minif_rot, tuple(nose_rt), 2, (255,0,0),1)

        nose_pixels = [int(nose_rt[0]),int(nose_rt[1])]
        print nose_pixels
        cv2.imshow('rot', minif_rot)
        cv2.waitKey(1000)
        print minif_rot.shape
        minif_cropped = minif_rot[nose_pixels[1]-7:nose_pixels[1]+25,nose_pixels[0]-16:nose_pixels[0]+16]
        print minif_cropped.shape

        portrait = fillSquareFrame(minif_cropped,bkgSamp)
        cv2.imshow('rot_crop', portrait)
        cv2.waitKey(1000)
        # print curvature
    #     X = np.squeeze(contour)[:,0]
    #     Y = np.squeeze(contour)[:,1]
    #     plt.figure()
    #     plt.subplot(1,2,1)
    #     plt.imshow(minif,interpolation='none',cmap='gray')
    #     plt.plot(X,Y,'r')
    #     for i,maxCoor in enumerate(maxCoord[:2]):
    #         plt.scatter(maxCoor[0],maxCoor[1],s=100,c='b')
    #         plt.scatter(support[i][0],support[i][1],s=100,c='r')
    #
    #     plt.scatter(m[0],m[1],s=100,c='g')
    #     plt.plot([m[0],nose[0]],[m[1],nose[1]],'r')
    #     plt.plot([0,minif.shape[1]],[m[1],m[1]])
    #     plt.plot([m[0],m[0]], [0,minif.shape[0]])
    #     plt.scatter(nose[0],nose[1],s=100,c='g')
    #     plt.axis('equal')
    #
    #     plt.subplot(1,2,2)
    #     plt.plot(curvature)
    #     plt.scatter(sorted_locations[:2],curvature[sorted_locations[:2]],s=50,c='b')
    #     # plt.axvline(zero)
    #     # plt.axvline(zero+len(contour)/2)
    # plt.show()
# counter += 1
