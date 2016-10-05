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
import time
import re
from joblib import Parallel, delayed
import multiprocessing
import pylab as pl
import math

def full2miniframe(cnt, boundingBox):
    """
    Push a point in the fullframe to miniframe coordinate system.
    Here it is use for centroids
    """
    cnt = np.squeeze(cnt)
    return np.asarray(cnt) - np.asarray([boundingBox[0][0],boundingBox[0][1]])

def getMFandC(path):
    """
    path: path to dataframe
    generate a list of arrays containing miniframes and centroids detected in
    path at this point we can already discard miniframes that does not belong
    to a specific fragments
    """
    #load dataframe
    print path
    df = pd.read_pickle(path)

    miniframes = np.asarray(df.loc[:, 'miniFrames'])
    centroids = np.asarray(df.loc[:, 'centroids'])
    pixels = np.asarray(df.loc[:, 'pixels'])
    contours = np.asarray(df.loc[:, 'contours'])
    boundingBoxes = np.asarray(df.loc[:, 'boundingBoxes'])
    return miniframes.tolist(),centroids.tolist(),pixels.tolist(), contours.tolist(), boundingBoxes.tolist()

def smoother(contour):
    """
    smooth contour by convolution
    """

    X = contour[:,0]
    print '1, ', X.shape
    X = np.append(X,X[:5])
    print '2, ', X.shape
    Y = contour[:,1]
    Y = np.append(Y,Y[:5])

    bkg = np.ones((np.max(Y),np.max(X)))

    G = cv2.transpose(cv2.getGaussianKernel(5, 16, cv2.CV_64FC1))

    X_smooth = np.convolve(X,G[0],mode='same')
    Y_smooth = np.convolve(Y,G[0],mode='same')
    print '3, ', X_smooth.shape
    X_smooth = X_smooth[2:-3]
    Y_smooth = Y_smooth[2:-3]
    print '4, ', X_smooth.shape
    return X_smooth, Y_smooth

def smooth_resample(contour,smooth = False):
    if smooth:
        x,y  = smoother(contour)
    else:
        x = contour[:,0]
        y = contour[:,1]

    x_new = np.append(x,x[:5])
    print '5, ', x_new.shape
    x_new = np.append(x[-5:],x_new)
    print '6, ', x_new.shape
    y_new = np.append(y,y[:5])
    y_new = np.append(y[-5:],y_new)


    M = 1000
    t = np.linspace(0, len(x_new), M)
    x = np.interp(t, np.arange(len(x_new)), x_new)
    print '7, ', x.shape
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
    print '8, ', xn.shape
    yn = y[idx]
    sContour = [[x,y] for (x,y) in zip(xn,yn)]

    return sContour


from scipy.interpolate import UnivariateSpline
import numpy as np

def curvature_splines(x, y=None, error=0.1):
    """Calculate the signed curvature of a 2D curve at each point
    using interpolating splines.
    Parameters
    ----------
    x,y: numpy.array(dtype=float) shape (n_points, )
         or
         y=None and
         x is a numpy.array(dtype=complex) shape (n_points, )
         In the second case the curve is represented as a np.array
         of complex numbers.
    error : float
        The admisible error when interpolating the splines
    Returns
    -------
    curvature: numpy.array shape (n_points, )
    Note: This is 2-3x slower (1.8 ms for 2000 points) than `curvature_gradient`
    but more accurate, especially at the borders.
    """

    # handle list of complex case
    if y is None:
        x, y = x.real, x.imag
    # print x, y
    t = np.arange(x.shape[0])
    std = error * np.ones_like(x)

    fx = UnivariateSpline(t, x, k=5, w=1 / np.sqrt(std))
    fy = UnivariateSpline(t, y, k=5, w=1 / np.sqrt(std))

    x1 = fx.derivative(1)(t)
    x2 = fx.derivative(2)(t)
    y1 = fy.derivative(1)(t)
    y2 = fy.derivative(2)(t)
    curvature = (x1* y2 - y1* x2) / np.power(x1** 2 + y1** 2, 3 / 2)
    return curvature

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

    return np.asarray(max_locations)-1, np.asarray(min_locations)-1

def getMiddle(points):
    l = points.shape[0]
    b = np.sum(points, axis=0)
    b = np.divide(b,l)
    return b

def getCurvature(contour, n=1, thresholded=False):
    contour = np.squeeze(contour)
    # print contour
    #
    # print contour[0,:]
    # print contour[0,:].shape
    # contour = np.append(contour, contour[0,:], axis=0)
    # print contour
    sContour = smooth_resample(contour,smooth = True)

    N = len(sContour)
    sContour = np.asarray(sContour)
    curvature = curvature_splines(sContour[:,0], y=sContour[:,1], error=0.1)
    # curvature = curvature_splines(contour[:,0], y=contour[:,1], error=0.001)
    print '9, ', curvature.shape

    sContour = np.asarray(sContour)
    firstPoint = sContour[0]
    # d = np.linalg.norm(firstPoint,sContour[1:])
    ds = [np.linalg.norm(firstPoint-sCnt) for sCnt in sContour[5:]]
    index = np.argmin(ds) + 6
    curvature = curvature[:index]
    curvature = np.insert(curvature, 0, 0)
    curvature = np.append(curvature, 0)
    sContour = sContour[:index]

    max_locations, min_locations = get_extrema(curvature)
    sorted_locations = max_locations[np.argsort(curvature[max_locations])][::-1]
    maxCoord = [sContour[max_loc,:] for max_loc in sorted_locations]

    cntL = sContour.shape[0]
    steps = 15
    support = np.asarray([sContour[(sorted_locations[1] + steps) % cntL,:], sContour[(sorted_locations[1] - steps) % cntL,:]])
    m = getMiddle(support)

    return curvature[1:], sContour, maxCoord, sorted_locations, support,m

def filterContoursBySize(contours,minArea,maxArea):
    goodContours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > minArea and area < maxArea:
            goodContours.append(contour)
    return goodContours

if __name__ == '__main__':


    path = '../Conflict8/_20161003155023/segmentation/conflict3and4_10.pkl'
    # df, numSegment = loadFile(paths[0], 'segmentation', time=0)
    df = pd.read_pickle(path)
    # print path


    numFrames = 20
    np.random.seed = 0
    fishIndex = np.random.randint(0,7,numFrames)
    for i in range(numFrames):
        numFish = fishIndex[i]
        contour = df.loc[i,'contours'][numFish]
        I = df.loc[i, 'miniFrames'][numFish]
        bb = df.loc[i, 'boundingBoxes'][numFish]
        plt.ion()
        # contour = np.squeeze(contour)
        # frame = np.zeros_like(minif)
        frame = np.zeros_like((500,500)).astype('uint8')
        frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
        cnt = full2miniframe(contour, bb)
        # print contour
        # cnt = smooth_resample(contour,smooth=True)

    # path = '../Conflict8/_20161003155023/segmentation/conflict3and4_5.pkl'
    #
    # miniframes, centroids, pixels, contours, boundingBoxes = getMFandC(path)
    # miniframes = np.asarray(miniframes)
    # """ Visualise """
    #
    # counter = 0
    # fish_number = 5
    # plt.ion()
    # while counter < 20:
    #     minif = miniframes[counter]
    #
    #     cnts = contours[counter]
    #
    #     bbs = boundingBoxes[counter]
    #     print len(minif)
    #     I = minif[fish_number]
    #     bb = bbs[fish_number]
    #
    #     ret, frame = cv2.threshold(I,0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     contours1, hierarchy = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #     goodContours = filterContoursBySize(contours1,250, 750)
    #
    #     # # goodContours = contours1
    #     # print '------------------------'
    #     # print contours1
    #     # print goodContours
    #     # print '------------------------'
    #
    #     cnt = cnts[fish_number]
    #     cnt = full2miniframe(cnt, bb)
    #     cnt = np.squeeze(cnt)
    #     print 'contour, ',cnt
    #     print 'contour shape ', cnt.shape
        # curvature, sContour, maxCoord, max_locations = getCurvature(cnt)
        # print goodContours[0].shape
        plt.figure()


        plt.subplot(1,2,2)
        plt.imshow(I,interpolation='none',cmap='gray')
        if cnt.shape[0] > 2:
            plt.plot(cnt[:,0], cnt[:,1],'b')
        curvature, sContour, maxCoord, max_locations, support, m = getCurvature(cnt)
        # curvature, sContour, maxCoord, max_locations, support, m = getCurvature(cnt)
        # print sContour.shape
        # print maxCoord
        # print max_locations

        X = np.squeeze(sContour)[:,0]
        Y = np.squeeze(sContour)[:,1]

        plt.plot(sContour[:,0],sContour[:,1],'r')

        for i,maxCoor in enumerate(maxCoord[:2]):
            plt.scatter(maxCoor[0],maxCoor[1],s=100,c='b')
            plt.scatter(support[i][0],support[i][1],s=100,c='r')

        plt.scatter(m[0],m[1],s=100,c='g')
        plt.plot([m[0],maxCoord[1][0]],[m[1],maxCoord[1][1]],'r')
        plt.plot([0,I.shape[1]],[m[1],m[1]])
        plt.plot([m[0],m[0]], [0,I.shape[0]])
        # plt.scatter(maxCoord[1][0],maxCoord[1][1],s=100,c='g')
        plt.axis('equal')
        plt.subplot(1,2,1)
        plt.plot(curvature)
        plt.scatter(max_locations,curvature[max_locations])
        plt.show()
        # counter += 1
