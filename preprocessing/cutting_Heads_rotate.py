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


def miniframeThresholder(miniframes, minThreshold, maxThreshold,EQ=False):
    """
    Applies a second threshold to the miniframes obtained in segmentation_parallel
    by Otsu transform
    """
    thMiniframes = []
    miniEllipses = []
    for miniframe in miniframes:
        if EQ:
            # Equalize image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))
            frame = clahe.apply(miniframe)
        ret, thMiniframe = cv2.threshold(miniframe,minThreshold,maxThreshold, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thMiniframe,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        miniEllipses.append(fitEllipse(contours))
        # goodContours = filterContoursBySize(contours,minArea, maxArea)

        thMiniframes.append(thMiniframe)
    return thMiniframes, miniEllipses

def fitEllipse(contours):
    maxCnt = np.argmax([len(cnt) for cnt in contours])
    cnt = contours[maxCnt]
    return cv2.fitEllipse(cnt)

def full2miniframe(point, boundingBox):
    """
    Push a point in the fullframe to miniframe coordinate system.
    Here it is use for centroids
    """
    return tuple(np.asarray(point) - np.asarray([boundingBox[0],boundingBox[1]]))

def getFoci(ellipse, centroid):
    def computeFoci(a,b):
        # if a < b:
        #     temp = a
        #     a = b
        #     b = temp

        c = np.divide(np.sqrt(a**2 - b**2),2).astype('int')
        return [(c,0), (-c,0)]

    def rotate(p, deg):
        deg = deg - 90
        theta = (deg/180.) * np.pi
        p1 = []
        cosTheta = np.cos(theta)
        sinTheta = np.sin(theta)
        if abs(cosTheta) < 0.0001:
            cosTheta = 0.
        if abs(sinTheta) < 0.0001:
            sinTheta = 0.
        p1.append( np.round(p[0]*np.cos(theta) - p[1]*np.sin(theta)).astype('int'))
        p1.append(np.round(p[0]*np.sin(theta) + p[1]*np.cos(theta)).astype('int'))
        return tuple(p1)

    def translate(p, h,k):
        return (p[0]+h , p[1]+k)

    def rototrans(p,deg,h,k):
        p_rot = rotate(p,deg)
        return translate(p_rot, h,k)


    def minDistance(p1, points):
        distances = []
        p1 = np.asarray(p1)
        for p in points:
            p = np.asarray(p)
            distances.append(np.linalg.norm(p1 - p))
        return np.argmax(distances)
    # centre
    cx = ellipse[0][0]
    cy = ellipse[0][1]
    # semi-axis (ordered by default using fitEllipse in OpenCV)
    a = ellipse[1][1] # major
    b = ellipse[1][0] # minor
    # rotation angle wrt y=0
    angle = ellipse[2]

    foci = computeFoci(a,b)
    EllipsesFoci = []
    for focus in foci:
        focus = rototrans(focus, angle, cx,cy)
        EllipsesFoci.append(focus)

    Focus = EllipsesFoci[minDistance(centroid, EllipsesFoci)]
    Focus =  tuple(np.round(Focus).astype('int'))
    return Focus




def getMFandC(path):
    """
    path: path to dataframe
    generate a list of arrays containing miniframes and centroids detected in
    path at this point we can already discard miniframes that does not belong
    to a specific fragments
    """
    #load dataframe
    df = pd.read_pickle(path)
    # print df
    # check if permutations are NaN (i.e. the frame is not included in a fragment)
    permutations = np.asarray(df['permutation'].notnull())
    #generate a lists of "admissible" miniframes and centroids
    boundingBoxes = np.asarray(df.loc[:, 'boundingBoxes'])
    # print boundigBoxes
    miniframes = np.asarray(df.loc[:, 'miniFrames'])
    centroids = np.asarray(df.loc[:, 'centroids'])

    boundingBoxes = boundingBoxes[np.where(permutations==True)[0]]
    miniframes = miniframes[np.where(permutations==True)[0]]
    centroids = centroids[np.where(permutations==True)[0]]

    return boundingBoxes.tolist(), miniframes.tolist(),centroids.tolist()

if __name__ == '__main__':
    paths = scanFolder('./Cafeina5peces/Caffeine5fish_20140206T122428_1.pkl')

    path = paths[1]
    boundingboxes, miniframes, centroids = getMFandC(path)
    miniframes = np.asarray(miniframes)
    """ Visualise """

    counter = 0
    while counter < len(miniframes):
            bbs = boundingboxes[counter]
            minif = miniframes[counter]
            cents = centroids[counter]
            thMinif, ellipses = miniframeThresholder(minif, 115, 255,True)
            # # create miniHeads' frames
            # miniHeads = np.zeros((40,40))
            for j, miniframe in enumerate(thMinif):
                cent = full2miniframe(cents[j], bbs[j])

                focus = getFoci(ellipses[j],cent)
                ### show points and line on miniframe
                # cv2.ellipse(minif[j],ellipses[j],(255,0,0),1)
                # cv2.circle(minif[j], cent,2,255,1)
                # cv2.circle(minif[j], focus,2,255,1)
                # cv2.line(minif[j], cent, focus, 255,1)
                ###
                rows, cols = minif[j].shape
                diag = np.round(np.sqrt(rows**2 + cols**2)).astype('int')
                new_frame = np.zeros((diag,diag))
                M = cv2.getRotationMatrix2D(cent, ellipses[j][2],1)
                print 'rotation matrix',M
                dst = cv2.warpAffine(minif[j], M, cent)
                cv2.imshow('thresholdedMiniframes'+str(j),dst)
                # cv2.imshow('thresholdedMiniframe'+str(j),minif[j])

            # cv2.imshow('frame',frame)
            k = cv2.waitKey(100) & 0xFF
            if k == 27: #pres esc to quit
                break
            counter += 1
