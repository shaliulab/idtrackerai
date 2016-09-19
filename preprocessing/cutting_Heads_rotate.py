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


def miniframeThresholder(miniframes, minThreshold, maxThreshold,EQ=False):
    """
    Applies a second threshold to the miniframes obtained in segmentation_parallel
    by Otsu transform
    """
    thMiniframes = []
    miniEllipses = []
    for i, miniframe in enumerate(miniframes):
        # print i
        if EQ:
            # Equalize image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))
            frame = clahe.apply(miniframe)
        ret, thMiniframe = cv2.threshold(miniframe,minThreshold,maxThreshold, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thMiniframe,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            ret, thMiniframe = cv2.threshold(miniframe,minThreshold+20,maxThreshold, cv2.THRESH_BINARY_INV)
            contours, hierarchy = cv2.findContours(thMiniframe,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            ellipse = fitEllipse(contours)

        ellipse = fitEllipse(contours)

        if ellipse == None:
            ret, thMiniframe = cv2.threshold(miniframe,minThreshold+20,maxThreshold, cv2.THRESH_BINARY_INV)
            contours, hierarchy = cv2.findContours(thMiniframe,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            ellipse = fitEllipse(contours)
        miniEllipses.append(fitEllipse(contours))
        # goodContours = filterContoursBySize(contours,minArea, maxArea)

        thMiniframes.append(thMiniframe)
    return thMiniframes, miniEllipses

def fitEllipse(contours):
    maxCnt = np.argmax([len(cnt) for cnt in contours])
    cnt = contours[maxCnt]

    if len(cnt) < 5 :
        ellipse = None
    else:
        ellipse = cv2.fitEllipse(cnt)
        if ellipse[0][0] < 0 or ellipse[0][1] < 0:
            ellipse = None
    return ellipse

def full2miniframe(point, boundingBox):
    """
    Push a point in the fullframe to miniframe coordinate system.
    Here it is use for centroids
    """
    return tuple(np.asarray(point) - np.asarray([boundingBox[0][0],boundingBox[0][1]]))

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

def getFoci(ellipse, centroid):
    def computeFoci(a,b):
        # if a < b:
        #     temp = a
        #     a = b
        #     b = temp

        c = np.divide(np.sqrt(a**2 - b**2),2).astype('int')
        return [(c,0), (-c,0)]


    def extremaEllipse(a):
        return [(-a/2,0),(a/2,0)]


    def translate(p, h,k):
        return (p[0]+h , p[1]+k)

    def rototrans(p,deg,h,k):
        p_rot = rotate(p,deg)
        return translate(p_rot, h,k)


    def maxDistance(p1, points):
        distances = []
        p1 = np.asarray(p1)
        for p in points:
            p = np.asarray(p)
            distances.append(np.linalg.norm(p1 - p))
        return np.argmax(distances)

    def minDistance(p1, points):
        distances = []
        p1 = np.asarray(p1)
        for p in points:
            p = np.asarray(p)
            distances.append(np.linalg.norm(p1 - p))
        return np.argmin(distances)
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

    Focus = EllipsesFoci[maxDistance(centroid, EllipsesFoci)]

    exts = extremaEllipse(a)
    extrema = []
    for p in exts:
        rot_ext = rototrans(p, angle, cx,cy)
        extrema.append(rot_ext)
    Ext = extrema[minDistance(Focus, extrema)]

    Focus = tuple(np.round(Focus).astype('int'))
    Ext = tuple(np.round(Ext).astype('int'))
    return Focus, Ext

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
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    numSegment = int(filename.split('_')[-1])
    #load dataframe
    df = pd.read_pickle(path)
    # print df
    # check if permutations are NaN (i.e. the frame is not included in a fragment)
    permutationsBool = np.asarray(df['permutation'].notnull())
    #generate a lists of "admissible" miniframes and centroids
    permutations = np.asarray(df['permutation'])
    boundingBoxes = np.asarray(df.loc[:, 'boundingBoxes'])
    miniframes = np.asarray(df.loc[:, 'miniFrames'])
    centroids = np.asarray(df.loc[:, 'centroids'])
    bkgSamples = np.asarray(df.loc[:,'bkgSamples'])
    goodIndices = np.where(permutationsBool==True)[0]
    goodFrameIndices, segmentIndices = getEncompassingIndices(frameIndices, numSegment, goodIndices)
    boundingBoxes = boundingBoxes[goodIndices]
    miniframes = miniframes[goodIndices]
    centroids = centroids[goodIndices]
    bkgSamples = bkgSamples[goodIndices]

    permutations = permutations[goodIndices]
    return boundingBoxes.tolist(), miniframes.tolist(), centroids.tolist(), bkgSamples.tolist(), goodFrameIndices, segmentIndices, permutations.tolist()

def fillSquareFrame(square_frame,bkgSamps):
    bkgSamps = bkgSamps[bkgSamps > 150]
    numSamples = len(bkgSamps)

    numSamplesRequested = np.sum(square_frame == 0)
    indicesSamples = np.random.randint(0,numSamples,numSamplesRequested)
    square_frame[square_frame == 0] = bkgSamps[indicesSamples]
    return square_frame


def reaper(path, frameIndices):
    # print 'segment number ', i
    print 'reaping', path
    boundingboxes, miniframes, centroids, bkgSamples, goodFrameIndices, segmentIndices, permutations = getMFandC(path,frameIndices)
    miniframes = np.asarray(miniframes)
    """ Visualise """
    AllPortraits = pd.DataFrame(index = segmentIndices, columns= ['images', 'permutations'])

    counter = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    while counter < len(miniframes):
        portraits = []
        bbs = boundingboxes[counter]
        minif = miniframes[counter]
        cents = centroids[counter]
        bkgSamps = bkgSamples[counter]
        thMinif, ellipses = miniframeThresholder(minif, 110, 255,True) # 90 for conlfict
        for j, miniframe in enumerate(thMinif):
            cent = full2miniframe(cents[j], bbs[j])
            focus, ext = getFoci(ellipses[j],cent)
            ###rotation
            rowsMin, colsMin = minif[j].shape
            diag = np.round(np.sqrt(rowsMin**2 + colsMin**2)).astype('int')
            new_frame = np.zeros((diag,diag)).astype('uint8')

            x_offset = np.ceil((diag-colsMin)/2).astype('int')
            y_offset = np.ceil((diag-rowsMin)/2).astype('int')

            new_frame_centre = (diag/2, diag/2)
            new_frame[y_offset:y_offset + rowsMin, x_offset:x_offset+colsMin] = minif[j]
            new_cent = tuple(np.asarray([cent[0]+x_offset, cent[1]+y_offset]).astype('int'))
            new_ext = tuple(np.asarray([ext[0]+x_offset, ext[1]+y_offset]).astype('int'))

            rot_deg = ellipses[j][2]

            M = cv2.getRotationMatrix2D(new_frame_centre, rot_deg,1)
            R = np.matrix([M[0][0:2], M[1][0:2]])
            T = np.matrix([M[0][2],M[1][2]])
            new_frame = cv2.warpAffine(new_frame, M, (diag,diag),flags = cv2.INTER_NEAREST)

            ext_rt = np.asarray(np.add(np.squeeze(np.asarray(np.dot(R, np.asmatrix(new_ext).T))),T).astype('int'))[0]
            cent_rt = np.asarray(np.add(np.squeeze(np.asarray(np.dot(R, np.asmatrix(new_cent).T))),T).astype('int'))[0]

            if ext_rt[1] >= cent_rt[1]:
                M1 = cv2.getRotationMatrix2D(new_frame_centre, 180,1)
                R1 = np.matrix([M1[0][0:2], M1[1][0:2]])
                T1 = np.matrix([M1[0][2],M1[1][2]])
                new_frame = cv2.warpAffine(new_frame, M1, (diag,diag),flags = cv2.INTER_NEAREST)
                ext_rt = np.asarray(np.add(np.squeeze(np.asarray(np.dot(R1, np.asmatrix(ext_rt).T))),T1).astype('int'))[0]
                cent_rt = np.asarray(np.add(np.squeeze(np.asarray(np.dot(R1, np.asmatrix(cent_rt).T))),T1).astype('int'))[0]

            bigger_frame = np.zeros((diag,2*diag)).astype('uint8')
            bigger_frame[0:diag,diag/2:3*diag/2] = new_frame
            rect1 = (ext_rt[0]-16+diag/2, 0)
            rect2 = (ext_rt[0]+16+diag/2, 32)
            square_frame = bigger_frame[rect1[1]:rect2[1],rect1[0]:rect2[0]]

            square_frame_filled = fillSquareFrame(square_frame,bkgSamps[j])
            # get all the heads in a single list
            portraits.append(square_frame_filled)
            cv2.imshow('thresholdedMiniframes'+str(j),square_frame_filled)

        AllPortraits.set_value(goodFrameIndices[counter], 'images', np.asarray(portraits))
        AllPortraits.set_value(goodFrameIndices[counter], 'permutations', permutations[counter])

        k = cv2.waitKey(1) & 0xFF
        if k == 27: #pres esc to quit
            break
        counter += 1
    return AllPortraits

if __name__ == '__main__':
    # frameIndices = pd.read_pickle('../Conflict8/conflict3and4_frameIndices.pkl')
    frameIndices = pd.read_pickle('../Cafeina5peces/Caffeine5fish_frameIndices.pkl')
    # paths = scanFolder('../Conflict8/conflict3and4_20120316T155032_1.pkl')
    paths = scanFolder('../Cafeina5peces/Caffeine5fish_20140206T122428_1.pkl')

    num_cores = multiprocessing.cpu_count()

    num_cores = 1
    allPortraits = Parallel(n_jobs=num_cores)(delayed(reaper)(path,frameIndices) for path in paths)
    allPortraits = pd.concat(allPortraits)
    allPortraits = allPortraits.sort_index(axis=0,ascending=True)
    path = paths[0]
    folder = os.path.dirname(path)
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    filename = filename.split('_')[0]
    allPortraits.to_pickle(folder +'/'+ filename + '_portraits' + '.pkl')

    # for i, path in enumerate(paths):
    #     AllPortraits = reaper(path, frameIndices)
    #     print len(AllPortraits)
    #     print AllPortraits
    #     # print 'segment number ', i
    #     boundingboxes, miniframes, centroids, bkgSamples, goodFrameIndices, segmentIndices = getMFandC(path,frameIndices)
    #     miniframes = np.asarray(miniframes)
    #     """ Visualise """
    #     AllPortraits = pd.DataFrame(index = segmentIndices, columns= 'images')
    #     counter = 0
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     while counter < len(miniframes):
    #         portraits = []
    #         # print 'frame ', indices[counter]
    #         bbs = boundingboxes[counter]
    #         minif = miniframes[counter]
    #         cents = centroids[counter]
    #         bkgSamps = bkgSamples[counter]
    #         thMinif, ellipses = miniframeThresholder(minif, 110, 255,True)
    #         # # create miniHeads' frames
    #         # miniHeads = np.zeros((40,40))
    #         for j, miniframe in enumerate(thMinif):
    #             print 'segment, ', i, ', frame, ', goodFrameIndices[counter], ', blob, ', j
    #             cent = full2miniframe(cents[j], bbs[j])
    #             focus, ext = getFoci(ellipses[j],cent)
    #
    #             ### show points and line on miniframe
    #             cv2.ellipse(minif[j],ellipses[j],(255,0,0),1)
    #             h_minif, w_minif = minif[j].shape
    #             cv2.circle(minif[j], ext,2,255,1)
    #             cv2.circle(minif[j], cent,2,255,2)
    #             # # inter = intersectLines(cent,focus, w_minif, h_minif)
    #
    #             # cv2.line(minif[j], tuple(np.asarray(ellipses[j][0]).astype('int')), ext, 255,2)
    #
    #             ###
    #             cv2.imshow('thresholdedMiniframes1'+str(j),minif[j])
    #             # ###rotation
    #             rowsMin, colsMin = minif[j].shape
    #             diag = np.round(np.sqrt(rowsMin**2 + colsMin**2)).astype('int')
    #             new_frame = np.zeros((diag,diag)).astype('uint8')
    #             # print new_frame.shape
    #             # rows, cols = new_frame.shape
    #             x_offset = np.ceil((diag-colsMin)/2).astype('int')
    #             y_offset = np.ceil((diag-rowsMin)/2).astype('int')
    #             # print x_offset, y_offset
    #             # bb = bbs[j]
    #             # bb = ((bb[0][0]-x_offset,bb[0][1]-y_offset),(bb[1][0]+x_offset,bb[1][1]+y_offset))
    #             # new_frame = bkg[bb[0][0]:bb[1][0],bb[0][1]:bb[1][1]]
    #
    #             new_frame_centre = (diag/2, diag/2)
    #             new_frame[y_offset:y_offset + rowsMin, x_offset:x_offset+colsMin] = minif[j]
    #             new_cent = tuple(np.asarray([cent[0]+x_offset, cent[1]+y_offset]).astype('int'))
    #             new_ext = tuple(np.asarray([ext[0]+x_offset, ext[1]+y_offset]).astype('int'))
    #             # cv2.circle(new_frame, new_cent,2,255,2)
    #             # cv2.circle(new_frame, new_ext,2,0,2)
    #             cv2.imshow('a', new_frame)
    #             # temp = new_frame.copy()
    #             rot_deg = ellipses[j][2]
    #             # print rot_deg
    #
    #             M = cv2.getRotationMatrix2D(new_frame_centre, rot_deg,1)
    #             R = np.matrix([M[0][0:2], M[1][0:2]])
    #             T = np.matrix([M[0][2],M[1][2]])
    #             new_frame = cv2.warpAffine(new_frame, M, (diag,diag),flags = cv2.INTER_NEAREST)
    #
    #             # print new_frame.shape
    #
    #             # print 'extremum maximal axis,', new_ext
    #             # print 'centroid,', new_cent
    #             ext_rt = np.asarray(np.add(np.squeeze(np.asarray(np.dot(R, np.asmatrix(new_ext).T))),T).astype('int'))[0]
    #             cent_rt = np.asarray(np.add(np.squeeze(np.asarray(np.dot(R, np.asmatrix(new_cent).T))),T).astype('int'))[0]
    #             # print 'rotation matrix, ', R
    #             # cent_rt = np.add(np.dot(R, cent_rt), T.T )
    #             # print 'extremum maximal axis rotated, ', ext_rt
    #             # print 'centroid rotated, ', cent_rt
    #             if ext_rt[1] >= cent_rt[1]:
    #                 M1 = cv2.getRotationMatrix2D(new_frame_centre, 180,1)
    #                 R1 = np.matrix([M1[0][0:2], M1[1][0:2]])
    #                 T1 = np.matrix([M1[0][2],M1[1][2]])
    #                 new_frame = cv2.warpAffine(new_frame, M1, (diag,diag),flags = cv2.INTER_NEAREST)
    #                 # cent_rt = np.add(np.dot(R1, cent_rt), T1.T )
    #                 ext_rt = np.asarray(np.add(np.squeeze(np.asarray(np.dot(R1, np.asmatrix(ext_rt).T))),T1).astype('int'))[0]
    #                 cent_rt = np.asarray(np.add(np.squeeze(np.asarray(np.dot(R1, np.asmatrix(cent_rt).T))),T1).astype('int'))[0]
    #
    #             bigger_frame = np.zeros((diag,2*diag)).astype('uint8')
    #             bigger_frame[0:diag,diag/2:3*diag/2] = new_frame
    #             # cv2.circle(new_frame, tuple(np.asarray([cent_rt[0], cent_rt[1]]).astype('int')),2,0,1)
    #             # cv2.circle(new_frame, tuple(np.asarray([ext_rt[0], ext_rt[1]]).astype('int')),2,255,1)
    #             # cv2.putText(new_frame,str(j),(50,50), font, 1,255)
    #             # if ext_rt[0] < 16 or (diag - ext_rt[0]) < 16:
    #             #     padding_frame = np.zeros()
    #             rect1 = (ext_rt[0]-15+diag/2, 0)
    #             rect2 = (ext_rt[0]+16+diag/2, 31)
    #             # cv2.circle(new_frame, tuple(np.asarray([cent_rt[0], cent_rt[1]]).astype('int')),2,255,1)
    #             # cv2.rectangle(bigger_frame, rect1, rect2, 255, 1)
    #             cv2.imshow('b', bigger_frame)
    #             square_frame = bigger_frame[rect1[1]:rect2[1],rect1[0]:rect2[0]]
    #
    #             square_frame_filled = fillSquareFrame(square_frame,bkgSamps[j])
    #             # get all the heads in a single list
    #             portraits.append(square_frame_filled)
    #
    #             cv2.imshow('thresholdedMiniframes'+str(j),square_frame_filled)
    #
    #         AllPortraits.set_value(goodFrameIndices[counter], 'images', portraits)
    #         k = cv2.waitKey(1) & 0xFF
    #         if k == 27: #pres esc to quit
    #             break
    #         counter += 1
