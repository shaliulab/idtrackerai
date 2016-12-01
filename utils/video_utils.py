import cv2
import sys
sys.path.append('../utils')

from py_utils import *

import numpy as np
from matplotlib import pyplot as plt
from Tkinter import *
import tkMessageBox
import os
import glob
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from skimage import data
from skimage.viewer.canvastools import RectangleTool, PaintTool
from skimage.viewer import ImageViewer
from scipy import ndimage
import Tkinter, tkSimpleDialog

"""
Get general information from video
"""
def getVideoInfo(paths):
    if len(paths) == 1:
        path = paths
    elif len(paths) > 1:
        path = paths[0]
    else:
        raise ValueError('the path (or list of path) seems to be empty')
    cap = cv2.VideoCapture(paths[0])
    width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    return width, height

def getNumFrame(path):
    cap = cv2.VideoCapture(path)
    return int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

def collectAndSaveVideoInfo(path, numFrames, height, width, numAnimals, numCores, minThreshold,maxThreshold,maxArea,maxNumBlobs):
    """
    saves general info about the video in a pickle (_videoinfo.pkl)
    """
    videoInfo = {
        'path': path,
        'numFrames': numFrames,
        'height':height,
        'width': width,
        'numAnimals':numAnimals,
        'numCores':numCores,
        'minThreshold':minThreshold,
        'maxThreshold':maxThreshold,
        'maxArea': maxArea,
        'maxNumBlobs':maxNumBlobs
        }
    print videoInfo
    saveFile(path, videoInfo, 'videoInfo',time = 0)

def generateVideoTOC(allSegments, path):
    """
    generates a dataframe mapping frames to segments and save it as pickle
    """
    segmentsTOC = []
    framesTOC = []
    for segment, frame in allSegments:
        segmentsTOC.append(segment)
        framesTOC.append(frame)
    segmentsTOC = flatten(segmentsTOC)
    framesTOC = flatten(framesTOC)
    videoTOC =  pd.DataFrame({'segment':segmentsTOC, 'frame': framesTOC})
    numFrames = len(videoTOC)
    saveFile(path, videoTOC, 'frameIndices', time = 0)
    return numFrames

"""
Compute background and threshold
"""
def computeBkgPar(path,bkg,EQ):
    print 'Adding video %s to background' % path
    cap = cv2.VideoCapture(path)
    counter = 0
    numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    numFramesBkg = 0
    while counter < numFrame:
        counter += 100;
        ret, frameBkg = cap.read()
        gray = cv2.cvtColor(frameBkg, cv2.COLOR_BGR2GRAY)
        # gray = checkEq(EQ, gray)
        gray = np.true_divide(gray,np.mean(gray))
        bkg = bkg + gray
        numFramesBkg += 1

    return bkg, numFramesBkg

def computeBkg(paths, EQ, width, height):
    # This holds even if we have not selected a ROI because then the ROI is
    # initialized as the full frame
    bkg = np.zeros((height,width))

    num_cores = multiprocessing.cpu_count()
    # num_cores = 1
    # numFrame = Parallel(n_jobs=num_cores)(delayed(getNumFrame)(path) for path in paths)
    output = Parallel(n_jobs=num_cores)(delayed(computeBkgPar)(path,bkg,EQ) for path in paths)
    partialBkg = [bkg for (bkg,_) in output]
    totNumFrame = np.sum([numFrame for (_,numFrame) in output])
    # allBkg = np.asarray(partialBkg)
    # print '********************** ', np.asarray(partialBkg).shape
    bkg = np.sum(np.asarray(partialBkg),axis=0)
    # totNumFrame = allBkg.shape[0]
    bkg = np.true_divide(bkg, totNumFrame)
    # bkg is the backgorund computed by summing all the averaged frames
    # of the video and dividing by the number of frames in the video.
    # print '**********bkg, ', bkg.shape
    # plt.imshow(bkg)
    # plt.show()
    return bkg

def checkBkg(useBkg, usePreviousBkg, paths, EQ, width, height):
    ''' Compute Bkg ''' ###TODO This can be done in a smarter way...
    path = paths[0]
    if useBkg:
        if usePreviousBkg:
            bkg = loadFile(path, 'bkg',0,hdfpkl='pkl')
            # bkgDict = bkgDict.to_dict()[0]
            # bkg = bkgDict['bkg']
            # maxIntensity = bkgDict['maxIntensity']
            # maxBkg = bkgDict['maxBkg']
        else:
            bkg = computeBkg(paths, EQ, width, height)
            # bkgDict = {'bkg': bkg}#, 'maxIntensity': maxIntensity, 'maxBkg': maxBkg}
            saveFile(path, bkg, 'bkg', time = 0,hdfpkl='pkl')
        return bkg#, maxIntensity, maxBkg
    else:
        return None#, maxIntensity, None

"""
Image equalization
"""
def checkEq(EQ, frame):
    if EQ:
        # Equalize image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))
        frame = clahe.apply(frame)
    return frame


def segmentVideo(frame, minThreshold, maxThreshold, bkg, mask, useBkg):
    #Apply background substraction if requested and threshold image
    # print 'minThreshold, ', minThreshold
    # print 'maxThreshold, ', maxThreshold

    # compute the average frame
    frame = np.true_divide(frame,np.mean(frame))
    if useBkg:

        # print 'max and min frame, ', np.max(frame), np.min(frame)
        # print 'max and min bkg, ', np.max(bkg), np.min(bkg)
        frameSubtracted = uint8caster(np.abs(np.subtract(bkg,frame)))
        frameSubtractedMasked = cv2.addWeighted(frameSubtracted,1,mask,1,0)
        # Uncomment to plot
        # cv2.imshow('frame', uint8caster(frame))
        # cv2.imshow('frameSubtractedMasked',frameSubtractedMasked)
        frameSubtractedMasked = 255-frameSubtractedMasked
        # cv2.imshow('frameSubtractedMasked',frameSubtractedMasked)
        ret, frameSegmented = cv2.threshold(frameSubtractedMasked,minThreshold,maxThreshold, cv2.THRESH_BINARY)
    else:
        frameMasked = cv2.addWeighted(uint8caster(frame),1,mask,1,0)
        ### Uncomment to plot
        # cv2.imshow('frameMasked',frameMasked)
        ret, frameSegmented = cv2.threshold(frameMasked,minThreshold,maxThreshold, cv2.THRESH_BINARY)
    return frameSegmented

# def segmentVideoIdTracker(frame, minThreshold, maxThreshold, bkg, mask, useBkg):
#     #Apply background substraction if requested and threshold image
#     # print 'minThreshold, ', minThreshold
#     # print 'maxThreshold, ', maxThreshold
#
#     # compute the average frame
#     mask[mask == 255] = 1
#     frame = np.true_divide(frame,np.mean(frame))
#     if useBkg:
#         frameMasked = frame + mask
#         bkgMasked = bkg + mask
#         # cv2.imshow('frameMasked', uint8caster(frameMasked))
#         # cv2.imshow('bkgMasked',uint8caster(bkgMasked))
#
#         frameMaskedThresholded = frameMasked < minThreshold
#         bkgMaskedThresholded = bkgMasked < minThreshold
#
#         # cv2.imshow('frame', uint8caster(frameMaskedThresholded))
#         # cv2.imshow('bkgMaskedThresholded',uint8caster(bkgMaskedThresholded))
#
#         frameSegmented = frameMaskedThresholded-bkgMaskedThresholded
#         # cv2.imshow('bkgMaskedThresholded',uint8caster(frameSegmented))
#         frameSegmented = np.uint8(frameSegmented)
#         # cv2.waitKey(1)
#     else:
#         frameMasked = frame + mask
#         frameSegmented = frameMasked < minThreshold
#         frameSegmented = np.uint8(frameSegmented)
#     return frameSegmented


"""
Get information from blobs
"""
def filterContoursBySize(contours,minArea,maxArea):
    goodContours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > minArea and area < maxArea:
            goodContours.append(contour)
    return goodContours

def coordROI2Full(x,y,ROI):
    if ROI[0][0] > ROI[1][0]:
        ROI = [ROI[1],ROI[0]]
    x = x + ROI[0][0]
    y = y + ROI[0][1]
    return x, y

def cntROI2OriginalFrame(contours,ROI):
    contoursFull = []
    for cnt in contours:
        cnt = cnt + np.asarray([ROI[0][0],ROI[0][1]])
        contoursFull.append(cnt)
    return contoursFull

def cnt2BoundingBox(cnt,boundingBox):
    return cnt - np.asarray([boundingBox[0][0],boundingBox[0][1]])

def cntBB2Full(cnt,boundingBox):
    return cnt + np.asarray([boundingBox[0][0],boundingBox[0][1]])

def getBoundigBox(cnt, width, height):
    x,y,w,h = cv2.boundingRect(cnt)
    n = 10
    if x - n > 0: # We only expand the
        x = x - n
    else:
        x = 0
    if y - n > 0:
        y = y - n
    else:
        y = 0
    if x + w + 2*n < width:
        w = w + 2*n
    else:
        w = width - x
    if y + h + 2*n < height:
        h = h + 2*n
    else:
        h = height - y
    return ((x,y),(x+w,y+h))

def boundingBox_ROI2Full(bb, ROI):
    """
    Gives back only the first point of bb translated in the full frame
    """
    return ((bb[0][0] + ROI[0][0], bb[0][1] + ROI[0][1]),(bb[1][0] + ROI[0][0], bb[1][1] + ROI[0][1]))

def getCentroid(cnt):
    M = cv2.moments(cnt)
    x = int(M['m10']/M['m00'])
    y = int(M['m01']/M['m00'])
    # x, y = coordROI2Full(x,y,ROI)
    return (x,y)

def getPixelsList(cnt, width, height):
    cimg = np.zeros((height, width))
    cv2.drawContours(cimg, [cnt], -1, color=255, thickness = -1)
    # Access the image pixels and create a 1D numpy array then add to list
    pts = np.where(cimg == 255)
    return zip(pts[0],pts[1])

def sampleBkg(cntBB, miniFrame):
    frame = np.zeros((500,500)).astype('uint8')
    cv2.drawContours(miniFrame, [cntBB], -1, color=255, thickness = -1)
    # Access the image pixels and create a 1D numpy array then add to list
    bkgSample = miniFrame[np.where(miniFrame != 255)]
    return bkgSample

def getMiniFrame(frame, cnt, height, width):
    boundingBox = getBoundigBox(cnt, width, height)
    miniFrame = frame[boundingBox[0][1]:boundingBox[1][1], boundingBox[0][0]:boundingBox[1][0]]
    cntBB = cnt2BoundingBox(cnt,boundingBox)
    miniFrameBkg = miniFrame.copy()
    bkgSample = sampleBkg(cntBB, miniFrameBkg)
    pixelsInBB = getPixelsList(cntBB, np.abs(boundingBox[0][0]-boundingBox[1][0]), np.abs(boundingBox[0][1]-boundingBox[1][1]))
    pixelsInFullF = pixelsInBB + np.asarray([boundingBox[0][1],boundingBox[0][0]])
    pixelsInFullFF = np.ravel_multi_index([pixelsInFullF[:,0],pixelsInFullF[:,1]],(height,width))
    return boundingBox, miniFrame, pixelsInFullFF, bkgSample

def getBlobsInfoPerFrame(frame, contours, height, width):
    boundingBoxes = []
    miniFrames = []
    centroids = []
    areas = []
    pixels = []
    bkgSamples = []
    for i, cnt in enumerate(contours):
        boundingBox, miniFrame, pixelsInFullF, bkgSample = getMiniFrame(frame, cnt, height, width)
        bkgSamples.append(bkgSample)
        boundingBoxes.append(boundingBox)
        # miniframes
        miniFrames.append(miniFrame)
        # centroid
        centroids.append(getCentroid(cnt))
        # area
        areas.append(cv2.contourArea(cnt))
        # pixels list
        pixels.append(pixelsInFullF)
    return boundingBoxes, miniFrames, centroids, areas, pixels, bkgSamples

def blobExtractor(segmentedFrame, frame, minArea, maxArea, height, width):
    # contours, hierarchy = cv2.findContours(segmentedFrame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(segmentedFrame,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # Filter contours by size
    goodContoursFull = filterContoursBySize(contours,minArea, maxArea)
    # get contours properties
    boundingBoxes, miniFrames, centroids, areas, pixels, bkgSamples = getBlobsInfoPerFrame(frame, goodContoursFull, height, width)

    return boundingBoxes, miniFrames, centroids, areas, pixels, goodContoursFull, bkgSamples
