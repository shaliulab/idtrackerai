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

def collectAndSaveVideoInfo(path, height, width, mask, ROICenters, numAnimals, numCores, minThreshold,maxThreshold,maxArea,maxNumBlobs):
    """
    saves general info about the video in a pickle (_videoinfo.pkl)
    """
    videoInfo = {
        'path': path,
        'height':height,
        'width': width,
        'mask': mask,
        'ROICenters': ROICenters,
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
    saveFile(path, videoTOC, 'frameIndices', time = 0)

"""
Compute background and threshold
"""
def computeBkgPar(path,bkg,EQ):
    print 'Adding video %s to background' % path
    cap = cv2.VideoCapture(path)
    counter = 0
    numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    while counter < numFrame:
        counter += 1;
        ret, frameBkg = cap.read()
        gray = cv2.cvtColor(frameBkg, cv2.COLOR_BGR2GRAY)
        # gray = checkEq(EQ, gray)
        gray,_ = frameAverager(gray) ##XXX
        bkg = bkg + gray

    return bkg

def computeBkg(paths, EQ, width, height):
    # This holds even if we have not selected a ROI because then the ROI is
    # initialized as the full frame
    bkg = np.zeros((height,width))

    num_cores = multiprocessing.cpu_count()
    # num_cores = 1
    numFrame = Parallel(n_jobs=num_cores)(delayed(getNumFrame)(path) for path in paths)
    partialBkg = Parallel(n_jobs=num_cores)(delayed(computeBkgPar)(path,bkg,EQ) for path in paths)
    bkg = np.sum(np.asarray(partialBkg),axis=0)
    totNumFrame = sum(numFrame)
    bkg = np.true_divide(bkg, totNumFrame)
    # bkg is the backgorund computed by summing all the averaged frames
    # of the video and dividing by the number of frames in the video.
    return bkg

def checkBkg(useBkg, usePreviousBkg, paths, EQ, width, height):
    ''' Compute Bkg '''
    path = paths[0]
    if useBkg:
        if usePreviousBkg:
            bkg = loadFile(path, 'bkg',0)
        else:
            bkg = computeBkg(paths, EQ, width, height)
            saveFile(path, bkg, 'bkg', time = 0)
        return bkg
    else:
        return None


# """
# ROI selector GUI
# """
# def ROIselector(frame):
#     plt.ion()
#     f, ax = plt.subplots()
#     ax.imshow(frame, interpolation='nearest', cmap='gray')
#     props = {'facecolor': '#000070',
#              'edgecolor': 'white',
#              'alpha': 0.3}
#     rect_tool = RectangleTool(ax, rect_props=props)
#
#     plt.show()
#     numROIs = getInput('Number of ROIs','Type the number of ROIs to be selected')
#     numROIs = int(numROIs)
#     print 'The number of ROIs to select is ', numROIs
#     counter = 0
#     ROIsCoords = []
#     centers = []
#     ROIsShapes = []
#     mask = np.ones_like(frame,dtype='uint8')*255
#     while counter < numROIs:
#         ROIshape = getInput('Roi shape','r= rect, c=circ')
#         # ROIshape = raw_input('ROI shape (r/c/p)? (press enter after selection)')
#
#         if ROIshape == 'r' or ROIshape == 'c':
#             ROIsShapes.append(ROIshape)
#
#             rect_tool.callback_on_enter(rect_tool.extents)
#             coord = np.asarray(rect_tool.extents).astype('int')
#
#             print 'ROI coords, ', coord
#             # goodROI=raw_input('Is the selection correct? [y]/n: ')
#             text = 'Is ' + str(coord) + ' the ROI you wanted to select? y/n'
#             goodROI = getInput('Confirm selection',text)
#             if goodROI == 'y':
#                 ROIsCoords.append(coord)
#                 if ROIshape == 'r':
#                     cv2.rectangle(mask,(coord[0],coord[2]),(coord[1],coord[3]),0,-1)
#                     centers.append(None)
#                 if ROIshape == 'c':
#                     center = ((coord[1]+coord[0])/2,(coord[3]+coord[2])/2)
#                     angle = 90
#                     axes = tuple(sorted(((coord[1]-coord[0])/2,(coord[3]-coord[2])/2)))
#                     print center, angle, axes
#                     cv2.ellipse(mask,center,axes,angle,0,360,0,-1)
#                     centers.append(center)
#
#         counter = len(ROIsCoords)
#     plt.close("all")
#
#     return mask, centers
#
# def checkROI(selectROI,frame, path):
#     ''' Select ROI '''
#     if selectROI:
#         try:
#             mask = loadFile(path, 'mask',0)
#             center = loadFile(path, 'center',0)
#         except:
#             print '\n Selecting ROI ...'
#             mask, centers = ROIselector(frame)
#     else:
#         mask = np.zeros_like(frame)
#         centers = []
#     return mask, centers

"""
Normalize by the average intensity
"""
def frameAverager(frame):
    avFrame = np.divide(frame,np.mean(frame))
    # avFrame = np.multiply(np.true_divide(avFrame,np.max(avFrame)),255).astype('uint8')
    return avFrame, np.mean(avFrame)

"""
Image equalization
"""
def checkEq(EQ, frame):
    if EQ:
        # Equalize image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))
        frame = clahe.apply(frame)
    return frame

"""
Image segmentation
"""
def segmentVideo(frame, minThreshold, maxThreshold, bkg, mask, useBkg):
    #Apply background substraction if requested and threshold image
    if useBkg:
        frameSubtracted = uint8caster(np.abs(np.subtract(bkg,frame)))
        frameSubtractedMasked = cv2.addWeighted(frameSubtracted,1,mask,1,0)
        ### Uncomment to plot
        # cv2.imshow('frameSubtractedMasked',frameSubtractedMasked)
        # ret, frame = cv2.threshold(frame,minThreshold,maxThreshold, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        frameSubtractedMasked = 255-frameSubtractedMasked
        ret, frameSegmented = cv2.threshold(frameSubtractedMasked,minThreshold,maxThreshold, cv2.THRESH_BINARY)
    else:
        frameMasked = cv2.addWeighted(uint8caster(frame),1,mask,1,0)
        ### Uncomment to plot
        # cv2.imshow('frameMasked',frameMasked)
        ret, frameSegmented = cv2.threshold(frameMasked,minThreshold,maxThreshold, cv2.THRESH_BINARY)
    return frameSegmented

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
