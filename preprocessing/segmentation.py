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
import cPickle as pickle
import gc
import datetime
"""
Display messages and errors
"""
def displayMessage(title,message):
    window = Tk()
    window.wm_withdraw()

    #centre screen message
    window.geometry("1x1+"+str(window.winfo_screenwidth()/2)+"+"+str(window.winfo_screenheight()/2))
    tkMessageBox.showinfo(title=title, message=message)

def displayError(title, message):
    #message at x:200,y:200
    window = Tk()
    window.wm_withdraw()

    window.geometry("1x1+200+200")#remember its .geometry("WidthxHeight(+or-)X(+or-)Y")
    tkMessageBox.showerror(title=title,message=message,parent=window)

def collectAndSaveVideoInfo(path, height, width, ROI, numAnimals, numCores, minThreshold,maxThreshold,maxArea,maxNumBlobs):
    """
    saves general info about the video in a pickle (_videoinfo.pkl)
    """
    videoInfo = {
        'path': path,
        'height':height,
        'width': width,
        'ROI': tuple(ROI),
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
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = checkEq(EQ, gray)
        # gray,_ = frameAverager(gray) ##XXX
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

def checkBkg(bkgSubstraction, paths, ROI, EQ, width, height):
    ''' Compute Bkg '''
    if bkgSubstraction:
        video = os.path.basename(paths[0])
        folder = os.path.dirname(paths[0])
        filename, extension = os.path.splitext(video)

        subFolders = natural_sort(glob.glob(folder +"/*/"))[::-1]
        if len(subFolders) >= 2:
            subFolder = subFolders[1]
            filename = subFolder + filename.split('_')[0] + '_bkg.pkl'
            print filename
            if os.path.isfile(filename):
            # if False:
                print '\n Loading background ...\n'
                bkg = loadFile(paths[0], 'bkg',1)
                path = paths[0]
                saveFile(path, bkg, 'bkg', time = 0)

            else:
                print '\n Computing background ...\n'
                bkg = computeBkg(paths, EQ, width, height)
                path = paths[0]
                saveFile(path, bkg, 'bkg', time = 0)

        else:
            print '\n Computing background ...\n'
            bkg = computeBkg(paths, EQ, width, height)
            path = paths[0]
            saveFile(path, bkg, 'bkg', time = 0)

        bkg = cropper(bkg, ROI)
        return bkg

    else:
        return None

"""
ROI selector GUI
"""

def ROIselector(frame):
    global ROI_
    ROI_ = []
    def ROIselectorCallBack(event, x, y, flags, params):
    	# grab references to the global variables
        global ROI_, select

    	# if the left mouse button was clicked, record the starting
    	# (x, y) coordinates and indicate that cropping is being
    	# performed
    	if event == cv2.EVENT_LBUTTONDOWN:
    		ROI_ = [(x, y)]
    		select = True

    	# check to see if the left mouse button was released
    	elif event == cv2.EVENT_LBUTTONUP:
    		# record the ending (x, y) coordinates and indicate that
    		# the cropping operation is finished
    		ROI_.append((x, y))
    		select = False

    		# draw a rectangle around the region of interest
    		cv2.rectangle(frame, ROI_[0], ROI_[1], (0, 255, 0), 2)
    		cv2.imshow("ROIselector", frame)

    # cap = cv2.VideoCapture(paths[0])
    # ret, frame = cap.read()
    cv2.namedWindow("ROIselector")
    cv2.setMouseCallback("ROIselector", ROIselectorCallBack)
    # if there are two reference points, the ROI has been selected properly
    displayMessage('ROI selector warning','Click and drag to select a rectangular region of interest')
    # otherwise we will need to set at while loop
    while len(ROI_) != 2:
        # if len(ROI) != 2 and ROI is not []:
        # keep looping until the 'q' key is pressed
        while True:
        	# display the image and wait for a keypress
        	cv2.imshow("ROIselector", frame)
        	key = cv2.waitKey(1) & 0xFF

            # if the 'c' key is pressed, break from the loop
        	if key == ord("c"):
        		break
    return ROI_

def checkROI(selectROI,frame):
    height, width, channels = frame.shape
    ''' Select ROI '''
    if selectROI:
        print '\n Selecting ROI ...'
        ROI = ROIselector(frame)
    else:
        ROI = [(0,0),(width, height)]
    return ROI

"""
Cropper and masker
"""
def maskROI(frame, ROI):
    mask = np.zeros_like(frame).astype('uint8')
    mask[ROI[0][1]:ROI[1][1], ROI[0][0]:ROI[1][0]] = 255.
    maskedFrame = cv2.addWeighted(frame,0.7,mask,0.3,0)
    # maskedFrame = frame * mask
    return maskedFrame

def cropper(frame, ROI):
    return frame[ROI[0][1]:ROI[1][1], ROI[0][0]:ROI[1][0]]

def masker(frame, maskFrame, ROI, selectROI):
    if selectROI:
        if maskFrame:
            frame = maskROI(frame, ROI)
    return frame


"""
Normalize by the average intensity
"""

def frameAverager(frame):
    avFrame = np.divide(frame,np.mean(frame))
    return np.multiply(np.true_divide(avFrame,np.max(avFrame)),255).astype('uint8'), np.mean(frame)

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
def segmentVideo(frame, minThreshold, maxThreshold, bkg, bkgSubstraction):
    #Apply background substractions if requested and thresholding image
    if bkgSubstraction:
        frame = np.abs(np.subtract(frame, bkg))
        frame = np.multiply(np.true_divide(frame,np.max(frame)),255).astype('uint8')
        # frameBkg = frame.copy()
        # ret, frame = cv2.threshold(frame,minThreshold,maxThreshold, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret, frame = cv2.threshold(frame,minThreshold,maxThreshold, cv2.THRESH_BINARY_INV)
    else:
        frame = np.multiply(np.true_divide(frame,np.max(frame)),255).astype('uint8')
        # frameBkg = frame.copy()
        ret, frame = cv2.threshold(frame,minThreshold,maxThreshold, cv2.THRESH_BINARY_INV)
    return frame#, frameBkg

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

def cntROI2BoundingBox(cnt,boundingBox):
    return cnt - np.asarray([boundingBox[0][0],boundingBox[0][1]])

# def getBoundigBox(cnt):
#     x,y,w,h = cv2.boundingRect(cnt)
#     n = 10
#     if (x > n and y > n): # We only expand the
#         x = x - n #2
#         y = y - n #2
#         w = w + 2*n #4
#         h = h + 2*n #4
#     return ((x,y),(x+w,y+h))

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

def getCentroid(cnt,ROI):
    M = cv2.moments(cnt)
    x = int(M['m10']/M['m00'])
    y = int(M['m01']/M['m00'])
    x, y = coordROI2Full(x,y,ROI)
    return (x,y)

def getPixelsList(cnt, width, height):
    cimg = np.zeros((height, width))
    cv2.drawContours(cimg, [cnt], -1, color=255, thickness = -1)
    # cv2.imshow('cnt',cimg)
    # Access the image pixels and create a 1D numpy array then add to list
    pts = np.where(cimg == 255)
    return zip(pts[0],pts[1])

def sampleBkg(cntBB, miniFrame):
    # print cntBB
    # print miniFrame.shape
    frame = np.zeros((500,500)).astype('uint8')
    # cv2.drawContours(frame, [cntBB], -1, color=255, thickness = -1)
    # cv2.imshow('a',frame)
    # cv2.waitKey(1)
    cv2.drawContours(miniFrame, [cntBB], -1, color=255, thickness = -1) # FIXME there is a bug that depends on the ROI

    # cv2.imshow('cnt',cimg)
    # Access the image pixels and create a 1D numpy array then add to list
    bkgSample = miniFrame[np.where(miniFrame != 255)]
    return bkgSample

def getMiniFrame(frame, cnt, ROI, height, width):
    # print '1', ROI
    boundingBox = getBoundigBox(cnt, width, height)
    bb = boundingBox_ROI2Full(boundingBox, ROI)
    # print 'ROI ', ROI
    # print 'frame shape, ', frame.shape
    # print 'boundingBox, ', boundingBox
    miniFrame = frame[boundingBox[0][1]:boundingBox[1][1], boundingBox[0][0]:boundingBox[1][0]]
    # print 'miniFrame shape, ', miniFrame.shape
    cntBB = cntROI2BoundingBox(cnt,boundingBox)
    miniFrameBkg = miniFrame.copy()
    # print 'miniFrameBkg shape, ', miniFrameBkg.shape
    bkgSample = sampleBkg(cntBB, miniFrameBkg)
    pixelsInBB = getPixelsList(cntBB, np.abs(boundingBox[0][0]-boundingBox[1][0]), np.abs(boundingBox[0][1]-boundingBox[1][1]))
    pixelsInROI = pixelsInBB + np.asarray([boundingBox[0][1],boundingBox[0][0]])
    # print '2', ROI
    pixelsInFullF = pixelsInROI + np.asarray([ROI[0][1],ROI[0][0]])
    pixelsInFullFF = np.ravel_multi_index([pixelsInFullF[:,0],pixelsInFullF[:,1]],(height,width))
    return bb, miniFrame, pixelsInFullFF, bkgSample

def getBlobsInfoPerFrame(frame, contours, ROI, height, width):
    boundingBoxes = []
    miniFrames = []
    centroids = []
    areas = []
    pixels = []
    bkgSamples = []
    for i, cnt in enumerate(contours):
        # boundigBox
        # print '-------------'
        # print 'contour, ', i
        boundingBox, miniFrame, pixelsInFullF, bkgSample = getMiniFrame(frame, cnt, ROI, height, width)
        bkgSamples.append(bkgSample)
        boundingBoxes.append(boundingBox)
        # miniframes
        miniFrames.append(miniFrame)
        # centroid
        centroids.append(getCentroid(cnt,ROI))
        # area
        areas.append(cv2.contourArea(cnt))
        # pixels list
        pixels.append(pixelsInFullF)
    return boundingBoxes, miniFrames, centroids, areas, pixels, bkgSamples

def blobExtractor(segmentedFrame, frame, minArea, maxArea, ROI, height, width):
    # contours, hierarchy = cv2.findContours(segmentedFrame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(segmentedFrame,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    # Filter contours by size
    goodContours = filterContoursBySize(contours,minArea, maxArea)
    # Pass contours' coordinates from ROI to full size image if the ROI
    # has been cropped
    goodContoursFull = cntROI2OriginalFrame(goodContours,ROI) #only used to plot in the full frame

    ### uncomment to plot contours
    # cv2.drawContours(frame, goodContours, -1, (255,0,0), -1)
    # get contours properties
    boundingBoxes, miniFrames, centroids, areas, pixels, bkgSamples = getBlobsInfoPerFrame(frame, goodContours, ROI, height, width)

    return boundingBoxes, miniFrames, centroids, areas, pixels, goodContoursFull, bkgSamples

def segmentAndSave(path, height, width, ROI, selectROI, bkg, bkgSubstraction, EQ, maskFrame, minThreshold, maxThreshold, minArea, maxArea):
    print 'Segmenting video %s' % path
    cap = cv2.VideoCapture(path)
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    numSegment = int(filename.split('_')[-1])
    numFrames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    counter = 0
    df = pd.DataFrame(columns=('avIntensity', 'boundingBoxes','miniFrames', 'contours', 'centroids', 'areas', 'pixels', 'numberOfBlobs', 'bkgSamples'))
    maxNumBlobs = 0
    while counter < numFrames:
        # print counter
        #Get frame from video file
        ret, frame = cap.read()
        # frameToPlot = masker(frame, maskFrame, ROI, selectROI)
        #Color to gray scale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Equalize image to enhance contrast
        frame = checkEq(EQ, frame)
        frameToPlot = masker(frame, maskFrame, ROI, selectROI)
        # mask or crop the image
        frame = cropper(frame, ROI)
        avFrame = frame
        #Normalize each frame by its mean intensity
        frame, avIntensity = frameAverager(frame)

        # perform background subtraction if needed
        # segmentedFrame, frameBkg = segmentVideo(frame, minThreshold, maxThreshold, bkg, bkgSubstraction)
        segmentedFrame = segmentVideo(frame, minThreshold, maxThreshold, bkg, bkgSubstraction)
        # frameBkg = 255 - frameBkg
        # Find contours in the segmented image
        # print '***********************'
        # print 'frame, ', counter
        boundingBoxes, miniFrames, centroids, areas, pixels, goodContoursFull, bkgSamples = blobExtractor(segmentedFrame, avFrame, minArea, maxArea, ROI, height, width)

        if len(centroids) > maxNumBlobs:
            maxNumBlobs = len(centroids)

        ### UNCOMMENT TO PLOT ##################################################
        cv2.drawContours(frameToPlot,goodContoursFull,-1,color=(255,0,0),thickness=-1)
        cv2.imshow('checkcoord', frameToPlot)
        k = cv2.waitKey(30) & 0xFF
        if k == 27: #pres esc to quit
            break
        ########################################################################

        # Add frame imformation to DataFrame
        df.loc[counter] = [avIntensity, boundingBoxes, miniFrames, goodContoursFull, centroids, areas, pixels, len(centroids), bkgSamples]
        counter += 1
    cap.release()
    cv2.destroyAllWindows()

    saveFile(path, df, 'segment', time = 0)

    return np.multiply(numSegment,np.ones(numFrames)).astype('int').tolist(), np.arange(numFrames).tolist(), maxNumBlobs


def segment(paths, name, numAnimals,
            bkgSubstraction, selectROI, maskFrame, EQ,
            minThreshold, maxThreshold,
            minArea, maxArea):

    createFolder(paths[0], name = name, timestamp = True)

    width, height = getVideoInfo(paths)
    cap = cv2.VideoCapture(paths[0])
    ret, frame = cap.read()
    cap.release()

    ROI = []
    ROI = checkROI(selectROI, frame)
    bkg = checkBkg(bkgSubstraction, paths, ROI, EQ, width, height)
    ''' Entering loop for segmentation of the video '''

    # for path in paths:
    #     df = segmentAndSave(path)

    num_cores = multiprocessing.cpu_count()

    num_cores = 1
    print 'Entering to the parallel loop'
    OupPutParallel = Parallel(n_jobs=num_cores)(delayed(segmentAndSave)(path, height, width, ROI, selectROI, bkg, bkgSubstraction, EQ, maskFrame,minThreshold,maxThreshold,minArea,maxArea) for path in paths)
    allSegments = [(out[0],out[1]) for out in OupPutParallel]
    # print allSegments
    maxNumBlobs = max([out[2] for out in OupPutParallel])
    # print maxNumBlobs
    allSegments = sorted(allSegments, key=lambda x: x[0][0])
    generateVideoTOC(allSegments, paths[0])
    collectAndSaveVideoInfo(paths[0], height, width, ROI, numAnimals, num_cores, minThreshold,maxThreshold,maxArea,maxNumBlobs)

if __name__ == '__main__':

    # videoPath = '../Cafeina5peces/Caffeine5fish_20140206T122428_1.avi'
    videoPath = '../Conflict8/conflict3and4_20120316T155032_1.avi'

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', default = videoPath, type = str)
    parser.add_argument('--folder_name', default = '', type = str)
    parser.add_argument('--bkg_subtraction', default = 1, type = int)
    parser.add_argument('--ROI_selection', default = 1, type = int)
    parser.add_argument('--mask_frame', default = 1, type= int)
    parser.add_argument('--Eq_image', default = 0, type = int)
    parser.add_argument('--min_th', default = 90, type = int)
    parser.add_argument('--max_th', default = 255, type = int)
    parser.add_argument('--min_area', default = 250, type = int)
    parser.add_argument('--max_area', default = 500, type = int)
    parser.add_argument('--num_animals', default = 8, type = int)
    args = parser.parse_args()

    ''' Parameters for the segmentation '''
    numAnimals = args.num_animals
    bkgSubstraction = args.bkg_subtraction
    selectROI = args.ROI_selection
    maskFrame = args.mask_frame
    EQ = args.Eq_image
    minThreshold = args.min_th
    maxThreshold = args.max_th
    minArea = args.min_area # in pixels
    maxArea = args.max_area # in pixels

    ''' Path to video/s '''
    paths = scanFolder(args.path)
    name  = args.folder_name

    segment(paths, name, numAnimals,
                bkgSubstraction, selectROI, maskFrame, EQ,
                minThreshold, maxThreshold,
                minArea, maxArea)
