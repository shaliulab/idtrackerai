import cv2
import sys
import time
import numpy as np
from matplotlib import pyplot as plt
from Tkinter import *
import tkMessageBox
import argparse
import os
import glob
import pandas as pd
import h5py
import time
import re

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

"""
Scan folders  for videos
"""
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def scanFolder(path):
    paths = [path]
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    folder = os.path.dirname(path)
    # maybe write check on video extension supported by opencv2
    if filename[-2:] == '_1':
        paths = natural_sort(glob.glob(folder + "/" + filename[:-1] + "*" + extension))
    return paths

"""
Get general information from video
"""
def getVideoInfo(paths):
    cap = cv2.VideoCapture(paths[0])
    width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    return width, height

"""
Compute background and threshold
"""
def computeBkg(paths, ROI, EQ):
    # This holds even if we have not selected a ROI because then the ROI is the
    # full frame
    bkg = np.zeros(
    (
    np.abs(np.subtract(ROI[0][1],ROI[1][1])),
    np.abs(np.subtract(ROI[0][0],ROI[1][0]))
    )
    )

    totNumFrame = 0
    for path in paths:
        print 'Adding video %s to background' % path
        cap = cv2.VideoCapture(path)
        numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        # width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        counter = 0
        totNumFrame = totNumFrame + numFrame
        while counter < numFrame:
            counter += 1;
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cropper(gray, ROI)
            gray = checkEq(EQ, gray)
            gray,_ = frameAverager(gray)
            bkg = bkg + gray
    bkg = np.true_divide(bkg, totNumFrame)
    # bkg is the backgorund computed by summing all the averaged frames
    # of the video and dividing by the number of frames in the video.
    return bkg

def bkgSubtraction(frame, bkg):
    return np.abs(np.subtract(frame,bkg))

def checkBkg(bkgSubstraction, paths, ROI, EQ):
    ''' Compute Bkg '''
    if bkgSubstraction:
        print '\n Computing background ...\n'
        bkg = computeBkg(paths, ROI, EQ)

        return bkg
    else:
        return None

"""
ROI selector GUI
"""
def ROIselector(paths):

    def ROIselectorCallBack(event, x, y, flags, params):
    	# grab references to the global variables
    	global ROI, selectROI_

    	# if the left mouse button was clicked, record the starting
    	# (x, y) coordinates and indicate that cropping is being
    	# performed
    	if event == cv2.EVENT_LBUTTONDOWN:
    		ROI = [(x, y)]
    		selectROI_ = True

    	# check to see if the left mouse button was released
    	elif event == cv2.EVENT_LBUTTONUP:
    		# record the ending (x, y) coordinates and indicate that
    		# the cropping operation is finished
    		ROI.append((x, y))
    		selectROI_ = False

    		# draw a rectangle around the region of interest
    		cv2.rectangle(frame, ROI[0], ROI[1], (0, 255, 0), 2)
    		cv2.imshow("ROIselector", frame)

    cap = cv2.VideoCapture(paths[0])
    ret, frame = cap.read()
    cv2.namedWindow("ROIselector")
    cv2.setMouseCallback("ROIselector", ROIselectorCallBack)
    # if there are two reference points, the ROI has been selected properly
    displayMessage('ROI selector warning','Click and drag to select a rectangular region of interest')
    # otherwise we will need to set at while loop
    while len(ROI) != 2:
        # if len(ROI) != 2 and ROI is not []:
        # keep looping until the 'q' key is pressed
        while True:
        	# display the image and wait for a keypress
        	cv2.imshow("ROIselector", frame)
        	key = cv2.waitKey(1) & 0xFF

            # if the 'c' key is pressed, break from the loop
        	if key == ord("c"):
        		break

    cap.release()
    cv2.destroyAllWindows()
    return ROI

def checkROI(selectROI):
    ROI = []
    if not selectROI:
        ROI = [(0,0),(width, height)]
    ''' Select ROI '''
    if selectROI:
        print '\n Selecting ROI ...'
        ROI = ROIselector(paths)
        print "ROI: ",ROI
    return ROI

"""
Cropper and masker
"""
def maskROI(frame, ROI):
    mask = np.zeros_like(frame)
    mask[ROI[0][1]:ROI[1][1], ROI[0][0]:ROI[1][0]] = 1.
    maskedFrame = frame * mask
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
    return np.divide(frame,np.mean(frame)), np.mean(frame)

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
        frame = np.multiply(frame/np.max(frame),255).astype('uint8')
        ret, frame = cv2.threshold(frame,minThreshold,maxThreshold, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # ret, frame = cv2.threshold(frame,minThreshold,maxThreshold, cv2.THRESH_BINARY_INV)
    else:
        frame = np.multiply(frame/np.max(frame),255).astype('uint8')
        ret, frame = cv2.threshold(frame,minThreshold,maxThreshold, cv2.THRESH_BINARY_INV)
    return frame

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

def getBoundigBox(cnt):
    x,y,w,h = cv2.boundingRect(cnt)
    if (x > 2 and y > 2):
        x = x - 2
        y = y - 2
        w = w + 4
        h = h + 4
    return ((x,y),(x+w,y+h))

def getCentroid(cnt,ROI):
    M = cv2.moments(cnt)
    x = int(M['m10']/M['m00'])
    y = int(M['m01']/M['m00'])
    x, y = coordROI2Full(x,y,ROI)
    return (x,y)

def getPixelsList(cnt, width, height):
    cimg = np.zeros((height, width))
    cv2.drawContours(cimg, cnt, -1, (255,0,0), -1)
    # Access the image pixels and create a 1D numpy array then add to list
    pts = np.where(cimg == 255)
    return zip(pts[0],pts[1])

def getMiniFrame(avFrame, cnt, ROI):
    boundingBox = getBoundigBox(cnt)
    miniFrame = avFrame[boundingBox[0][1]:boundingBox[1][1], boundingBox[0][0]:boundingBox[1][0]]
    cntBB = cntROI2BoundingBox(cnt,boundingBox)
    pixelsInBB = getPixelsList(cntBB, np.abs(boundingBox[0][0]-boundingBox[1][0]), np.abs(boundingBox[0][1]-boundingBox[1][1]))
    pixelsInROI = pixelsInBB + np.asarray([boundingBox[0][0],boundingBox[0][1]])
    pixelsInFullF = pixelsInROI + np.asarray([ROI[0][0],ROI[0][1]])
    return miniFrame, pixelsInFullF

def getBlobsInfoPerFrame(avFrame, contours, ROI):
    miniFrames = []
    centroids = []
    areas = []
    pixels = []
    for cnt in contours:
        # boundigBox
        miniFrame, pixelsInFullF = getMiniFrame(avFrame, cnt, ROI)
        miniFrames.append(miniFrame)
        # centroid
        centroids.append(getCentroid(cnt,ROI))
        # area
        areas.append(cv2.contourArea(cnt))
        # pixels list
        pixels.append(pixelsInFullF)
    return miniFrames, centroids, areas, pixels

def blobExtractor(segmentedFrame, avFrame, minArea, maxArea, ROI):
    contours, hierarchy = cv2.findContours(segmentedFrame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours by size
    goodContours = filterContoursBySize(contours,minArea, maxArea)
    # Pass contours' coordinates from ROI to full size image if the ROI
    # has been cropped
    goodContoursFull = cntROI2OriginalFrame(goodContours,ROI) #only used to plot in the full frame

    ### uncomment to plot contours
    # cv2.drawContours(frame, goodContours, -1, (255,0,0), -1)
    # get contours properties
    miniFrames, centroids, areas, pixels = getBlobsInfoPerFrame(avFrame, goodContours, ROI)

    return miniFrames, centroids, areas, pixels, goodContoursFull

if __name__ == '__main__':

    # prep for args
    parser = argparse.ArgumentParser()
    videoPath = './Cafeina5peces/Caffeine5fish_20140206T122428_1.avi'
    # testPath = './test_1.avi'
    parser.add_argument('--path', default = videoPath, type = str)
    parser.add_argument('--bkg_subtraction', default = False, type = bool)
    parser.add_argument('--ROI_selection', default = True, type = bool)
    parser.add_argument('--mask_frame', default = True, type= bool)
    parser.add_argument('--Eq_image', default = False, type = bool)
    parser.add_argument('--min_th', default = 180, type = int)
    parser.add_argument('--max_th', default = 255, type = int)
    parser.add_argument('--min_area', default = 350, type = int)
    parser.add_argument('--max_area', default = 2000, type = int)
    args = parser.parse_args()

    ''' Parameters for the segmentation '''
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

    width, height = getVideoInfo(paths)
    ROI = []
    ROI = checkROI(selectROI)
    bkg = checkBkg(bkgSubstraction, paths, ROI, EQ)
    ''' Entering loop for segmentation of the video '''
    totalFrameCounter = 0
    df = pd.DataFrame(columns=('avIntensity','miniFrames', 'centroids', 'areas', 'pixels'))
    for path in paths:
        print 'Segmenting video %s' % path
        cap = cv2.VideoCapture(path)
        numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        counter = 0
        while counter < numFrame:
            counter += 1
            #Get frame from video file
            ret, frame = cap.read()
            frameToPlot = masker(frame, maskFrame, ROI, selectROI)
            #Color to gray scale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Equalize image to enhance contrast
            frame = checkEq(EQ, frame)
            # mask or crop the image
            frame = cropper(frame, ROI)
            avFrame = frame
            #Normalize each frame by its mean intensity
            frame, avIntensity = frameAverager(frame)
            # avFrame = frame
            # perform background subtraction if needed
            frame = segmentVideo(frame, minThreshold, maxThreshold, bkg, bkgSubstraction)
            # Find contours in the segmented image
            miniFrames, centroids, areas, pixels, goodContoursFull = blobExtractor(frame, avFrame, minArea, maxArea, ROI)

            # Add frame imformation to DataFrame
            df.loc[totalFrameCounter] = [avIntensity, miniFrames, centroids, areas, pixels]

            ### uncomment to plot centroids and blobs
            # cv2.drawContours(frameToPlot, goodContoursFull, -1, (255,0,0), -1)
            # for c in centroids:
            #     cv2.circle(frameToPlot,c,3,(0,0,1),4)
            # # Visualization of the process
            # cv2.imshow('ROIFrameContours',frameToPlot)

            ### Plot miniframes
            # for i, miniFrame in enumerate(miniFrames):
            #     cv2.imshow('miniFrame' + str(i), miniFrame)

            # k = cv2.waitKey(30) & 0xFF
            # if k == 27: #pres esc to quit
            #     break
            totalFrameCounter += 1

    # df.to_hdf('test.h5','df',mode='w',format='table',data_columns=True)
    df.to_pickle('testNatureMethods.pkl')



    # # nameDatabase =  ageInDpf + '_' + str(numIndiv) + 'indiv_' + str(int(numImagesPerIndiv)) + 'ImPerInd_' + preprocessing
    # f = h5py.File('../data/' + 'test' + '_%i.hdf5', driver='family')
    # grp = f.create_group("segmentation")
    #
    #
    # dset1 = grp.create_dataset("avIntensity", data.shape, dtype='f')
    # dset2 = grp.create_dataset("miniFrames", label.shape, dtype='i')
    # dset1 = grp.create_dataset("centroids", data.shape, dtype='f')
    # dset2 = grp.create_dataset("areas", label.shape, dtype='f')
    # dset2 = grp.create_dataset("pixels", label.shape, dtype='f')
    #
    #
    # dset1[...] = data
    # dset2[...] = label
    #
    # grp.attrs['originalMatPath'] = pathToDatabase
    # grp.attrs['numIndiv'] = numIndiv
    # grp.attrs['imageSize'] = imsize
    # grp.attrs['numImagesPerIndiv'] = numImagesPerIndiv
    # grp.attrs['ageInDpf'] = ageInDpf
    # grp.attrs['preprocessing'] = preprocessing
    #
    # f.close()

    cap.release()
    cv2.destroyAllWindows()
