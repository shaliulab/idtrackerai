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
Compute background and threshold
"""
def computeBkg(paths, ROI, EQ):
    bkg = np.zeros(
    (
    np.abs(np.subtract(ROI[0][1],ROI[1][1])),
    np.abs(np.subtract(ROI[0][0],ROI[1][0]))
    )
    )
    totNumFrame = 0
    for path in paths:
        cap = cv2.VideoCapture(path)
        numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        heigth = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        counter = 0
        totNumFrame = totNumFrame + numFrame
        while counter < numFrame:
            counter += 1;
            ret, frame = cap.read()
            frame = frame[ROI[0][1]:ROI[1][1], ROI[0][0]:ROI[1][0]]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = checkEq(EQ, gray)
            gray = frameAverager(gray)
            bkg = bkg + gray
    bkg = np.true_divide(bkg, totNumFrame)
    # bkg is the backgorund computed by summing all the averaged frames
    # of the video and dividing by the number of frames in the video.
    return bkg

def bkgSubtraction(frame, bkg):
    return np.abs(np.subtract(frame,bkg))

def frameAverager(frame):
    return np.divide(frame,np.mean(frame))

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

def getBoundigBox(cnt):
    x,y,w,h = cv2.boundingRect(cnt)
    if (x > 2 and y > 2):
        x = x - 2
        y = y - 2
        w = w + 4
        h = h + 4
    return ((x,y),(x+w,y+h))

def getCentroid(cnt):
    M = cv2.moments(cnt)
    x = int(M['m10']/M['m00'])
    y = int(M['m01']/M['m00'])
    return (x,y)

def getPixelsList(cnt, width, height):
    cimg = np.zeros((height, width))
    cv2.drawContours(cimg, cnt, -1, (255,0,0), -1)
    # Access the image pixels and create a 1D numpy array then add to list
    pts = np.where(cimg == 255)
    return zip(pts[0],pts[1])

def getBlobsInfoPerFrame(blobs, width, height):
    boundingBoxes = []
    centroids = []
    areas = []
    pixels = []
    for blob in blobs:
        # boundigBox
        boundingBoxes.append(getBoundigBox(blob))
        # centroid
        centroids.append(getCentroid(blob))
        # area
        areas.append(cv2.contourArea(blob))
        # pixels list
        pixels.append(getPixelsList(blob,width, height))
    return boundingBoxes, centroids, areas, pixels

def ROIselector(paths):

    def ROIselectorCallBack(event, x, y, flags, params):
    	# grab references to the global variables
    	global ROI, selectROI

    	# if the left mouse button was clicked, record the starting
    	# (x, y) coordinates and indicate that cropping is being
    	# performed
    	if event == cv2.EVENT_LBUTTONDOWN:
    		ROI = [(x, y)]
    		selectROI = True

    	# check to see if the left mouse button was released
    	elif event == cv2.EVENT_LBUTTONUP:
    		# record the ending (x, y) coordinates and indicate that
    		# the cropping operation is finished
    		ROI.append((x, y))
    		selectROI = False

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

def scanFolder(path):
    paths = [path]
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    folder = os.path.dirname(path)
    # maybe write check on video extension supported by opencv2
    if filename[-1] == '0':
        paths = glob.glob(folder + "/" + filename[:-1] + "*" + extension)
    return paths

def checkROI(selectROI):
    ROI = []
    if not selectROI:
        ROI = [(0,0),(width, heigth)]
    print ROI
    ''' Select ROI '''
    if selectROI:
        print '\n Selecting ROI ...'
        ROI = ROIselector(paths)
        print "ROI: ",ROI
    return ROI

def checkBkg(bkgSubstraction, paths, ROI, EQ):
    ''' Compute Bkg '''
    if bkgSubstraction:
        print '\n Computing background ...\n'
        bkg = computeBkg(paths, ROI, EQ)

        return bkg
    else:
        return None

def checkEq(EQ, frame):
    if EQ:
        # Equalize image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))
        frame = clahe.apply(frame)
    return frame

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

def getVideoInfo(paths):
    cap = cv2.VideoCapture(paths[0])
    width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    heigth = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    return width, heigth

if __name__ == '__main__':

    # prep for args
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./test0.avi', type = str)
    parser.add_argument('--bkg_subtraction', default=True, type = bool)
    parser.add_argument('--ROI_selection', default=False, type = bool)
    parser.add_argument('--Eq_image', default=False, type = bool)
    parser.add_argument('--min_th', default=180, type = int)
    parser.add_argument('--max_th', default=255, type = int)
    parser.add_argument('--min_area', default=50, type = int)
    parser.add_argument('--max_area', default=500, type = int)
    args = parser.parse_args()

    ''' Parameters for the segmentation '''
    bkgSubstraction = args.bkg_subtraction
    selectROI = args.ROI_selection
    EQ = args.Eq_image
    minThreshold = args.min_th
    maxThreshold = args.max_th
    minArea = args.min_area # in pixels
    maxArea = args.max_area # in pixels

    ''' Path to video/s '''
    paths = scanFolder(args.path)
    width, heigth = getVideoInfo(paths)
    ROI = []
    ROI = checkROI(selectROI)
    bkg = checkBkg(bkgSubstraction, paths, ROI, EQ)
    ''' Entering loop for segmentation of the video '''
    for path in paths:
        print 'Segmenting video %s' % path
        cap = cv2.VideoCapture(path)
        numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        counter = 0
        while counter < numFrame:
            counter += 1
            #Get frame from video file
            ret, frame = cap.read()
            originalFrame = frame
            #Color to gray scale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Equalize image to enhance contrast
            frame = checkEq(EQ, frame)
            #crop according to ROI
            frame = frame[ROI[0][1]:ROI[1][1], ROI[0][0]:ROI[1][0]]
            #Normalize each frame by its mean intensity
            frame = np.divide(frame,np.mean(frame))
            # perform backgourn subtraction if needed
            frame = segmentVideo(frame, minThreshold, maxThreshold, bkg, bkgSubstraction)
            # Find contours in the segmented image
            contours, hierarchy = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            # Filter contours by size
            goodContours = filterContoursBySize(contours,minArea, maxArea)
            # Pass contours' coordinates from ROI to full size image
            goodContoursFull = cntROI2OriginalFrame(goodContours,ROI)
            ### uncomment to plot contours
            cv2.drawContours(originalFrame, goodContoursFull, -1, (255,0,0), -1)
            # get contours properties
            boundingBoxes, centroids, areas, pixels = getBlobsInfoPerFrame(goodContoursFull, width, heigth)
            ### uncomment to plot centroids
            for c in centroids:
                cv2.circle(originalFrame,c,3,(0,0,1),4)
            # Visualization of the process
            cv2.imshow('ROIFrameContours',originalFrame)
            k = cv2.waitKey(30) & 0xFF
            if k == 27: #pres esc to quit
                break
    cap.release()
    cv2.destroyAllWindows()
