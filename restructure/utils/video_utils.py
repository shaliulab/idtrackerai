# Import standard libraries
import os
import sys
import numpy as np
import multiprocessing

# Import third party libraries
import cv2
from joblib import Parallel, delayed

# Import application/library specifics
sys.path.append('../utils')
sys.path.append('../IdTrackerDeep')
from py_utils import *
from video import Video

"""
Compute background and threshold
"""

def computeBkgParSingleVideo(starting_frame, ending_frame, video_path, bkg):
    cap = cv2.VideoCapture(video_path)
    print 'Adding from starting frame %i to background' %starting_frame
    numFramesBkg = 0
    frameInds = range(starting_frame,ending_frame, 100)
    for ind in frameInds:
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,ind)
        ret, frameBkg = cap.read()
        if ret:
            gray = cv2.cvtColor(frameBkg, cv2.COLOR_BGR2GRAY)
            gray = np.true_divide(gray,np.mean(gray))
            bkg = bkg + gray
            numFramesBkg += 1
            
    cap.release()

    return bkg, numFramesBkg

def computeBkgParSegmVideo(video_path, bkg):
    print 'Adding video %s to background' % video_path
    cap = cv2.VideoCapture(video_path)
    counter = 0
    numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    numFramesBkg = 0
    frameInds = range(0,numFrame,100)
    for ind in frameInds:
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,ind)
        ret, frameBkg = cap.read()
        gray = cv2.cvtColor(frameBkg, cv2.COLOR_BGR2GRAY)
        gray = np.true_divide(gray,np.mean(gray))
        bkg = bkg + gray
        numFramesBkg += 1

    return bkg, numFramesBkg

def computeBkg(video):
    """Compute background to perform background subtraction.
    At this stage the function treats differently videos already segmented in
    chunks and videos composed by a single file."""
    # This holds even if we have not selected a ROI because then the ROI is
    # initialized as the full frame
    bkg = np.zeros((video._height, video._width))
    num_cores = multiprocessing.cpu_count()
    # num_cores = 1
    if video._paths_to_video_segments is None: # one single video
        print 'one single video, computing bkg in parallel from single video'
        output = Parallel(n_jobs=num_cores)(delayed(computeBkgParSingleVideo)(starting_frame, ending_frame, video.video_path, bkg) for (starting_frame, ending_frame) in video._episodes_start_end)
    else: # multiple segments video
        output = Parallel(n_jobs=num_cores)(delayed(computeBkgParSegmVideo)(videoPath,bkg) for videoPath in video._paths_to_video_segments)

    partialBkg = [bkg for (bkg,_) in output]
    totNumFrame = np.sum([numFrame for (_,numFrame) in output])
    bkg = np.sum(np.asarray(partialBkg),axis=0)
    bkg = np.true_divide(bkg, totNumFrame)
    return bkg.astype('float32')

def checkBkg(video, old_video, usePreviousBkg):
    bkg = None

    if video.subtract_bkg:
        if usePreviousBkg and old_video.bkg is not None:
            bkg = old_video.bkg
        else:
            bkg = computeBkg(video)

    return bkg

def segmentVideo(frame, minThreshold, maxThreshold, bkg, ROI, useBkg):
    """Applies background substraction if requested and thresholds image
    :param frame: original frame normalised by the mean. Must be float32
    :param minThreshold: minimum intensity threshold (0-255)
    :param maxThreshold: maximum intensity threshold (0-255)
    :param bkg: background frame (normalised by mean???). Must be float32
    :param mask: boolean mask of region of interest where thresholding is performed. uint8, 255 valid, 0 invalid.
    :param useBkg: boolean determining if background subtraction is performed
    """
    if useBkg:
        frame = cv2.absdiff(bkg,frame) #only step where frame normalization is important, because the background is normalised

    frameSegmented = cv2.inRange(frame * (255.0/frame.max()), minThreshold, maxThreshold) #output: 255 in range, else 0
    frameSegmentedMasked = cv2.bitwise_and(frameSegmented,frameSegmented, mask=ROI) #Applying the mask
    return frameSegmentedMasked

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

def full2BoundingBox(point, boundingBox):
    return point - np.asarray([boundingBox[0][0],boundingBox[0][1]])

def cntBB2Full(cnt,boundingBox):
    return cnt + np.asarray([boundingBox[0][0],boundingBox[0][1]])

def getBoundigBox(cnt, width, height):
    x,y,w,h = cv2.boundingRect(cnt)
    n = 25
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
    #FIXME TO BE REMOVED
    frame = np.zeros((500,500)).astype('uint8')
    cv2.drawContours(miniFrame, [cntBB], -1, color=255, thickness = -1)
    # Access the image pixels and create a 1D numpy array then add to list
    bkgSample = miniFrame[np.where(miniFrame != 255)]
    return bkgSample

def getMiniFrame(frame, cnt, height, width):
    boundingBox = getBoundigBox(cnt, width, height)
    miniFrame = frame[boundingBox[0][1]:boundingBox[1][1], boundingBox[0][0]:boundingBox[1][0]]
    cntBB = cnt2BoundingBox(cnt,boundingBox)
    #miniFrameBkg = miniFrame.copy()
    # bkgSample = sampleBkg(cntBB, miniFrameBkg)
    pixelsInBB = getPixelsList(cntBB, np.abs(boundingBox[0][0] - boundingBox[1][0]), np.abs(boundingBox[0][1] - boundingBox[1][1]))
    pixelsInFullF = pixelsInBB + np.asarray([boundingBox[0][1], boundingBox[0][0]])
    pixelsInFullFF = np.ravel_multi_index([pixelsInFullF[:,0], pixelsInFullF[:,1]],(height,width))
    return boundingBox, miniFrame, pixelsInFullFF

def getBlobsInfoPerFrame(frame, contours, height, width):
    boundingBoxes = []
    miniFrames = []
    centroids = []
    areas = []
    pixels = []

    for i, cnt in enumerate(contours):
        boundingBox, miniFrame, pixelsInFullF = getMiniFrame(frame, cnt, height, width)
        #bounding boxes
        boundingBoxes.append(boundingBox)
        # miniframes
        miniFrames.append(miniFrame)
        # centroids
        centroids.append(getCentroid(cnt))
        # areas
        areas.append(cv2.contourArea(cnt))
        # pixels lists
        pixels.append(pixelsInFullF)

    return boundingBoxes, miniFrames, centroids, areas, pixels

def blobExtractor(segmentedFrame, frame, minArea, maxArea, height, width):
    contours, hierarchy = cv2.findContours(segmentedFrame,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # Filter contours by size
    goodContoursFull = filterContoursBySize(contours,minArea, maxArea)
    # get contours properties
    boundingBoxes, miniFrames, centroids, areas, pixels = getBlobsInfoPerFrame(frame, goodContoursFull, height, width)

    return boundingBoxes, miniFrames, centroids, areas, pixels, goodContoursFull
