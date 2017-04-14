# Import standard libraries
import os
import sys
import numpy as np
import multiprocessing

# Import third party libraries
import cv2
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

# Import application/library specifics
sys.path.append('IdTrackerDeep/utils')
from py_utils import *

def computeBkgPar(startingFrame,endingFrame,videoPath,bkg,framesPerSegment):
    # Open cap
    cap = cv2.VideoCapture(videoPath)
    print 'Adding from starting frame %i to background' %startingFrame
    numFramesBkg = 0
    frameInds = range(startingFrame,endingFrame,10)
    for ind in frameInds:
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,ind)
        ret, frameBkg = cap.read()
        gray = cv2.cvtColor(frameBkg, cv2.COLOR_BGR2GRAY)
        # gray = checkEq(EQ, gray)
        gray = np.true_divide(gray,np.mean(gray))
        bkg = bkg + gray
        numFramesBkg += 1
    cap.release()

    return bkg, numFramesBkg

def computeBkg(videoPath, width, height, numFrames):
    # This holds even if we have not selected a ROI because then the ROI is
    # initialized as the full frame
    bkg = np.zeros((height,width))

    num_cores = multiprocessing.cpu_count()
    # num_cores = 1
    framesPerSegment = 500
    startingFrames = range(0,numFrames-1,framesPerSegment)
    endingFrames = range(framesPerSegment-1,numFrames-1,framesPerSegment)
    endingFrames.append(numFrames-1)
    print startingFrames
    output = Parallel(n_jobs=num_cores)(delayed(computeBkgPar)(startingFrame,endingFrames,videoPath,bkg,framesPerSegment) for startingFrame, endingFrames in zip(startingFrames,endingFrames))
    partialBkg = [bkg for (bkg,_) in output]
    totNumFrame = np.sum([numFrame for (_,numFrame) in output])
    bkg = np.sum(np.asarray(partialBkg),axis=0)
    bkg = np.true_divide(bkg, totNumFrame)
    return bkg

def computeBkgParOLD(videoPath,bkg,EQ):
    print 'Adding video %s to background' % videoPath
    cap = cv2.VideoCapture(videoPath)
    counter = 0
    numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    numFramesBkg = 0
    frameInds = range(0,numFrame,10)
    for ind in frameInds:
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,ind)
        ret, frameBkg = cap.read()
        gray = cv2.cvtColor(frameBkg, cv2.COLOR_BGR2GRAY)
        # gray = checkEq(EQ, gray)
        gray = np.true_divide(gray,np.mean(gray))
        bkg = bkg + gray
        numFramesBkg += 1

    return bkg, numFramesBkg

def computeBkgOLD(videoPaths, EQ, width, height):
    # This holds even if we have not selected a ROI because then the ROI is
    # initialized as the full frame
    bkg = np.zeros((height,width))

    num_cores = multiprocessing.cpu_count()
    # num_cores = 1
    output = Parallel(n_jobs=num_cores)(delayed(computeBkgParOLD)(videoPath,bkg,EQ) for videoPath in videoPaths)
    partialBkg = [bkg for (bkg,_) in output]
    totNumFrame = np.sum([numFrame for (_,numFrame) in output])
    bkg = np.sum(np.asarray(partialBkg),axis=0)
    bkg = np.true_divide(bkg, totNumFrame)
    return bkg
#
# videoPath = '/home/chaos/Desktop/IdTrackerDeep/videos/fish5-INDP2016/fullVideo/GroupFish2017-01-24T09_40_48.avi'
# cap = cv2.VideoCapture(videoPath)
# numFrames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
# cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,0)
# flag, frame = cap.read()
# cap.release()
# height = frame.shape[0]
# width = frame.shape[1]
# bkg = computeBkg(videoPath, width, height, numFrames)

videoPaths = scanFolder('/home/chaos/Desktop/IdTrackerDeep/videos/fish5-INDP2016/GroupFish2017-01-24T09_40_48_1.avi')
cap = cv2.VideoCapture(videoPaths[0])
numFrames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,0)
flag, frame = cap.read()
cap.release()
height = frame.shape[0]
width = frame.shape[1]
bkg = computeBkgOLD(videoPaths, 0, width, height)
