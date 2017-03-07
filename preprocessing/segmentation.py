from __future__ import division

# Import standard libraries
import os
import sys
import numpy as np
import multiprocessing

# Import third party libraries
import cv2
import pandas as pd
from joblib import Parallel, delayed
import gc

# Import application/library specifics
sys.path.append('IdTrackerDeep/utils')

from py_utils import flatten, loadFile
from video_utils import saveFile, collectAndSaveVideoInfo, generateVideoTOC, segmentVideo, getVideoInfo, blobExtractor, getNumFrame

def segmentAndSave(path, height, width, mask, useBkg, bkg, EQ, minThreshold, maxThreshold, minArea, maxArea, segmFrameInd = None,framesPerSegment=None):
    # locally called

    cap = cv2.VideoCapture(path)
    if segmFrameInd == None:
        print 'Segmenting video %s' % path
        video = os.path.basename(path)
        filename, extension = os.path.splitext(video)
        numSegment = int(filename.split('_')[-1])
        numFrames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        counter = 0
    else:
        numSegment = int(segmFrameInd[0]/framesPerSegment) + 1
        print 'Segment video %s from frame %i to frame %i (segment %i)' %(path, segmFrameInd[0], segmFrameInd[1], numSegment)
        numFrames = segmFrameInd[1] - segmFrameInd[0] + 1
        counter = 0
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,segmFrameInd[0])
    df = pd.DataFrame(columns=('avIntensity', 'boundingBoxes','miniFrames', 'contours', 'centroids', 'areas', 'pixels', 'numberOfBlobs', 'bkgSamples'))
    maxNumBlobs = 0
    while counter < numFrames:
        #Get frame from video file
        ret, frame = cap.read()
        #Color to gray scale
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # origFrame = frameGray.copy()
        # print avIntensity
        avIntensity = np.float32(np.mean(frameGray))
        segmentedFrame = segmentVideo(frameGray/avIntensity, minThreshold, maxThreshold, bkg, mask, useBkg)
        # segmentedFrameCopy = segmentedFrame.copy()
        # Find contours in the segmented image
        boundingBoxes, miniFrames, centroids, areas, pixels, goodContoursFull, bkgSamples = blobExtractor(segmentedFrame, frameGray, minArea, maxArea, height, width)
        if len(centroids) > maxNumBlobs:
            maxNumBlobs = len(centroids)
        ### UNCOMMENT TO PLOT ##################################################
        # cv2.drawContours(origFrame,goodContoursFull,-1,color=(255,0,0),thickness=-1)
        # cv2.imshow('checkcoord', origFrame)
        # k = cv2.waitKey(100) & 0xFF
        # if k == 27: #pres esc to quit
        #     break
        ########################################################################

        # Add frame imformation to DataFrame
        df.loc[counter] = [avIntensity, boundingBoxes, miniFrames, goodContoursFull, centroids, areas, pixels, len(centroids), bkgSamples]
        counter += 1

    cap.release()
    cv2.destroyAllWindows()
    saveFile(path, df, 'segment',nSegment = str(numSegment))
    gc.collect()

    return np.multiply(numSegment,np.ones(numFrames)).astype('int').tolist(), np.arange(numFrames).tolist(), maxNumBlobs

def segment(videoPaths,preprocParams, mask, centers, useBkg, bkg, EQ):
    # this func is called from idTrackerDeepGUI
    numAnimals = preprocParams['numAnimals']
    minThreshold = preprocParams['minThreshold']
    maxThreshold = preprocParams['maxThreshold']
    minArea = preprocParams['minArea']
    maxArea = preprocParams['maxArea']

    width, height = getVideoInfo(videoPaths)


    print 'videoPaths here, ', videoPaths
    if len(videoPaths) == 1:
        print '**************************************'
        print 'There is only one path, segmenting by frame indices'
        print '**************************************'
        '''Define list of starting and ending frames'''
        frameIndices = loadFile(videoPaths[0], 'frameIndices')
        framesPerSegment = len(np.where(frameIndices.loc[:,'segment'] == 1)[0])
        segments = np.unique(frameIndices.loc[:,'segment'])
        startingFrames = [frameIndices[frameIndices['segment']==seg].index[0] for seg in segments]
        endingFrames = [frameIndices[frameIndices['segment']==seg].index[-1] for seg in segments]
        segmFramesIndices = zip(startingFrames,endingFrames)
        ''' Spliting frames list into sublists '''
        num_cores = multiprocessing.cpu_count()
        # num_cores = 1
        segmFramesIndicesSubLists = [segmFramesIndices[i:i+num_cores] for i in range(0,len(segmFramesIndices),num_cores)]
        print 'Entering to the parallel loop...\n'
        allSegments = []
        numBlobs = []
        path = videoPaths[0]
        for segmFramesIndicesSubList in segmFramesIndicesSubLists:
            OupPutParallel = Parallel(n_jobs=num_cores)(delayed(segmentAndSave)(path, height, width, mask, useBkg, bkg, EQ, minThreshold, maxThreshold, minArea, maxArea, segmFrameInd, framesPerSegment) for segmFrameInd in segmFramesIndicesSubList)
            allSegmentsSubList = [(out[0],out[1]) for out in OupPutParallel]
            allSegments.append(allSegmentsSubList)
            numBlobs.append([out[2] for out in OupPutParallel])

    else:
        ''' splitting videoPaths list into sublists '''
        num_cores = multiprocessing.cpu_count()
        #num_cores = 1
        pathsSubLists = [videoPaths[i:i+num_cores] for i in range(0,len(videoPaths),num_cores)]
        ''' Entering loop for segmentation of the video '''
        print 'Entering to the parallel loop...\n'
        allSegments = []
        numBlobs = []
        for pathsSubList in pathsSubLists:
            OupPutParallel = Parallel(n_jobs=num_cores)(delayed(segmentAndSave)(path, height, width, mask, useBkg, bkg, EQ, minThreshold, maxThreshold, minArea, maxArea) for path in pathsSubList)
            allSegmentsSubList = [(out[0],out[1]) for out in OupPutParallel]
            allSegments.append(allSegmentsSubList)
            numBlobs.append([out[2] for out in OupPutParallel])

    allSegments = flatten(allSegments)
    maxNumBlobs = max(flatten(numBlobs))
    # OupPutParallel = Parallel(n_jobs=num_cores)(delayed(segmentAndSave)(path, height, width, mask, useBkg, bkg, EQ, minThreshold, maxThreshold, minArea, maxArea) for path in videoPaths)
    # allSegments = [(out[0],out[1]) for out in OupPutParallel]
    # # print allSegments
    # maxNumBlobs = max([out[2] for out in OupPutParallel])
    # # print maxNumBlobs
    allSegments = sorted(allSegments, key=lambda x: x[0][0])
    numFrames = generateVideoTOC(allSegments, videoPaths[0])
    collectAndSaveVideoInfo(videoPaths[0], numFrames, height, width, numAnimals, num_cores, minThreshold,maxThreshold,maxArea,maxNumBlobs)
