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
from py_utils import *

# """
# Split video
# """
# def splitVideo(videoPath):
#     # load video
#     cap = cv2.VideoCapture(videoPath)
#     print 'opening cap ', cap.isOpened()
#     # retrieve info
#     video = os.path.basename(videoPath)
#     filename, extension = os.path.splitext(video)
#     folder = os.path.dirname(videoPath)
#     fps = cv2.cv.CV_CAP_PROP_FPS
#     print 'fps ',fps
#     fourcc = cv2.cv.CV_CAP_PROP_FOURCC
#     print fourcc
#     fourcc = cv2.cv.CV_FOURCC(*'MP4A')
#     # fourcc = cv2.cv.CV_FOURCC(*'X264')
#     # fourcc = cv2.cv.CV_FOURCC(*'XVID')
#     width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
#     numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
#     print 'numFrames, ', numFrame
#     currentFrame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
#     # chunk it
#     numSeg = 1
#     name = folder +'/'+ filename + '_' + str(numSeg) + extension
#     out = cv2.VideoWriter(name, fourcc, fps, (width, height))
#     while currentFrame < numFrame:
#         print currentFrame, numFrame
#         ret = cap.grab()
#         # print 'reading ', ret
#         frame = cap.retrieve()
#         # print 'frame', frame
#         if ret:
#             print 'ret', ret
#             if currentFrame % 500 == 0:
#                 name = folder +'/'+ filename + '_' + str(numSeg) + extension
#                 numSeg += 1
#                 out.release()
#                 out = cv2.VideoWriter(name, fourcc, fps, (width, height))
#                 print 'saving ', name
#
#             out.write(frame[1])
#             currentFrame += 1
#             ## Uncomment to show the video
#             # cv2.imshow('IdPlayer',frame)
#             # m = cv2.waitKey(30) & 0xFF
#             # if m == 27: #pres esc to quit
#             #     break
#         else:
#             raise ValueError('Something went wrong when loading the video')
#
#     cap.release()

"""
Get general information from video
"""
def getVideoInfo(videoPaths):
    if len(videoPaths) == 1:
        videoPath = videoPaths
    elif len(videoPaths) > 1:
        videoPath = videoPaths[0]
    else:
        raise ValueError('the videoPath (or list of videoPaths) seems to be empty')
    cap = cv2.VideoCapture(videoPaths[0])
    width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    return width, height

def getNumFrame(videoPath):
    cap = cv2.VideoCapture(videoPath)
    return int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

def collectAndSaveVideoInfo(videoPath, numFrames, height, width, numAnimals, numCores, minThreshold,maxThreshold,maxArea,maxNumBlobs):
    """
    saves general info about the video in a pickle (_videoinfo.pkl)
    """
    videoInfo = {
        'path': videoPath,
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
    print 'videoInfo, ', videoInfo
    saveFile(videoPath, videoInfo, 'videoInfo',hdfpkl='pkl')

def generateVideoTOC(allSegments, videoPath):
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
    saveFile(videoPath, videoTOC, 'frameIndices')
    return numFrames

def getSegmPaths(videoPaths,framesPerSegment=500):
    folder = os.path.dirname(videoPaths[0])
    print folder
    if len(videoPaths) == 1: # The video is in a single filename
        numFrames = getNumFrame(videoPaths[0])
        numSegments = np.ceil(np.true_divide(numFrames,framesPerSegment)).astype('int')
        framesTOC = [frame % framesPerSegment for frame in range(numFrames)]
        segmentsTOC = [frame/framesPerSegment+1 for frame in range(numFrames)]
        segmPaths = [folder + '/segm_%s.plh' %str(seg+1) for seg in range(numSegments)]
    else:
        framesTOC = [range(0,getNumFrame(videoPath)) for videoPath in videoPaths]
        segmentsTOC = [[seg+1 for i in range(getNumFrame(videoPath))] for seg, videoPath in enumerate(videoPaths)]
        framesTOC = flatten(framesTOC)
        segmentsTOC = flatten(segmentsTOC)
        segmPaths = videoPaths

    frameIndices =  pd.DataFrame({'segment':segmentsTOC, 'frame': framesTOC})
    saveFile(videoPaths[0], frameIndices, 'frameIndices')
    return frameIndices, segmPaths

"""
Compute background and threshold
"""
# def computeBkgPar(videoPath,bkg,EQ):
#     print 'Adding video %s to background' % videoPath
#     cap = cv2.VideoCapture(videoPath)
#     numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
#     numFramesBkg = 0
#     frameInds = range(0,numFrame,100)
#     for ind in frameInds:
#         cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,ind)
#         ret, frameBkg = cap.read()
#         gray = cv2.cvtColor(frameBkg, cv2.COLOR_BGR2GRAY)
#         # gray = checkEq(EQ, gray)
#         gray = np.true_divide(gray,np.float32(np.mean(gray)))
#         bkg = bkg + gray
#         numFramesBkg += 1
#
#     return bkg, numFramesBkg
#
# def computeBkg(videoPaths, EQ, width, height):
#     # This holds even if we have not selected a ROI because then the ROI is
#     # initialized as the full frame
#     bkg = np.zeros((height,width),dtype=np.float32)
#
#     num_cores = multiprocessing.cpu_count()
#     # num_cores = 1
#     output = Parallel(n_jobs=num_cores)(delayed(computeBkgPar)(videoPath,bkg,EQ) for videoPath in videoPaths)
#     partialBkg = [bkg for (bkg,_) in output]
#     totNumFrame = np.sum([numFrame for (_,numFrame) in output])
#     bkg = np.sum(np.asarray(partialBkg),axis=0)
#     bkg = np.true_divide(bkg, totNumFrame)
#     return bkg

def computeBkgParSingleVideo(startingFrame,endingFrame,videoPath,bkg,framesPerSegment):
    # Open cap
    cap = cv2.VideoCapture(videoPath)
    print 'Adding from starting frame %i to background' %startingFrame
    numFramesBkg = 0
    frameInds = range(startingFrame,endingFrame,100)
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

def computeBkgParSegmVideo(videoPath,bkg):
    print 'Adding video %s to background' % videoPath
    cap = cv2.VideoCapture(videoPath)
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

def computeBkg(videoPaths, width, height):
    # This holds even if we have not selected a ROI because then the ROI is
    # initialized as the full frame
    bkg = np.zeros((height,width))
    num_cores = multiprocessing.cpu_count()
    # num_cores = 1
    if len(videoPaths) == 1: # one single video
        print 'one single video, computing bkg in parallel from single video'
        videoPath = videoPaths[0]
        frameIndices = loadFile(videoPaths[0], 'frameIndices')
        framesPerSegment = len(np.where(frameIndices.loc[:,'segment'] == 1)[0])
        segments = np.unique(frameIndices.loc[:,'segment'])
        startingFrames = [frameIndices[frameIndices['segment']==seg].index[0] for seg in segments]
        endingFrames = [frameIndices[frameIndices['segment']==seg].index[-1] for seg in segments]
        output = Parallel(n_jobs=num_cores)(delayed(computeBkgParSingleVideo)(startingFrame,endingFrames,videoPath,bkg,framesPerSegment) for startingFrame, endingFrames in zip(startingFrames,endingFrames))
    else: # multiple segments video
        output = Parallel(n_jobs=num_cores)(delayed(computeBkgParSegmVideo)(videoPath,bkg) for videoPath in videoPaths)

    partialBkg = [bkg for (bkg,_) in output]
    totNumFrame = np.sum([numFrame for (_,numFrame) in output])
    bkg = np.sum(np.asarray(partialBkg),axis=0)
    bkg = np.true_divide(bkg, totNumFrame)
    return bkg

def checkBkg(videoPaths, useBkg, usePreviousBkg, EQ, width, height):
    videoPath = videoPaths[0]
    if useBkg:
        if usePreviousBkg:
            bkg = loadFile(videoPath, 'bkg',hdfpkl='pkl')
        else:
            bkg = computeBkg(videoPaths, width, height)
            saveFile(videoPath, bkg, 'bkg', hdfpkl='pkl')
        return bkg.astype(np.float32)
    else:
        return None

def getSegmPaths(videoPaths,framesPerSegment=500):
    folder = os.path.dirname(videoPaths[0])
    print folder
    if len(videoPaths) == 1: # The video is in a single filename
        numFrames = getNumFrame(videoPaths[0])
        numSegments = np.ceil(np.true_divide(numFrames,framesPerSegment)).astype('int')
        framesTOC = [frame % framesPerSegment for frame in range(numFrames)]
        segmentsTOC = [frame/framesPerSegment+1 for frame in range(numFrames)]
        segmPaths = [folder + '/segm_%s.plh' %str(seg+1) for seg in range(numSegments)]
    else:
        framesTOC = [range(0,getNumFrame(videoPath)) for videoPath in videoPaths]
        segmentsTOC = [[seg+1 for i in range(getNumFrame(videoPath))] for seg, videoPath in enumerate(videoPaths)]
        framesTOC = flatten(framesTOC)
        segmentsTOC = flatten(segmentsTOC)
        segmPaths = videoPaths

    frameIndices =  pd.DataFrame({'segment':segmentsTOC, 'frame': framesTOC})
    saveFile(videoPaths[0], frameIndices, 'frameIndices')
    return frameIndices, segmPaths
