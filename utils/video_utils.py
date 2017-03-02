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

"""
Split video
"""
def splitVideo(videoPath):
    # load video
    cap = cv2.VideoCapture(videoPath)
    print 'opening cap ', cap.isOpened()
    # retrieve info
    video = os.path.basename(videoPath)
    filename, extension = os.path.splitext(video)
    folder = os.path.dirname(videoPath)
    fps = cv2.cv.CV_CAP_PROP_FPS
    print 'fps ',fps
    fourcc = cv2.cv.CV_CAP_PROP_FOURCC
    # print fourcc
    # fourcc = cv2.cv.CV_FOURCC(*'MP4A')
    # fourcc = cv2.cv.CV_FOURCC(*'X264')
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    print 'numFrames, ', numFrame
    currentFrame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    # chunk it
    numSeg = 1
    name = folder +'/'+ filename + '_' + str(numSeg) + extension
    out = cv2.VideoWriter(name, fourcc, fps, (width, height))
    while currentFrame < numFrame:
        print currentFrame, numFrame
        ret = cap.grab()
        # print 'reading ', ret
        frame = cap.retrieve()
        # print 'frame', frame
        if ret:
            print 'ret', ret
            if currentFrame % 500 == 0:
                name = folder +'/'+ filename + '_' + str(numSeg) + extension
                numSeg += 1
                out.release()
                out = cv2.VideoWriter(name, fourcc, fps, (width, height))
                print 'saving ', name

            out.write(frame[1])
            currentFrame += 1
            ## Uncomment to show the video
            # cv2.imshow('IdPlayer',frame)
            # m = cv2.waitKey(30) & 0xFF
            # if m == 27: #pres esc to quit
            #     break
        else:
            raise ValueError('Something went wrong when loading the video')

    cap.release()


# splitVideo('/media/chaos/idZebLib_TU31012017/Group_1_2/full video/video_02174359.avi' )
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

"""
Compute background and threshold
"""
def computeBkgPar(videoPath,bkg,EQ):
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
        # gray = checkEq(EQ, gray)
        gray = np.true_divide(gray,np.mean(gray))
        bkg = bkg + gray
        numFramesBkg += 1

    return bkg, numFramesBkg

def computeBkg(videoPaths, EQ, width, height):
    # This holds even if we have not selected a ROI because then the ROI is
    # initialized as the full frame
    bkg = np.zeros((height,width))

    num_cores = multiprocessing.cpu_count()
    # num_cores = 1
    output = Parallel(n_jobs=num_cores)(delayed(computeBkgPar)(videoPath,bkg,EQ) for videoPath in videoPaths)
    partialBkg = [bkg for (bkg,_) in output]
    totNumFrame = np.sum([numFrame for (_,numFrame) in output])
    bkg = np.sum(np.asarray(partialBkg),axis=0)
    bkg = np.true_divide(bkg, totNumFrame)
    return bkg

def checkBkg(videoPaths, useBkg, usePreviousBkg, EQ, width, height):
    ''' Compute Bkg ''' ###TODO This can be done in a smarter way...
    videoPath = videoPaths[0]
    if useBkg:
        if usePreviousBkg:
            bkg = loadFile(videoPath, 'bkg',hdfpkl='pkl')
        else:
            bkg = computeBkg(videoPaths, EQ, width, height)
            saveFile(videoPath, bkg, 'bkg', hdfpkl='pkl')
        return bkg
    else:
        return None

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

    # compute the average frame
    frame = np.true_divide(frame,np.mean(frame))
    if useBkg:

        frameSubtracted = uint8caster(np.abs(np.subtract(bkg,frame)))
        frameSubtractedMasked = cv2.addWeighted(frameSubtracted,1,mask,1,0)
        ### Uncomment to plot
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
    contours, hierarchy = cv2.findContours(segmentedFrame,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # Filter contours by size
    goodContoursFull = filterContoursBySize(contours,minArea, maxArea)
    # get contours properties
    boundingBoxes, miniFrames, centroids, areas, pixels, bkgSamples = getBlobsInfoPerFrame(frame, goodContoursFull, height, width)

    return boundingBoxes, miniFrames, centroids, areas, pixels, goodContoursFull, bkgSamples
