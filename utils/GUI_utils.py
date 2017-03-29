# Import standard libraries
import os
import sys
import numpy as np

# Import third party libraries
import cv2
from pylab import *
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
sns.set(style="white", context="talk")
import pyautogui

# Import application/library specifics
sys.path.append('IdTrackerDeep/utils')
sys.path.append('IdTrackerDeep/preprocessing')

from segmentation import *
from fragmentation import *
from get_portraits import *
from video_utils import *
from py_utils import *

''' ****************************************************************************
ROI selector GUI
*****************************************************************************'''
def getMask(im):
    """Returns a uint8 mask following openCV convention
    as used in segmentation (0 invalid, 255 valid)
    """
    def line_select_callback(eclick, erelease):
        'eclick and erelease are the press and release events'
        global coord
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        coord = (x1, y1, x2, y2)

    def toggle_selector(event):
        global coord
        # if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        #     print(' RectangleSelector deactivated.')
        #     toggle_selector.RS.set_active(False)
        #
        # if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        #     print(' RectangleSelector activated.')
        #     toggle_selector.RS.set_active(True)

        if event.key == 'r':
            coordinates.append(coord)
            c = coordinates[-1]
            p = patches.Rectangle((c[0], c[1]), c[2]-c[0], c[3]-c[1],alpha=0.4)
            p1 = patches.Rectangle((c[0], c[1]), c[2]-c[0], c[3]-c[1],facecolor="white")
            # mask_ax.add_patch(p1)
            current_ax.add_patch(p)
            plt.draw()
            coord = np.asarray(c).astype('int')
            cv2.rectangle(maskout,(coord[0],coord[1]),(coord[2],coord[3]),255,-1)
            centers.append(None)

        if event.key == 'c':
            coordinates.append(coord)
            c = coordinates[-1]
            w = c[2]-c[0]
            h = c[3]-c[1]
            p = patches.Ellipse((c[0]+w/2, c[1]+h/2), w, h, angle=0.0, alpha=0.4)
            p1 = patches.Ellipse((c[0]+w/2, c[1]+h/2), w, h, angle=0.0,facecolor="white")
            current_ax.add_patch(p)
            # mask_ax.add_patch(p1)
            plt.draw()
            coord = np.asarray(c).astype('int')
            center = ((coord[2]+coord[0])/2,(coord[3]+coord[1])/2)
            angle = 90
            axes = tuple(sorted(((coord[2]-coord[0])/2,(coord[3]-coord[1])/2)))
            cv2.ellipse(maskout,center,axes,angle,0,360,255,-1)
            centers.append(center)

    coordinates = []
    centers = []
    w, h = pyautogui.size()

    fig, ax_arr = plt.subplots(1,1, figsize=(w/96,h/96))
    fig.suptitle('Select mask')
    current_ax = ax_arr
    current_ax.set_title('Drag on the image, adjust,\n press r, or c to get a rectangular or a circular ROI')
    sns.despine(fig=fig, top=True, right=True, left=True, bottom=True)
    current_ax.imshow(im, cmap = 'gray')
    mask = np.zeros_like(im)
    maskout = np.zeros_like(im,dtype='uint8')

    # fig, ax_arr = plt.subplots(1,2, figsize=(w/96,h/96))
    # fig.suptitle('Select mask')
    # current_ax = ax_arr[0]
    # mask_ax = ax_arr[1]
    # current_ax.set_title('Drag on the image, adjust,\n press r, or c to get a rectangular or a circular ROI')
    # mask_ax.set_title('Visualise the mask')
    # sns.despine(fig=fig, top=True, right=True, left=True, bottom=True)
    # current_ax.imshow(im, cmap = 'gray')
    # mask = np.zeros_like(im)
    # maskout = np.zeros_like(im,dtype='uint8')
    # mask_ax.imshow(mask, cmap = 'gray')

    toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
                                           drawtype='box', useblit=True,
                                           button=[1, 3],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    plt.connect('key_press_event', toggle_selector)
    plt.show()

    return maskout, centers


def checkROI(useROI, usePreviousROI, frame, videoPath):
    ''' Select ROI '''
    if useROI:
        if usePreviousROI:
            mask = loadFile(videoPath, 'ROI')
            mask = np.asarray(mask)
            centers= loadFile(videoPath, 'centers')
            centers = np.asarray(centers) ### TODO maybe we need to pass to a list of tuples
        else:
            print '\n Selecting ROI ...'
            mask, centers = getMask(frame)
            # mask = adaptROI(mask)
    else:
        print '\n No ROI selected ...'
        mask = np.ones_like(frame)*255
        centers = []
    return mask, centers

def ROISelectorPreview(videoPaths, useROI, usePreviousROI, numSegment=0):
    """
    loads a preview of the video for manual fine-tuning
    """
    cap2 = cv2.VideoCapture(videoPaths[0])
    flag, frame = cap2.read()
    cap2.release()
    height = frame.shape[0]
    width = frame.shape[1]
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask, centers = checkROI(useROI, usePreviousROI, frameGray, videoPaths[0])
    saveFile(videoPaths[0], mask, 'ROI')
    saveFile(videoPaths[0], centers, 'centers')
    return width, height, mask, centers

''' ****************************************************************************
First preview numAnimals, inspect parameters for segmentation and portraying
**************************************************************************** '''

# def SegmentationPreview(path, width, height, bkg, mask, useBkg, preprocParams,  size = 1):
#
#     ### FIXME Currently the scale factor of the image is not passed everytime we change the segment. It need to be changed so that we do not need to resize everytime we open a new segmen.
#     minArea = preprocParams['minArea']
#     maxArea = preprocParams['maxArea']
#     minThreshold = preprocParams['minThreshold']
#     maxThreshold = preprocParams['maxThreshold']
#     numAnimals = preprocParams['numAnimals']
#     if numAnimals == None:
#         numAnimals = getInput('Number of animals','Type the number of animals')
#         numAnimals = int(numAnimals)
#
#     global cap
#     cap = cv2.VideoCapture(path)
#     numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
#
#     def thresholder(minTh, maxTh):
#         toile = np.zeros_like(frameGray, dtype='uint8')
#         segmentedFrame = segmentVideo(avFrame, minTh, maxTh, bkg, mask, useBkg)
#         #contours, hierarchy = cv2.findContours(segmentedFrame,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#         maxArea = cv2.getTrackbarPos('maxArea', 'Bars')
#         minArea = cv2.getTrackbarPos('minArea', 'Bars')
#         bbs, miniFrames, _, _, _, goodContours, bkgSamples = blobExtractor(segmentedFrame, frameGray, minArea, maxArea, height, width)
#
#         cv2.drawContours(toile, goodContours, -1, color=255, thickness = -1)
#         shower = cv2.addWeighted(frameGray,1,toile,.5,0)
#         showerCopy = shower.copy()
#         # print showerCopy.shape
#         resUp = cv2.getTrackbarPos('ResUp', 'Bars')
#         resDown = cv2.getTrackbarPos('ResDown', 'Bars')
#
#         showerCopy = cv2.resize(showerCopy,None,fx = resUp, fy = resUp)
#         showerCopy = cv2.resize(showerCopy,None, fx = np.true_divide(1,resDown), fy = np.true_divide(1,resDown))
#
#         numColumns = 5
#         numGoodContours = len(goodContours)
#         numBlackPortraits = numColumns - numGoodContours % numColumns
#         numPortraits = numGoodContours + numBlackPortraits
#
#         j = 0
#         portraitSize = 32
#         portraitsMat = []
#         rowPortrait = []
#         while j < numPortraits:
#             if j < numGoodContours:
#                 portrait,_ = getPortrait(miniFrames[j],goodContours[j],bbs[j],bkgSamples[j])
#                 portrait = cropPortrait(portrait,portraitSize,shift = (0,0))
#                 portrait = np.squeeze(portrait)
#             else:
#                 portrait = np.zeros((portraitSize,portraitSize),dtype='uint8')
#             rowPortrait.append(portrait)
#             if (j+1) % numColumns == 0:
#                 portraitsMat.append(np.hstack(rowPortrait))
#                 rowPortrait = []
#             j += 1
#
#         portraitsMat = np.vstack(portraitsMat)
#
#         cv2.imshow('Bars',np.squeeze(portraitsMat))
#
#         cv2.imshow('IdPlayer', showerCopy)
#         cv2.moveWindow('Bars', 10,10 )
#         cv2.moveWindow('IdPlayer', 200, 10 )
#
#     def scroll(trackbarValue):
#         global frame, avFrame, frameGray
#         cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
#         ret, frame = cap.read()
#         frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         avIntensity = np.float32(np.mean(frameGray))
#         avFrame = np.divide(frameGray,avIntensity)
#         minTh = cv2.getTrackbarPos('minTh', 'Bars')
#         maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
#         thresholder(minTh, maxTh)
#         pass
#
#
#     def changeMinTh(minTh):
#         minTh = cv2.getTrackbarPos('minTh', 'Bars')
#         maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
#         thresholder(minTh, maxTh)
#         pass
#
#     def changeMaxTh(maxTh):
#         minTh = cv2.getTrackbarPos('minTh', 'Bars')
#         maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
#         thresholder(minTh, maxTh)
#         pass
#
#     def changeMinArea(x):
#         minTh = cv2.getTrackbarPos('minTh', 'Bars')
#         maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
#         thresholder(minTh, maxTh)
#         pass
#
#     def changeMaxArea(maxArea):
#         minTh = cv2.getTrackbarPos('minTh', 'Bars')
#         maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
#         thresholder(minTh, maxTh)
#         pass
#
#     def resizeImageUp(res):
#         minTh = cv2.getTrackbarPos('minTh', 'Bars')
#         maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
#         thresholder(minTh, maxTh)
#         pass
#
#     def resizeImageDown(res):
#         minTh = cv2.getTrackbarPos('minTh', 'Bars')
#         maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
#         thresholder(minTh, maxTh)
#         pass
#
#     cv2.createTrackbar('start', 'Bars', 0, numFrame-1, scroll )
#     cv2.createTrackbar('minTh', 'Bars', 0, 255, changeMinTh)
#     cv2.createTrackbar('maxTh', 'Bars', 0, 255, changeMaxTh)
#     cv2.createTrackbar('minArea', 'Bars', 0, 1000, changeMinArea)
#     cv2.createTrackbar('maxArea', 'Bars', 0, 60000, changeMaxArea)
#     cv2.createTrackbar('ResUp', 'Bars', 1, 20, resizeImageUp)
#     cv2.createTrackbar('ResDown', 'Bars', 1, 20, resizeImageDown)
#
#     defFrame = 1
#     defMinTh = minThreshold
#     defMaxTh = maxThreshold
#     defMinA = minArea
#     defMaxA = maxArea
#     defRes = size
#
#     scroll(defFrame)
#     cv2.setTrackbarPos('start', 'Bars', defFrame)
#     changeMaxArea(defMaxA)
#     cv2.setTrackbarPos('maxArea', 'Bars', defMaxA)
#     changeMinArea(defMinA)
#     cv2.setTrackbarPos('minArea', 'Bars', defMinA)
#     changeMinTh(defMinTh)
#     cv2.setTrackbarPos('minTh', 'Bars', defMinTh)
#     changeMaxTh(defMaxTh)
#     cv2.setTrackbarPos('maxTh', 'Bars', defMaxTh)
#     resizeImageUp(defRes)
#     cv2.setTrackbarPos('ResUp', 'Bars', defRes)
#     resizeImageDown(defRes)
#     cv2.setTrackbarPos('ResDown', 'Bars', defRes)
#
#     #start = cv2.getTrackbarPos('start','Bars')
#     #minThresholdStart = cv2.getTrackbarPos('minTh', 'Bars')
#     #minAreaStart = cv2.getTrackbarPos('minArea', 'Bars')
#     #maxAreaStart = cv2.getTrackbarPos('maxArea', 'Bars')
#
#     cv2.waitKey(0)
#
#     preprocParams = {
#                 'minThreshold': cv2.getTrackbarPos('minTh', 'Bars'),
#                 'maxThreshold': cv2.getTrackbarPos('maxTh', 'Bars'),
#                 'minArea': cv2.getTrackbarPos('minArea', 'Bars'),
#                 'maxArea': cv2.getTrackbarPos('maxArea', 'Bars'),
#                 'numAnimals': numAnimals}
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#     cv2.waitKey(1)
#     cv2.destroyAllWindows()
#     cv2.waitKey(1)
#
#     return preprocParams
#
# def selectPreprocParams(videoPaths, usePreviousPrecParams, width, height, bkg, mask, useBkg):
#     if not usePreviousPrecParams:
#         videoPath = videoPaths[0]
#         preprocParams = {
#                     'minThreshold': 0,
#                     'maxThreshold': 155,
#                     'minArea': 150,
#                     'maxArea': 60000,
#                     'numAnimals': None
#                     }
#         preprocParams = SegmentationPreview(videoPath, width, height, bkg, mask, useBkg,preprocParams)
#
#         cv2.waitKey(1)
#         cv2.destroyAllWindows()
#         cv2.waitKey(1)
#
#         end = False
#         while not end:
#             numSegment = getInput('Segment number','Type the segment to be visualized')
#             if numSegment == 'q' or numSegment == 'quit' or numSegment == 'exit':
#                 end = True
#             else:
#                 cv2.namedWindow('Bars')
#                 end = False
#                 usePreviousBkg = 1
#                 videoPath = videoPaths[int(numSegment)]
#
#                 mask = loadFile(videoPaths[0], 'ROI')
#                 mask = np.asarray(mask)
#                 centers= loadFile(videoPaths[0], 'centers')
#                 centers = np.asarray(centers) ### TODO maybe we need to pass to a list of tuples
#                 EQ = 0
#                 ### FIXME put usePreviousBkg to 1 no to recompute it everytime we change the segment
#                 bkg = checkBkg(videoPaths, useBkg, usePreviousBkg, EQ, width, height)
#
#                 preprocParams = SegmentationPreview(videoPath, width, height, bkg, mask, useBkg, preprocParams)
#
#             cv2.waitKey(1)
#             cv2.destroyAllWindows()
#             cv2.waitKey(1)
#
#         saveFile(videoPath, preprocParams, 'preprocparams',hdfpkl='pkl')
#     else:
#         preprocParams= loadFile(videoPaths[0], 'preprocparams',hdfpkl = 'pkl')
#     return preprocParams


def SegmentationPreview(videoPaths, width, height, bkg, mask, useBkg, preprocParams, frameIndices, size = 1):

    ### FIXME Currently the scale factor of the image is not passed everytime we change the segment. It need to be changed so that we do not need to resize everytime we open a new segmen.
    minArea = preprocParams['minArea']
    maxArea = preprocParams['maxArea']
    minThreshold = preprocParams['minThreshold']
    maxThreshold = preprocParams['maxThreshold']
    numAnimals = preprocParams['numAnimals']
    if numAnimals == None:
        numAnimals = getInput('Number of animals','Type the number of animals')
        numAnimals = int(numAnimals)

    global cap, currentSegment
    currentSegment = 0
    cap = cv2.VideoCapture(videoPaths[0])
    numFrames = len(frameIndices)

    def thresholder(minTh, maxTh):
        toile = np.zeros_like(frameGray, dtype='uint8')
        segmentedFrame = segmentVideo(avFrame, minTh, maxTh, bkg, mask, useBkg)
        #contours, hierarchy = cv2.findContours(segmentedFrame,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        maxArea = cv2.getTrackbarPos('maxArea', 'Bars')
        minArea = cv2.getTrackbarPos('minArea', 'Bars')
        bbs, miniFrames, _, _, _, goodContours, bkgSamples = blobExtractor(segmentedFrame, frameGray, minArea, maxArea, height, width)

        cv2.drawContours(toile, goodContours, -1, color=255, thickness = -1)
        shower = cv2.addWeighted(frameGray,1,toile,.5,0)
        showerCopy = shower.copy()
        # print showerCopy.shape
        resUp = cv2.getTrackbarPos('ResUp', 'Bars')
        resDown = cv2.getTrackbarPos('ResDown', 'Bars')

        showerCopy = cv2.resize(showerCopy,None,fx = resUp, fy = resUp)
        showerCopy = cv2.resize(showerCopy,None, fx = np.true_divide(1,resDown), fy = np.true_divide(1,resDown))

        numColumns = 5
        numGoodContours = len(goodContours)
        numBlackPortraits = numColumns - numGoodContours % numColumns
        numPortraits = numGoodContours + numBlackPortraits

        j = 0
        portraitSize = 32
        portraitsMat = []
        rowPortrait = []
        while j < numPortraits:
            if j < numGoodContours:
                portrait,_,_= getPortrait(miniFrames[j],goodContours[j],bbs[j],bkgSamples[j])
                portrait = cropPortrait(portrait,portraitSize,shift = (0,0))
                portrait = np.squeeze(portrait)
            else:
                portrait = np.zeros((portraitSize,portraitSize),dtype='uint8')
            rowPortrait.append(portrait)
            if (j+1) % numColumns == 0:
                portraitsMat.append(np.hstack(rowPortrait))
                rowPortrait = []
            j += 1

        portraitsMat = np.vstack(portraitsMat)

        cv2.imshow('Bars',np.squeeze(portraitsMat))

        cv2.imshow('IdPlayer', showerCopy)
        cv2.moveWindow('Bars', 10,10 )
        cv2.moveWindow('IdPlayer', 200, 10 )

    def scroll(trackbarValue):
        global frame, avFrame, frameGray, cap, currentSegment

        # Select segment dataframe and change cap if needed
        sNumber = frameIndices.loc[trackbarValue,'segment']
        sFrame = frameIndices.loc[trackbarValue,'frame']

        if sNumber != currentSegment: # we are changing segment
            print 'Changing segment...'
            currentSegment = sNumber

            if len(videoPaths) > 1:
                cap = cv2.VideoCapture(videoPaths[sNumber-1])

        #Get frame from video file
        if len(videoPaths) > 1:
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,sFrame)
        else:
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
        ret, frame = cap.read()

        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avIntensity = np.float32(np.mean(frameGray))
        avFrame = np.divide(frameGray,avIntensity)
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        pass


    def changeMinTh(minTh):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        pass

    def changeMaxTh(maxTh):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        pass

    def changeMinArea(x):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        pass

    def changeMaxArea(maxArea):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        pass

    def resizeImageUp(res):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        pass

    def resizeImageDown(res):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        pass

    cv2.createTrackbar('start', 'Bars', 0, numFrames-1, scroll )
    cv2.createTrackbar('minTh', 'Bars', 0, 255, changeMinTh)
    cv2.createTrackbar('maxTh', 'Bars', 0, 255, changeMaxTh)
    cv2.createTrackbar('minArea', 'Bars', 0, 1000, changeMinArea)
    cv2.createTrackbar('maxArea', 'Bars', 0, 60000, changeMaxArea)
    cv2.createTrackbar('ResUp', 'Bars', 1, 20, resizeImageUp)
    cv2.createTrackbar('ResDown', 'Bars', 1, 20, resizeImageDown)

    defFrame = 1
    defMinTh = minThreshold
    defMaxTh = maxThreshold
    defMinA = minArea
    defMaxA = maxArea
    defRes = size

    scroll(defFrame)
    cv2.setTrackbarPos('start', 'Bars', defFrame)
    changeMaxArea(defMaxA)
    cv2.setTrackbarPos('maxArea', 'Bars', defMaxA)
    changeMinArea(defMinA)
    cv2.setTrackbarPos('minArea', 'Bars', defMinA)
    changeMinTh(defMinTh)
    cv2.setTrackbarPos('minTh', 'Bars', defMinTh)
    changeMaxTh(defMaxTh)
    cv2.setTrackbarPos('maxTh', 'Bars', defMaxTh)
    resizeImageUp(defRes)
    cv2.setTrackbarPos('ResUp', 'Bars', defRes)
    resizeImageDown(defRes)
    cv2.setTrackbarPos('ResDown', 'Bars', defRes)

    #start = cv2.getTrackbarPos('start','Bars')
    #minThresholdStart = cv2.getTrackbarPos('minTh', 'Bars')
    #minAreaStart = cv2.getTrackbarPos('minArea', 'Bars')
    #maxAreaStart = cv2.getTrackbarPos('maxArea', 'Bars')

    cv2.waitKey(0)

    preprocParams = {
                'minThreshold': cv2.getTrackbarPos('minTh', 'Bars'),
                'maxThreshold': cv2.getTrackbarPos('maxTh', 'Bars'),
                'minArea': cv2.getTrackbarPos('minArea', 'Bars'),
                'maxArea': cv2.getTrackbarPos('maxArea', 'Bars'),
                'numAnimals': numAnimals}

    cap.release()
    cv2.destroyAllWindows()

    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    return preprocParams

def selectPreprocParams(videoPaths, usePreviousPrecParams, width, height, bkg, mask, useBkg, frameIndices):
    if not usePreviousPrecParams:
        videoPath = videoPaths[0]
        preprocParams = {
                    'minThreshold': 0,
                    'maxThreshold': 155,
                    'minArea': 150,
                    'maxArea': 60000,
                    'numAnimals': None
                    }
        preprocParams = SegmentationPreview(videoPaths, width, height, bkg, mask, useBkg, preprocParams, frameIndices)

        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        saveFile(videoPaths[0], preprocParams, 'preprocparams',hdfpkl='pkl')
    else:
        preprocParams = loadFile(videoPaths[0], 'preprocparams',hdfpkl='pkl')
    return preprocParams

''' ****************************************************************************
Fragmentation inspector
*****************************************************************************'''
def playFragmentation(videoPaths,segmPaths,dfGlobal,visualize = False):
    from fragmentation import computeFrameIntersection ### FIXME For some reason it does not import well in the top and I have to import it here
    """
    IdInspector
    """
    if visualize:
        info = loadFile(videoPaths[0], 'videoInfo', hdfpkl = 'pkl')
        width = info['width']
        height = info['height']
        numAnimals = info['numAnimals']
        maxNumBlobs = info['maxNumBlobs']
        numSegment = 0
        frameIndices = loadFile(videoPaths[0], 'frameIndices')
        path = videoPaths[numSegment]

        def IdPlayerFragmentation(videoPaths,segmPaths,numAnimals, width, height,frameIndices):

            global segmDf, cap, currentSegment
            segmDf,sNumber = loadFile(segmPaths[0], 'segmentation')
            currentSegment = int(sNumber)
            print 'Visualizing video %s' % path
            cap = cv2.VideoCapture(videoPaths[0])
            numFrames = len(frameIndices)
            # numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

            def onChange(trackbarValue):
                global segmDf, cap, currentSegment

                # Select segment dataframe and change cap if needed
                sNumber = frameIndices.loc[trackbarValue,'segment']
                sFrame = frameIndices.loc[trackbarValue,'frame']

                if sNumber != currentSegment: # we are changing segment
                    print 'Changing segment...'
                    prevSegmDf, _ = loadFile(segmPaths[sNumber-2], 'segmentation')
                    segmDf, _ = loadFile(segmPaths[sNumber-1], 'segmentation')
                    currentSegment = sNumber

                    if len(videoPaths) > 1:
                        cap = cv2.VideoCapture(videoPaths[sNumber-1])

                #Get frame from video file
                if len(videoPaths) > 1:
                    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,sFrame)
                else:
                    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
                ret, frame = cap.read()
                font = cv2.FONT_HERSHEY_SIMPLEX
                frameCopy = frame.copy()

                print '**********************************'
                print 'sNumber, ', sNumber
                print 'sFrame, ', sFrame
                print 'trackbarValue, ', trackbarValue

                permutation = dfGlobal.loc[trackbarValue,'permutations']
                centroids = dfGlobal.loc[trackbarValue,'centroids']
                pixelsA = segmDf.loc[sFrame-1,'pixels']
                pixelsB = segmDf.loc[sFrame,'pixels']
                print '------------------------------------------------------------'
                print 'previous frame, ', str(trackbarValue-1), ', permutation, ', dfGlobal.loc[trackbarValue-1,'permutations']
                print 'current frame, ', str(trackbarValue), ', permutation, ', permutation
                trueFragment, s, overlapMat = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
                print 'overlapMat, '
                print overlapMat
                print 'permutation, ', s

                #Color to gray scale
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Plot segmentated blobs
                for i, pixel in enumerate(pixelsB):
                    px = np.unravel_index(pixel,(height,width))
                    frame[px[0],px[1]] = 255

                for i, centroid in enumerate(centroids):
                    cv2.putText(frame,'i'+ str(permutation[i]) + '|h' +str(i),centroid, font, .7,0)

                cv2.putText(frame,str(trackbarValue),(50,50), font, 3,(255,0,0))

                # Visualization of the process
                cv2.imshow('IdPlayerFragmentation',frame)
                pass

            cv2.namedWindow('IdPlayerFragmentation')
            cv2.createTrackbar( 'start', 'IdPlayerFragmentation', 0, numFrames-1, onChange )

            onChange(1)
            cv2.waitKey()

        IdPlayerFragmentation(videoPaths,segmPaths,numAnimals, width, height,frameIndices)
