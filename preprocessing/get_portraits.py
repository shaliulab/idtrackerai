# Import standard libraries
import sys
import numpy as np
import multiprocessing
import math

# Import third party libraries
from matplotlib import pyplot as plt
import cv2
import pandas as pd
from joblib import Parallel, delayed

# Import application/library specifics
sys.path.append('IdTrackerDeep/utils')
from py_utils import loadFile, saveFile
from video_utils import cntBB2Full

from fishcontour import FishContour

def full2miniframe(point, boundingBox):
    """
    Push a point in the fullframe to miniframe coordinate system.
    Here it is use for centroids
    """
    return tuple(np.asarray(point) - np.asarray([boundingBox[0][0],boundingBox[0][1]]))

def getEncompassingIndices(frameIndices, num_segmnent, goodIndices):
    """
    frameIndices = dataframe containing the list of frame per segment
    num_segment
    goodIndices = indices in which the permutation is defined (non-crossing and overlapping)
    """
    frameSegment = frameIndices.loc[frameIndices.loc[:,'segment']==num_segmnent]
    goodFrameIndices = frameSegment.iloc[goodIndices].index.tolist()

    return goodFrameIndices, frameSegment.index.tolist()

def getMFandC(path, frameIndices):
    """
    path: path to dataframe
    generate a list of arrays containing miniframes and centroids detected in
    path at this point we can already discard miniframes that does not belong
    to a specific fragments
    """
    # get number of segment
    df, numSegment = loadFile(path, 'segmentation')
    # check if permutations are NaN (i.e. the frame is not included in a fragment)
    permutationsBool = np.asarray(df['permutation'].notnull())

    #generate a lists of "admissible" miniframes and centroids
    permutations = np.asarray(df['permutation'])
    boundingBoxes = np.asarray(df.loc[:, 'boundingBoxes'])
    miniframes = np.asarray(df.loc[:, 'miniFrames'])
    centroids = np.asarray(df.loc[:, 'centroids'])
    bkgSamples = np.asarray(df.loc[:,'bkgSamples'])

    goodIndices = np.where(permutationsBool==True)[0]
    goodFrameIndices, segmentIndices = getEncompassingIndices(frameIndices, int(numSegment), goodIndices)
    goodFrameIndices = segmentIndices

    return boundingBoxes.tolist(), miniframes.tolist(), centroids.tolist(), bkgSamples.tolist(), goodFrameIndices, segmentIndices, permutations.tolist()

def fillSquareFrame(square_frame,bkgSamps):
    '''Used in get_miniframes.py'''
    numSamples = 0
    threshold = 150
    while numSamples <= 10:

        bkgSampsNew = bkgSamps[bkgSamps > threshold]
        threshold -= 10
        if threshold == 0:
            plt.imshow(square_frame,cmap='gray',interpolation='none')
            plt.show()
            break
        numSamples = len(bkgSampsNew)
    bkgSamps = bkgSampsNew
    if numSamples == 0:
        raise ValueError('I do not have enough samples for the background')

    numSamplesRequested = np.sum(square_frame == 0)
    indicesSamples = np.random.randint(0,numSamples,numSamplesRequested)
    square_frame[square_frame == 0] = bkgSamps[indicesSamples]
    return square_frame

def getPortrait(miniframe,cnt,bb,bkgSamp,counter = None,px_nose_above_center = 9):
    """Given a miniframe (i.e. a minimal rectangular image containing an animal)
    it returns a 32x32 image centered on the head.

    :param miniframe: A numpy 2-dimensional array
    :param cnt: A cv2-style contour, i.e. (x,:,y)
    :param bb: Coordinates of the left-top corner of miniframe in the big frame
    :param bkgSamp: Not used in my implementation
    :param counter: Not used in my implementation
    :param px_nose_above_center: Number of pixels of nose above the center of portrait
    :return a smaller 2-dimensional array, and a tuple with all the nose coordinates in frame reference
    """

    # Extra parameters
    half_side_sq = 16 # Because we said that the final portrait is 32x32
    overhead = 30 # Extra pixels when performing rotation, around sqrt(half_side_sq**2 + (half_side_sq+px_nose_above_center)**2)

    # Calculating nose coordinates in the full frame reference
    contour_cnt = FishContour.fromcv2contour(cnt)
    noseFull, rot_ang, _ = contour_cnt.find_nose_and_orientation()

    # Calculating nose coordinates in miniframe reference
    nose = full2miniframe(noseFull,bb) #Float
    nose_pixels = np.array([int(nose[0]),int(nose[1])]) #int

    # Get roto-translation matrix and rotated miniframe
    # Rotation is performed around nose, nose coordinates stay constant
    # Final image gives an overhead above the nose coordinates, so the whole head should
    # stay visible in the final frame.
    # borderMode=cv2.BORDER_WRAP determines how source image is extended when needed
    M = cv2.getRotationMatrix2D(nose, rot_ang,1)
    minif_rot = cv2.warpAffine(miniframe, M, tuple(nose_pixels+overhead), borderMode=cv2.BORDER_WRAP, flags = cv2.INTER_NEAREST)

    # Crop the image in 32x32 frame around the nose
    x_range = xrange(nose_pixels[0]-half_side_sq,nose_pixels[0]+half_side_sq)
    y_range = xrange(nose_pixels[1]-half_side_sq+px_nose_above_center,nose_pixels[1]+half_side_sq+px_nose_above_center) #7,25
    portrait = minif_rot.take(y_range,mode='wrap',axis=0).take(x_range,mode='wrap',axis=1)

    #if portrait.shape[0] != 32 or portrait.shape[1] != 32: #This is redundant now by my use of take
    #    print portrait.shape
    #    raise ValueError('This portrait do not have 32x32 pixels. Changes in light during the video could deteriorate the blobs: try and rais the threshold in the preprocessing parametersm, and run segmentation and fragmentation again.')

    return portrait, tuple(noseFull.astype('int'))

def reaper(videoPath, frameIndices):
    # only function called from idTrackerDeepGUI
    print 'reaping', videoPath
    df, numSegment = loadFile(videoPath, 'segmentation')

    boundingboxes = np.asarray(df.loc[:, 'boundingBoxes']) #coordinate in the frame
    miniframes = np.asarray(df.loc[:, 'miniFrames']) #image containing the blob, same size
    miniframes = np.asarray(miniframes)
    contours = np.asarray(df.loc[:, 'contours'])
    bkgSamples = np.asarray(df.loc[:,'bkgSamples'])
    centroidsSegment = np.asarray(df.loc[:,'centroids'])

    segmentIndices = frameIndices.loc[frameIndices.loc[:,'segment']==int(numSegment)]
    segmentIndices = segmentIndices.index.tolist()


    """ Visualise """
    AllPortraits = pd.DataFrame(index = segmentIndices, columns= ['images'])
    AllNoses = pd.DataFrame(index = segmentIndices, columns= ['noses'])
    AllCentroids= pd.DataFrame(index = segmentIndices, columns= ['centroids'])

    counter = 0
    while counter < len(miniframes):
        portraits = []
        noses = []
        bbs = boundingboxes[counter]
        minif = miniframes[counter]
        cnts = contours[counter]
        bkgSamps = bkgSamples[counter]
        centroids = centroidsSegment[counter]
        for j, miniframe in enumerate(minif):
            ### Uncomment to plot
            # cv2.imshow('frame', miniframe)
            # cv2.waitKey()
            portrait, nose_pixels = getPortrait(miniframe,cnts[j],bbs[j],bkgSamps[j],j)

            # get all the heads in a single list
            portraits.append(portrait)
            noses.append(nose_pixels)
        ### UNCOMMENT TO PLOT ##################################################
        #    cv2.imshow(str(j),portrait)
        #
        #k = cv2.waitKey(100) & 0xFF
        #if k == 27: #pres esc to quit
        #    break
        ########################################################################

        AllPortraits.set_value(segmentIndices[counter], 'images', portraits)
        AllNoses.set_value(segmentIndices[counter], 'noses', noses)
        AllCentroids.set_value(segmentIndices[counter], 'centroids', centroids)
        counter += 1
    print 'you just reaped', videoPath
    return AllPortraits, AllNoses, AllCentroids

def portrait(videoPaths, dfGlobal):
    frameIndices = loadFile(videoPaths[0], 'frameIndices')
    num_cores = multiprocessing.cpu_count()
    # num_cores = 1
    allPortraitsAndNoses = Parallel(n_jobs=num_cores)(delayed(reaper)(videoPath,frameIndices) for videoPath in videoPaths)
    allPortraits = [t[0] for t in allPortraitsAndNoses]
    allNoses = [t[1] for t in allPortraitsAndNoses]
    allCentroids = [t[2] for t in allPortraitsAndNoses]
    allPortraits = pd.concat(allPortraits)
    allPortraits = allPortraits.sort_index(axis=0,ascending=True)
    allNoses = pd.concat(allNoses)
    allNoses = allNoses.sort_index(axis=0, ascending=True)
    allCentroids = pd.concat(allCentroids)
    allCentroids = allCentroids.sort_index(axis=0, ascending=True)


    if list(allPortraits.index) != list(dfGlobal.index):
        raise ValueError('The list of indexes in allPortraits and dfGlobal should be the same')
    dfGlobal['images'] = allPortraits
    dfGlobal['identities'] = dfGlobal['permutations']
    dfGlobal['noses'] = allNoses
    dfGlobal['centroids'] = allCentroids

    saveFile(videoPaths[0], dfGlobal, 'portraits')
    return dfGlobal
