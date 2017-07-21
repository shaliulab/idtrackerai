# Import standard libraries
from __future__ import division
import sys
import numpy as np
import multiprocessing
import math

# Import third party libraries
from matplotlib import pyplot as plt
import cv2
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

# Import application/library specifics
sys.path.append('IdTrackerDeep/utils')
from py_utils import loadFile, saveFile
from video_utils import cntBB2Full, full2BoundingBox

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

# def fillSquareFrame(square_frame, bkgSamps):
#     '''Used in get_miniframes.py'''
#     numSamples = 0
#     threshold = 150
#     while numSamples <= 10:
#
#         bkgSampsNew = bkgSamps[bkgSamps > threshold]
#         threshold -= 10
#         if threshold == 0:
#             plt.imshow(square_frame,cmap='gray',interpolation='none')
#             plt.show()
#             break
#         numSamples = len(bkgSampsNew)
#     bkgSamps = bkgSampsNew
#     if numSamples == 0:
#         raise ValueError('I do not have enough samples for the background')
#
#     numSamplesRequested = np.sum(square_frame == 0)
#     indicesSamples = np.random.randint(0,numSamples,numSamplesRequested)
#     square_frame[square_frame == 0] = bkgSamps[indicesSamples]
#     return square_frame

def cropPortrait(image, portraitSize, shift=(0,0)):
    """ Given a portait it crops it in a shape (portraitSize,portraitSize) with
    a shift in the rows and columns given by the variable shifts. The size of
    the portait must be bigger than

    :param portrait: portrait to be cropped, usually of shape (36x36)
    :param portraitSize: size of the new portrait, usually 32, since the network accepts images of 32x32  pixels
    :param shift: (x,y) displacement when cropping, it can only go from -maxShift to +maxShift
    :return
    """
    currentSize = image.shape[0]
    if currentSize < portraitSize:
        raise ValueError('The size of the input portrait must be bigger than portraitSize')
    elif currentSize == portraitSize:
        return image
    elif currentSize > portraitSize:
        maxShift = np.divide(currentSize - portraitSize,2)
        if np.max(shift) > maxShift:
            raise ValueError('The shift when cropping the portrait cannot be bigger than (currentSize - portraitSize)/2')
        croppedPortrait = image[maxShift + shift[1] : currentSize - maxShift + shift[1], maxShift + shift[0] : currentSize - maxShift + shift[0]]
        # print 'Portrait cropped'
        return croppedPortrait

def getPortrait(miniframe, cnt, bb, portrait_size, px_nose_above_center = 9):
    """Acquiring portraits from miniframe (for fish)

    Given a miniframe (i.e. a minimal rectangular image containing an animal)
    it returns a 36x36 image centered on the head.

    :param miniframe: A numpy 2-dimensional array
    :param cnt: A cv2-style contour, i.e. (x,:,y)
    :param bb: Coordinates of the left-top corner of miniframe in the big frame
    :param portrait_size: size of the portrait (input image to cnn)
    :param px_nose_above_center: Number of pixels of nose above the center of portrait
    :return a smaller 2-dimensional array, and a tuple with all the nose coordinates in frame reference
    """
    # Extra parameters
    half_side_sq = int(portrait_size/2)
    overhead = int(np.ceil(np.sqrt(half_side_sq**2 + (half_side_sq+px_nose_above_center)**2))) # Extra pixels when performing rotation, around sqrt(half_side_sq**2 + (half_side_sq+px_nose_above_center)**2)

    # Calculating nose coordinates in the full frame reference
    contour_cnt = FishContour.fromcv2contour(cnt)
    noseFull, rot_ang, head_centroid_full = contour_cnt.find_nose_and_orientation()

    # Calculating nose coordinates in miniframe reference
    nose = full2miniframe(noseFull,bb) #Float
    nose_pixels = np.array([int(nose[0]),int(nose[1])]) #int

    # Get roto-translation matrix and rotated miniframe
    # Rotation is performed around nose, nose coordinates stay constant
    # Final image gives an overhead above the nose coordinates, so the whole head should
    # stay visible in the final frame.
    # borderMode=cv2.BORDER_WRAP determines how source image is extended when needed
    M = cv2.getRotationMatrix2D(nose, rot_ang,1)
    minif_rot = cv2.warpAffine(miniframe, M, tuple(nose_pixels+overhead), borderMode=cv2.BORDER_WRAP, flags = cv2.INTER_CUBIC)

    # Crop the image in 32x32 frame around the nose
    x_range = xrange(nose_pixels[0]-half_side_sq,nose_pixels[0]+half_side_sq)
    y_range = xrange(nose_pixels[1]-half_side_sq+px_nose_above_center,nose_pixels[1]+half_side_sq+px_nose_above_center)
    portrait = minif_rot.take(y_range,mode='wrap',axis=0).take(x_range,mode='wrap',axis=1)

    return portrait, tuple(noseFull.astype('float32')), tuple(head_centroid_full.astype('float32')) #output as float because it is better for analysis.

def get_portrait_fly(video, miniframe, pixels, bb, portraitSize):
    """Acquiring portraits from miniframe (for flies)
    :param miniframe: A numpy 2-dimensional array
    :param cnt: A cv2-style contour, i.e. (x,:,y)
    :param bb: Coordinates of the left-top corner of miniframe in the big frame
    :param maximum_body_length: maximum body length of the blobs. It will be the size of the width and the height of the frame feed it to the CNN
    """
    miniframe = only_blob_pixels(video, miniframe, pixels, bb)
    pca = PCA()
    pxs = np.unravel_index(pixels,(video._height,video._width))
    pxs1 = np.asarray(zip(pxs[0],pxs[1]))
    pca.fit(pxs1)
    rot_ang = 180 - np.arctan(pca.components_[0][1]/pca.components_[0][0])*180/np.pi - 45 # we substract 45 so that the fish is aligned in the diagonal. This say we have smaller frames
    center = (pca.mean_[1], pca.mean_[0])
    # print("PCA center before: ", center)
    center = full2miniframe(center, bb)
    center = np.array([int(center[0]), int(center[1])])
    # print("PCA center and angle: ", center, rot_ang)

    #rotate
    diag = np.sqrt(np.sum(np.asarray(miniframe.shape)**2)).astype(int)
    diag = (diag, diag)
    M = cv2.getRotationMatrix2D(tuple(center), rot_ang, 1)
    minif_rot = cv2.warpAffine(miniframe, M, diag, borderMode=cv2.BORDER_WRAP, flags = cv2.INTER_CUBIC)

    crop_distance = int(portraitSize/2)
    x_range = xrange(center[0] - crop_distance, center[0] + crop_distance)
    y_range = xrange(center[1] - crop_distance, center[1] + crop_distance)
    portrait = minif_rot.take(y_range, mode = 'wrap', axis=0).take(x_range, mode = 'wrap', axis=1)
    height, width = portrait.shape

    rot_ang_rad = rot_ang * np.pi / 180
    h_or_t_1 = np.array([np.cos(rot_ang_rad), np.sin(rot_ang_rad)]) * rot_ang_rad
    h_or_t_2 = - h_or_t_1
    return portrait, tuple(h_or_t_1.astype('int')), tuple(h_or_t_2.astype('int'))

def only_blob_pixels(video, miniframe, pixels, bb):
    pxs = np.array(np.unravel_index(pixels,(video._height,video._width))).T
    # print("pixels ", pxs)
    pxs = np.array([pxs[:, 0] - bb[0][1], pxs[:, 1] - bb[0][0]])
    temp_image = np.zeros_like(miniframe).astype('uint8')
    temp_image[pxs[0,:], pxs[1,:]] = 255
    temp_image = cv2.dilate(temp_image, np.ones((3,3)).astype('uint8'), iterations = 1)
    rows, columns = np.where(temp_image == 255)
    dilated_pixels = np.array([rows, columns])
    print("dilated pixels ", dilated_pixels.shape)

    temp_image[dilated_pixels[0,:], dilated_pixels[1,:]] = miniframe[dilated_pixels[0,:], dilated_pixels[1,:]]
    return temp_image


def reaper(videoPath, frameIndices, animal_type):
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
    AllHeadCentroids = pd.DataFrame(index = segmentIndices, columns= ['head_centroids'])

    counter = 0
    while counter < len(miniframes):
        portraits = []
        noses = []
        head_centroids = []
        bbs = boundingboxes[counter]
        minif = miniframes[counter]
        cnts = contours[counter]
        bkgSamps = bkgSamples[counter]
        centroids = centroidsSegment[counter]
        for j, miniframe in enumerate(minif):
            ### Uncomment to plot
            # cv2.imshow('frame', miniframe)
            # cv2.waitKey()
            if animal_type == 'fish':
                portrait, nose_pixels, head_centroid_pixels = getPortrait(miniframe,cnts[j],bbs[j],bkgSamps[j],j)

                # get all the heads in a single list
                portraits.append(portrait)
                noses.append(nose_pixels)
                head_centroids.append(head_centroid_pixels)

            elif animal_type == 'fly':
                portrait, nose_pixels, head_centroid_pixels = get_portrait_fly(miniframe,cnts[j],bbs[j])

                # get all the heads in a single list
                portraits.append(portrait)
                noses.append(nose_pixels)
                head_centroids.append(head_centroid_pixels)

        ### UNCOMMENT TO PLOT ##################################################
        #    cv2.imshow(str(j),portrait)
        #
        #k = cv2.waitKey(100) & 0xFF
        #if k == 27: #pres esc to quit
        #    break
        ########################################################################

        AllPortraits.set_value(segmentIndices[counter], 'images', portraits)
        AllNoses.set_value(segmentIndices[counter], 'noses', noses)
        AllHeadCentroids.set_value(segmentIndices[counter], 'head_centroids', head_centroids)
        AllCentroids.set_value(segmentIndices[counter], 'centroids', centroids)
        counter += 1
    print 'you just reaped', videoPath
    return AllPortraits, AllNoses, AllCentroids, AllHeadCentroids

def portrait(videoPaths, dfGlobal, animal_type):
    frameIndices = loadFile(videoPaths[0], 'frameIndices')
    num_cores = multiprocessing.cpu_count()
    # num_cores = 1
    allPortraitsAndNoses = Parallel(n_jobs=num_cores)(delayed(reaper)(videoPath,frameIndices, animal_type) for videoPath in videoPaths)
    allPortraits = [t[0] for t in allPortraitsAndNoses]
    allNoses = [t[1] for t in allPortraitsAndNoses]
    allCentroids = [t[2] for t in allPortraitsAndNoses]
    allHeadCentroids = [t[3] for t in allPortraitsAndNoses]
    allPortraits = pd.concat(allPortraits)
    allPortraits = allPortraits.sort_index(axis=0,ascending=True)
    allNoses = pd.concat(allNoses)
    allNoses = allNoses.sort_index(axis=0, ascending=True)
    allHeadCentroids = pd.concat(allHeadCentroids)
    allHeadCentroids = allHeadCentroids.sort_index(axis=0, ascending=True)
    allCentroids = pd.concat(allCentroids)
    allCentroids = allCentroids.sort_index(axis=0, ascending=True)


    if list(allPortraits.index) != list(dfGlobal.index):
        raise ValueError('The list of indexes in allPortraits and dfGlobal should be the same')
    # dfGlobal1 = pd.DataFrame(index = range(len(dfGlobal)), columns=['images'])
    dfGlobal['images'] = allPortraits
    dfGlobal['identities'] = dfGlobal['permutations']
    dfGlobal['noses'] = allNoses
    dfGlobal['head_centroids'] = allHeadCentroids
    dfGlobal['centroids'] = allCentroids

    saveFile(videoPaths[0], dfGlobal, 'portraits',hdfpkl='pkl')
    return dfGlobal
