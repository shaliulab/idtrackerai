# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., De Polavieja, G.G.,
# (2018). idtracker.ai: Tracking all individuals in large collectives of unmarked animals (F.R.-F. and M.G.B. contributed equally to this work. Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)
 

from __future__ import division, absolute_import, print_function
import sys
import numpy as np
import multiprocessing
import math
from matplotlib import pyplot as plt
import cv2
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from idtrackerai.utils.py_utils import  loadFile, saveFile
from idtrackerai.utils.video_utils import cntBB2Full
from idtrackerai.preprocessing.fishcontour import FishContour
if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.get_portraits")

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

def cropPortrait(image, identificationImageSize, shift=(0,0)):
    """ Given a portait it crops it in a shape (identificationImageSize,identificationImageSize) with
    a shift in the rows and columns given by the variable shifts. The size of
    the portait must be bigger than

    :param portrait: portrait to be cropped, usually of shape (36x36)
    :param identificationImageSize: size of the new portrait, usually 32, since the network accepts images of 32x32  pixels
    :param shift: (x,y) displacement when cropping, it can only go from -maxShift to +maxShift
    :return
    """
    currentSize = image.shape[0]
    if currentSize < identificationImageSize:
        raise ValueError('The size of the input portrait must be bigger than identificationImageSize')
    elif currentSize == identificationImageSize:
        return image
    elif currentSize > identificationImageSize:
        maxShift = np.divide(currentSize - identificationImageSize,2)
        if np.max(shift) > maxShift:
            raise ValueError('The shift when cropping the portrait cannot be bigger than (currentSize - identificationImageSize)/2')
        croppedPortrait = image[maxShift + shift[1] : currentSize - maxShift + shift[1], maxShift + shift[0] : currentSize - maxShift + shift[0]]
        return croppedPortrait

def get_portrait(miniframe, cnt, bb, identification_image_size, px_nose_above_center = 9):
    """Acquiring portraits from miniframe (for fish)

    Given a miniframe (i.e. a minimal rectangular image containing an animal)
    it returns a 36x36 image centered on the head.

    :param miniframe: A numpy 2-dimensional array
    :param cnt: A cv2-style contour, i.e. (x,:,y)
    :param bb: Coordinates of the left-top corner of miniframe in the big frame
    :param identification_image_size: size of the portrait (input image to cnn)
    :param px_nose_above_center: Number of pixels of nose above the center of portrait
    :return a smaller 2-dimensional array, and a tuple with all the nose coordinates in frame reference
    """
    # Extra parameters
    half_side_sq = int(identification_image_size/2)
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

def get_body(height, width, miniframe, pixels, bb, identificationImageSize):
    """Acquiring portraits from miniframe (for flies)
    :param miniframe: A numpy 2-dimensional array
    :param cnt: A cv2-style contour, i.e. (x,:,y)
    :param bb: Coordinates of the left-top corner of miniframe in the big frame
    :param maximum_body_length: maximum body length of the blobs. It will be the size of the width and the height of the frame feed it to the CNN
    """
    miniframe = only_blob_pixels(height, width, miniframe, pixels, bb)
    pca = PCA()
    pxs = np.unravel_index(pixels,(height,width))
    pxs1 = np.asarray(zip(pxs[0],pxs[1]))
    pca.fit(pxs1)
    rot_ang = 180 - np.arctan(pca.components_[0][1]/pca.components_[0][0])*180/np.pi - 45 # we substract 45 so that the fish is aligned in the diagonal. This say we have smaller frames
    center = (pca.mean_[1], pca.mean_[0])
    center = full2miniframe(center, bb)
    center = np.array([int(center[0]), int(center[1])])
    #rotate
    diag = np.sqrt(np.sum(np.asarray(miniframe.shape)**2)).astype(int)
    diag = (diag, diag)
    M = cv2.getRotationMatrix2D(tuple(center), rot_ang, 1)
    minif_rot = cv2.warpAffine(miniframe, M, diag, borderMode=cv2.BORDER_CONSTANT, flags = cv2.INTER_CUBIC)
    crop_distance = int(identificationImageSize/2)
    x_range = xrange(center[0] - crop_distance, center[0] + crop_distance)
    y_range = xrange(center[1] - crop_distance, center[1] + crop_distance)
    portrait = minif_rot.take(y_range, mode = 'wrap', axis=0).take(x_range, mode = 'wrap', axis=1)
    height, width = portrait.shape
    rot_ang_rad = rot_ang * np.pi / 180
    h_or_t_1 = np.array([np.cos(rot_ang_rad), np.sin(rot_ang_rad)]) * rot_ang_rad
    h_or_t_2 = - h_or_t_1
    return portrait, tuple(h_or_t_1.astype('int')), tuple(h_or_t_2.astype('int'))

def only_blob_pixels(height, width, miniframe, pixels, bb):
    pxs = np.array(np.unravel_index(pixels,(height, width))).T
    pxs = np.array([pxs[:, 0] - bb[0][1], pxs[:, 1] - bb[0][0]])
    temp_image = np.zeros_like(miniframe).astype('uint8')
    temp_image[pxs[0,:], pxs[1,:]] = 255
    temp_image = cv2.dilate(temp_image, np.ones((3,3)).astype('uint8'), iterations = 1)
    rows, columns = np.where(temp_image == 255)
    dilated_pixels = np.array([rows, columns])
    temp_image[dilated_pixels[0,:], dilated_pixels[1,:]] = miniframe[dilated_pixels[0,:], dilated_pixels[1,:]]
    return temp_image

def reaper(videoPath, frameIndices, height, width):
    df, numSegment = loadFile(videoPath, 'segmentation')
    boundingboxes = np.asarray(df.loc[:, 'boundingBoxes']) #coordinate in the frame
    miniframes = np.asarray(df.loc[:, 'miniFrames']) #image containing the blob, same size
    miniframes = np.asarray(miniframes)
    contours = np.asarray(df.loc[:, 'contours'])
    centroidsSegment = np.asarray(df.loc[:,'centroids'])
    pixels = np.asarray(df.loc[:, 'pixels'])
    areasSegment = np.asarray(df.loc[:, 'areas'])
    segmentIndices = frameIndices.loc[frameIndices.loc[:,'segment']==int(numSegment)]
    segmentIndices = segmentIndices.index.tolist()
    AllPortraits = pd.DataFrame(index = segmentIndices, columns= ['portraits'])
    AllBodies = pd.DataFrame(index = segmentIndices, columns= ['bodies'])
    AllBodyBlobs = pd.DataFrame(index = segmentIndices, columns= ['bodyblobs'])
    AllNoses = pd.DataFrame(index = segmentIndices, columns= ['noses'])
    AllCentroids= pd.DataFrame(index = segmentIndices, columns= ['centroids'])
    AllHeadCentroids = pd.DataFrame(index = segmentIndices, columns= ['head_centroids'])
    AllAreas = pd.DataFrame(index = segmentIndices, columns= ['areas'])
    counter = 0
    while counter < len(miniframes):
        portraits = []
        bodies = []
        bodyblobs = []
        noses = []
        head_centroids = []
        areas = areasSegment[counter]
        bbs = boundingboxes[counter]
        minif = miniframes[counter]
        cnts = contours[counter]
        centroids = centroidsSegment[counter]
        pxs = pixels[counter]

        for j, miniframe in enumerate(minif):
            identificationImageSize = 36
            portrait, nose_pixels, head_centroid_pixels = get_portrait(miniframe,cnts[j],bbs[j],identificationImageSize)
            portraits.append(portrait)
            noses.append(nose_pixels)
            head_centroids.append(head_centroid_pixels)
            identificationImageSize = 52
            body, _, _ = get_body(height, width, miniframe, pxs[j], bbs[j], identificationImageSize, only_blob = False)
            bodies.append(body)
            bodyblob, _, _ = get_body(height, width, miniframe, pxs[j], bbs[j], identificationImageSize, only_blob = True)
            bodyblobs.append(bodyblob)

        AllPortraits.set_value(segmentIndices[counter], 'portraits', portraits)
        AllBodies.set_value(segmentIndices[counter], 'bodies', bodies)
        AllBodyBlobs.set_value(segmentIndices[counter], 'bodyblobs', bodyblobs)
        AllNoses.set_value(segmentIndices[counter], 'noses', noses)
        AllHeadCentroids.set_value(segmentIndices[counter], 'head_centroids', head_centroids)
        AllCentroids.set_value(segmentIndices[counter], 'centroids', centroids)
        AllAreas.set_value(segmentIndices[counter], 'areas', areas)

    return AllPortraits, AllBodies, AllBodyBlobs, AllNoses, AllHeadCentroids, AllCentroids, AllAreas

def portrait(videoPaths, dfGlobal, height, width):
    frameIndices = loadFile(videoPaths[0], 'frameIndices')
    num_cores = multiprocessing.cpu_count()
    out = Parallel(n_jobs=num_cores)(delayed(reaper)(videoPath, frameIndices, height, width) for videoPath in videoPaths)
    allPortraits = [t[0] for t in out]
    allPortraits = pd.concat(allPortraits)
    allPortraits = allPortraits.sort_index(axis=0,ascending=True)
    allBodies = [t[1] for t in out]
    allBodies = pd.concat(allBodies)
    allBodies = allBodies.sort_index(axis=0,ascending=True)
    allBodyBlobs = [t[2] for t in out]
    allBodyBlobs = pd.concat(allBodyBlobs)
    allBodyBlobs = allBodyBlobs.sort_index(axis=0,ascending=True)
    allNoses = [t[3] for t in out]
    allNoses = pd.concat(allNoses)
    allNoses = allNoses.sort_index(axis=0, ascending=True)
    allHeadCentroids = [t[4] for t in out]
    allHeadCentroids = pd.concat(allHeadCentroids)
    allHeadCentroids = allHeadCentroids.sort_index(axis=0, ascending=True)
    allCentroids = [t[5] for t in out]
    allCentroids = pd.concat(allCentroids)
    allCentroids = allCentroids.sort_index(axis=0, ascending=True)
    allAreas = [t[6] for t in out]
    allAreas = pd.concat(allAreas)
    allAreas = allAreas.sort_index(axis=0, ascending=True)
    if list(allPortraits.index) != list(dfGlobal.index):
        raise ValueError('The list of indexes in allPortraits and dfGlobal should be the same')
    dfGlobal['identities'] = dfGlobal['permutations']
    dfGlobal['portraits'] = allPortraits
    dfGlobal['bodies'] = allBodies
    dfGlobal['bodyblobs'] = allBodyBlobs
    dfGlobal['noses'] = allNoses
    dfGlobal['head_centroids'] = allHeadCentroids
    dfGlobal['centroids'] = allCentroids
    dfGlobal['areas'] = allAreas
    saveFile(videoPaths[0], dfGlobal, 'portraits',hdfpkl='pkl')
    return dfGlobal
