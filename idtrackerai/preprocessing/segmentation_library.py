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
# (2018). idtracker.ai: Tracking unmarked individuals in large collectives. (R-F.,F. and B.,M. contributed equally to this work.)
 
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
sys.path.append('../utils')

from idtrackerai.utils.py_utils import  flatten, loadFile, saveFile
from video_utils_library import collectAndSaveVideoInfo, generateVideoTOC, getVideoInfo
from idtrackerai.utils.video_utils import segment_frame, blob_extractor

def segment_episode(path, height, width, mask, useBkg, bkg, EQ, min_threshold, max_threshold, min_area, max_area, episode_start_end_frames = None,framesPerSegment=None):
    # locally called

    cap = cv2.VideoCapture(path)
    if episode_start_end_frames == None:
        print 'Segmenting video %s' % path
        video = os.path.basename(path)
        filename, extension = os.path.splitext(video)
        numSegment = int(filename.split('_')[-1])
        numFrames = int(cap.get(7))
        counter = 0
    else:
        numSegment = int(episode_start_end_frames[0]/framesPerSegment) + 1
        print 'Segment video %s from frame %i to frame %i (segment %i)' %(path, episode_start_end_frames[0], episode_start_end_frames[1], numSegment)
        numFrames = episode_start_end_frames[1] - episode_start_end_frames[0] + 1
        counter = 0
        cap.set(1,episode_start_end_frames[0])
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
        # print avIntensity
        # print frameGray.shape
        segmentedFrame = segment_frame(frameGray/avIntensity, min_threshold, max_threshold, bkg, mask, useBkg)
        # segmentedFrameCopy = segmentedFrame.copy()
        # Find contours in the segmented image
        boundingBoxes, miniFrames, centroids, areas, pixels, goodContoursFull, bkgSamples = blob_extractor(segmentedFrame, frameGray, min_area, max_area, height, width)
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
    min_threshold = preprocParams['min_threshold']
    max_threshold = preprocParams['max_threshold']
    min_area = preprocParams['min_area']
    max_area = preprocParams['max_area']

    width, height = getVideoInfo(videoPaths)


    print 'videoPaths here, ', videoPaths
    # num_cores = multiprocessing.cpu_count()
    num_cores = 4
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

        segmFramesIndicesSubLists = [segmFramesIndices[i:i+num_cores] for i in range(0,len(segmFramesIndices),num_cores)]
        print 'Entering to the parallel loop...\n'
        allSegments = []
        numBlobs = []
        path = videoPaths[0]
        for segmFramesIndicesSubList in segmFramesIndicesSubLists:
            OupPutParallel = Parallel(n_jobs=num_cores)(delayed(segment_episode)(path, height, width, mask, useBkg, bkg, EQ, min_threshold, max_threshold, min_area, max_area, episode_start_end_frames, framesPerSegment) for episode_start_end_frames in segmFramesIndicesSubList)
            allSegmentsSubList = [(out[0],out[1]) for out in OupPutParallel]
            allSegments.append(allSegmentsSubList)
            numBlobs.append([out[2] for out in OupPutParallel])

    else:
        ''' splitting videoPaths list into sublists '''
        pathsSubLists = [videoPaths[i:i+num_cores] for i in range(0,len(videoPaths),num_cores)]
        ''' Entering loop for segmentation of the video '''
        print 'Entering to the parallel loop...\n'
        allSegments = []
        numBlobs = []
        for pathsSubList in pathsSubLists:
            OupPutParallel = Parallel(n_jobs=num_cores)(delayed(segment_episode)(path, height, width, mask, useBkg, bkg, EQ, min_threshold, max_threshold, min_area, max_area) for path in pathsSubList)
            allSegmentsSubList = [(out[0],out[1]) for out in OupPutParallel]
            allSegments.append(allSegmentsSubList)
            numBlobs.append([out[2] for out in OupPutParallel])

    allSegments = flatten(allSegments)
    maxNumBlobs = max(flatten(numBlobs))
    # OupPutParallel = Parallel(n_jobs=num_cores)(delayed(segment_episode)(path, height, width, mask, useBkg, bkg, EQ, min_threshold, max_threshold, min_area, max_area) for path in videoPaths)
    # allSegments = [(out[0],out[1]) for out in OupPutParallel]
    # # print allSegments
    # maxNumBlobs = max([out[2] for out in OupPutParallel])
    # # print maxNumBlobs
    allSegments = sorted(allSegments, key=lambda x: x[0][0])
    numFrames = generateVideoTOC(allSegments, videoPaths[0])
    collectAndSaveVideoInfo(videoPaths[0], numFrames, height, width, numAnimals, num_cores, min_threshold,max_threshold,max_area,maxNumBlobs)
