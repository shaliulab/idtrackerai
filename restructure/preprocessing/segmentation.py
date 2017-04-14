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
sys.path.append('../IdTrackerDeep')
from blob import Blob

from py_utils import flatten
from video_utils import segmentVideo, blobExtractor

def segmentAndSave(video, path = None, segmFrameInd = None):
    blobs_in_episode = []

    if segmFrameInd == None:
        cap = cv2.VideoCapture(path)
        print 'Segmenting video %s' % path
        video = os.path.basename(path)
        filename, extension = os.path.splitext(video)
        numSegment = int(filename.split('_')[-1])
        numFrames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        counter = 0
    elif path is None:
        cap = cv2.VideoCapture(video.video_path)
        numSegment = video.in_which_episode(segmFrameInd[0] + 1)
        print 'Segment video %s from frame %i to frame %i (segment %i)' %(path, segmFrameInd[0], segmFrameInd[1], numSegment)
        numFrames = segmFrameInd[1] - segmFrameInd[0] + 1
        counter = 0
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,segmFrameInd[0])
    maxNumBlobs = 0

    while counter < numFrames:

        blobs_in_frame = []
        #Get frame from video file
        ret, frame = cap.read()
        #Color to gray scale
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avIntensity = np.float32(np.mean(frameGray))
        segmentedFrame = segmentVideo(frameGray/avIntensity, video._min_threshold, video._max_threshold, video.bkg, video.ROI, video.subtract_bkg)
        # Find contours in the segmented image
        bounding_boxes, miniframes, centroids, areas, pixels, contours = blobExtractor(segmentedFrame, frameGray,
                                                                                            video._min_area, video._max_area,
                                                                                            video._height, video._width)

        for i, bounding_box in enumerate(bounding_boxes):
            blob = Blob(centroids[i],
                        contours[i],
                        areas[i],
                        bounding_box,
                        bounding_box_image = miniframes[i])
            blobs_in_frame.append(blob)

        if len(centroids) > maxNumBlobs:
            maxNumBlobs = len(centroids)

        #store all the blobs encountered in the episode
        blobs_in_episode.append(blobs_in_frame)
        counter += 1

    cap.release()
    cv2.destroyAllWindows()
    gc.collect()
    return blobs_in_episode, numSegment, maxNumBlobs

def segment(video):
    # this func is called from idTrackerDeepGUI
    # avoid computing with all the cores in very large videos:
    # num_cores = multiprocessing.cpu_count()
    num_cores = 4
    #init variables to store data
    allSegments = []
    numBlobs = []
    #videoPaths is used to check if the video was previously split or not (it is either None or
    #a list of paths. Check attribute _paths_to_video_segments in Video)
    videoPaths = video._paths_to_video_segments
    if not videoPaths:
        print '**************************************'
        print 'There is only one path, segmenting by frame indices'
        print '**************************************'
        #Define list of starting and ending frames
        segmFramesIndices = video._episodes
        #Spliting frames list into sublists
        segmFramesIndicesSubLists = [segmFramesIndices[i:i+num_cores] for i in range(0,len(segmFramesIndices),num_cores)]

        for segmFramesIndicesSubList in segmFramesIndicesSubLists:
            OupPutParallel = Parallel(n_jobs=num_cores)(delayed(segmentAndSave)(video, None, segmFrameInd) for segmFrameInd in segmFramesIndicesSubList)
            allSegmentsSubList = [(out[0],out[1]) for out in OupPutParallel]
            allSegments.append(allSegmentsSubList)
            numBlobs.append([out[2] for out in OupPutParallel])
    else:
        #splitting videoPaths list into sublists
        pathsSubLists = [videoPaths[i:i+num_cores] for i in range(0,len(videoPaths),num_cores)]

        for pathsSubList in pathsSubLists:
            OupPutParallel = Parallel(n_jobs=num_cores)(delayed(segmentAndSave)(video, path, None) for path in pathsSubList)
            allSegmentsSubList = [(out[0],out[1]) for out in OupPutParallel]
            allSegments.append(allSegmentsSubList)
            numBlobs.append([out[2] for out in OupPutParallel])

    allSegments = flatten(allSegments)
    maxNumBlobs = max(flatten(numBlobs))
    allSegments = sorted(allSegments, key=lambda x: x[0][0])
