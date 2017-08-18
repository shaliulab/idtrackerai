from __future__ import division

# Import standard libraries
import os
import sys
import numpy as np
import multiprocessing

# Import third party libraries
import cv2
import cPickle as pickle
from joblib import Parallel, delayed
import gc
from tqdm import tqdm
from scipy import ndimage

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
        # print 'Segmenting video %s' % path
        video_name = os.path.basename(path)
        filename, extension = os.path.splitext(video_name)
        numSegment = int(filename.split('_')[-1])
        numFrames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        counter = 0
    elif path is None:
        cap = cv2.VideoCapture(video.video_path)
        numSegment = video.in_which_episode(segmFrameInd[0] + 1)
        # print 'Segment video %s from frame %i to frame %i (segment %i)' %(path, segmFrameInd[0], segmFrameInd[1], numSegment)
        numFrames = segmFrameInd[1] - segmFrameInd[0] + 1
        counter = 0
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,segmFrameInd[0])
    max_number_of_blobs = 0
    max_body_length = 0

    while counter < numFrames:
        blobs_in_frame = []
        #Get frame from video file
        ret, frame = cap.read()
        if video.resolution_reduction != 1 and ret:
            frame = cv2.resize(frame, None, fx = video.resolution_reduction, fy = video.resolution_reduction, interpolation = cv2.INTER_CUBIC)
        try:
            #Color to gray scale
            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avIntensity = np.float32(np.mean(frameGray))
            segmentedFrame = segmentVideo(frameGray/avIntensity, video._min_threshold, video._max_threshold, video.bkg, video.ROI, video.subtract_bkg)
            # Fill holes in the segmented frame to avoid duplication of contours
            segmentedFrame = ndimage.binary_fill_holes(segmentedFrame).astype('uint8')
            # Find contours in the segmented image
            bounding_boxes, miniframes, centroids, areas, pixels, contours, estimated_body_lengths = blobExtractor(segmentedFrame, frameGray,
                                                                                                    video._min_area, video._max_area,
                                                                                                    video._height, video._width)
        except:
            print("frame number, ", counter)
            print("ret, ", ret)
            print("frame, ", frame)
            bounding_boxes = []
            miniframes = []
            centroids = []
            areas = []
            pixels = []
            contours = []

        # create blob objects
        for i, bounding_box in enumerate(bounding_boxes):
            blob = Blob(centroids[i],
                        contours[i],
                        areas[i],
                        bounding_box,
                        bounding_box_image = miniframes[i],
                        estimated_body_length = estimated_body_lengths[i],
                        pixels = pixels[i],
                        number_of_animals = video.number_of_animals)
            blobs_in_frame.append(blob)

        if len(centroids) > max_number_of_blobs:
            max_number_of_blobs = len(centroids)

        #store all the blobs encountered in the episode
        blobs_in_episode.append(blobs_in_frame)
        counter += 1

    cap.release()
    cv2.destroyAllWindows()
    gc.collect()
    return blobs_in_episode, max_number_of_blobs, max_body_length

def segment(video):
    # avoid computing with all the cores in very large videos:
    num_cores = multiprocessing.cpu_count()
    # num_cores = 4
    #init variables to store data
    blobs_in_video = []
    number_of_blobs = []
    #videoPaths is used to check if the video was previously split or not (it is either None or
    #a list of paths. Check attribute _paths_to_video_segments in Video)
    videoPaths = video._paths_to_video_segments
    if not videoPaths:
        print '**************************************'
        print 'There is only one path, segmenting by frame indices'
        print '**************************************'
        #Define list of starting and ending frames
        segmFramesIndices = video._episodes_start_end
        #Spliting frames list into sublists
        segmFramesIndicesSubLists = [segmFramesIndices[i:i+num_cores] for i in range(0,len(segmFramesIndices),num_cores)]

        for segmFramesIndicesSubList in tqdm(segmFramesIndicesSubLists, desc = 'Segmentation progress'):
            OupPutParallel = Parallel(n_jobs=num_cores)(delayed(segmentAndSave)(video, None, segmFrameInd) for segmFrameInd in segmFramesIndicesSubList)
            blobs_in_episode = [out[0] for out in OupPutParallel]
            number_of_blobs.append([out[1] for out in OupPutParallel])
            blobs_in_video.append(blobs_in_episode)
    else:
        #splitting videoPaths list into sublists
        pathsSubLists = [videoPaths[i:i+num_cores] for i in range(0,len(videoPaths),num_cores)]

        for pathsSubList in tqdm(pathsSubLists, desc = 'Segmentation progress'):
            OupPutParallel = Parallel(n_jobs=num_cores)(delayed(segmentAndSave)(video, path, None) for path in pathsSubList)
            blobs_in_episode = [out[0] for out in OupPutParallel]
            number_of_blobs.append([out[1] for out in OupPutParallel])
            blobs_in_video.append(blobs_in_episode)

    video._maximum_number_of_blobs = max(flatten(number_of_blobs))
    #blobs_in_video is flattened to obtain a list of blobs per episode and then the list of all blobs
    blobs_in_video = flatten(flatten(blobs_in_video))
    # np.save(video.blobs_path, blobs_in_video, allow_pickle = True)
    return blobs_in_video
