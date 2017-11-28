from __future__ import absolute_import, division, print_function
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

def get_videoCapture(video, path, segmFrameInd):
    if segmFrameInd == None:
        cap = cv2.VideoCapture(path)
        # print 'Segmenting video %s' % path
        video_name = os.path.basename(path)
        filename, extension = os.path.splitext(video_name)
        number_of_frames_in_segment = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    elif path is None:
        cap = cv2.VideoCapture(video.video_path)
        number_of_frames_in_segment = segmFrameInd[1] - segmFrameInd[0] + 1
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,segmFrameInd[0])

    return cap, number_of_frames_in_segment

def get_blobs_in_frame(cap, video, segmentation_thresholds, max_number_of_blobs, counter):
    blobs_in_frame = []
    #Get frame from video file
    ret, frame = cap.read()
    if video.resolution_reduction != 1 and ret:
        frame = cv2.resize(frame, None, fx = video.resolution_reduction, fy = video.resolution_reduction, interpolation = cv2.INTER_CUBIC)

    try:
        #Color to gray scale
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avIntensity = np.float32(np.mean(frameGray))
        segmentedFrame = segmentVideo(frameGray/avIntensity, segmentation_thresholds['min_threshold'], segmentation_thresholds['max_threshold'], video.bkg, video.ROI, video.subtract_bkg)
        # Fill holes in the segmented frame to avoid duplication of contours
        segmentedFrame = ndimage.binary_fill_holes(segmentedFrame).astype('uint8')
        # Find contours in the segmented image
        bounding_boxes, miniframes, centroids, areas, pixels, contours, estimated_body_lengths = blobExtractor(segmentedFrame, frameGray,
                                                                                                segmentation_thresholds['min_area'], segmentation_thresholds['max_area'])
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

    return blobs_in_frame, max_number_of_blobs

def segmentAndSave(video, segmentation_thresholds, path = None, segmFrameInd = None):
    blobs_in_episode = []

    cap, number_of_frames_in_segment = get_videoCapture(video, path, segmFrameInd)
    max_number_of_blobs = 0
    frame_number = 0

    while frame_number < number_of_frames_in_segment:
        blobs_in_frame, maximum_number_of_blobs = get_blobs_in_frame(cap, video, segmentation_thresholds, max_number_of_blobs, frame_number)
        #store all the blobs encountered in the episode
        blobs_in_episode.append(blobs_in_frame)
        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()
    gc.collect()
    return blobs_in_episode, max_number_of_blobs

def segment(video):
    # avoid computing with all the cores in very large videos:
    # num_cores = multiprocessing.cpu_count()
    num_cores = 6
    #init variables to store data
    blobs_in_video = []
    number_of_blobs = []
    #videoPaths is used to check if the video was previously split or not (it is either None or
    #a list of paths. Check attribute paths_to_video_segments in Video)
    segmentation_thresholds = {'min_threshold': video.min_threshold,
                                'max_threshold': video.max_threshold,
                                'min_area': video.min_area,
                                'max_area': video.max_area}
    videoPaths = video.paths_to_video_segments
    if not videoPaths:
        print('**************************************')
        print('There is only one path, segmenting by frame indices')
        print('**************************************')
        #Define list of starting and ending frames
        segmFramesIndices = video.episodes_start_end
        #Spliting frames list into sublists
        segmFramesIndicesSubLists = [segmFramesIndices[i:i+num_cores] for i in range(0,len(segmFramesIndices),num_cores)]
        for segmFramesIndicesSubList in tqdm(segmFramesIndicesSubLists, desc = 'Segmentation progress'):
            OupPutParallel = Parallel(n_jobs=num_cores)(delayed(segmentAndSave)(video, segmentation_thresholds, None, segmFrameInd) for segmFrameInd in segmFramesIndicesSubList)
            blobs_in_episode = [out[0] for out in OupPutParallel]
            number_of_blobs.append([out[1] for out in OupPutParallel])
            blobs_in_video.append(blobs_in_episode)
    else:
        #splitting videoPaths list into sublists
        pathsSubLists = [videoPaths[i:i+num_cores] for i in range(0,len(videoPaths),num_cores)]

        for pathsSubList in tqdm(pathsSubLists, desc = 'Segmentation progress'):
            OupPutParallel = Parallel(n_jobs=num_cores)(delayed(segmentAndSave)(video, segmentation_thresholds, path, None) for path in pathsSubList)
            blobs_in_episode = [out[0] for out in OupPutParallel]
            number_of_blobs.append([out[1] for out in OupPutParallel])
            blobs_in_video.append(blobs_in_episode)

    video._maximum_number_of_blobs = max(flatten(number_of_blobs))
    print("video.maximum_number_of_blobs ", video.maximum_number_of_blobs)
    #blobs_in_video is flattened to obtain a list of blobs per episode and then the list of all blobs
    blobs_in_video = flatten(flatten(blobs_in_video))
    return blobs_in_video

def resegment(video, frame_number, list_of_blobs, new_preprocessing_parameters):
    episode_number = video.in_which_episode(frame_number)
    if not video.paths_to_video_segments:
        cap, _ = get_videoCapture(video, None, video.episodes_start_end[episode_number])
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,frame_number)
    else:
        path = video.paths_to_video_segments[episode_number]
        cap, _ = get_videoCapture(video, path, None)
        start = video.episodes_start_end[episode_number][0]
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,frame_number - start)

    blobs_in_resegmanted_frame, maximum_number_of_blobs = get_blobs_in_frame(cap, video, new_preprocessing_parameters, 0, frame_number)
    list_of_blobs.blobs_in_video[frame_number] = blobs_in_resegmanted_frame
    cap.release()
    cv2.destroyAllWindows()
    return maximum_number_of_blobs
