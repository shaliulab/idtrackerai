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


from __future__ import absolute_import, division, print_function
import os
import sys
import numpy as np
import multiprocessing
import cv2
import cPickle as pickle
from joblib import Parallel, delayed
import gc
from tqdm import tqdm
from scipy import ndimage
from idtrackerai.blob import Blob
from idtrackerai.utils.py_utils import  flatten
from idtrackerai.utils.video_utils import segment_frame, blob_extractor
if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.segmentation")

"""
The segmentation module
"""

def get_videoCapture(video, path, episode_start_end_frames):
    """Gives the VideoCapture (OpenCV) object to read the frames for the segmentation
    and the number of frames to read. If `episode_start_end_frames` is None then a `path` must be
    given as the video is assumed to be splitted in different files (episodes). If the `path`
    is None the video is assumed to be in a single file and the path is read from
    `video`, then `episode_start_end_frames` must be give.

    Parameters
    ----------
    video : <Video object>
        Object collecting all the parameters of the video and paths for saving and loading
    path : string
        Path to the video file from where to get the VideoCapture (OpenCV) object
    episode_start_end_frames : tuple
        Tuple (starting_frame, ending_frame) indicanting the start and end of the episode
        when the video is given in a single file

    Returns
    -------
    cap : <VideoCapture object>
        OpenCV object used to read the frames of the video
    number_of_frames_in_episode : int
        Number of frames in the episode of video being segmented
    """
    if path is not None:
        cap = cv2.VideoCapture(path)
        number_of_frames_in_episode = int(cap.get(7))
    elif path is None:
        cap = cv2.VideoCapture(video.video_path)
        number_of_frames_in_episode = episode_start_end_frames[1] - episode_start_end_frames[0] + 1
        cap.set(1,episode_start_end_frames[0])

    return cap, number_of_frames_in_episode

def get_blobs_in_frame(cap, video, segmentation_thresholds, max_number_of_blobs, frame_number):
    """Segments a frame read from `cap` according to the preprocessing parameters
    in `video`. Returns a list `blobs_in_frame` with the Blob objects in the frame
    and the `max_number_of_blobs` found in the video so far. Frames are segmented
    in gray scale.

    Parameters
    ----------
    cap : <VideoCapture object>
        OpenCV object used to read the frames of the video
    video : <Video object>
        Object collecting all the parameters of the video and paths for saving and loading
    segmentation_thresholds : dict
        Dictionary with the thresholds used for the segmentation: `min_threshold`,
        `max_threshold`, `min_area`, `max_area`
    max_number_of_blobs : int
        Maximum number of blobs found in the whole video so far in the segmentation process
    frame_number : int
        Number of the frame being segmented. It is used to print in the terminal the frames
        where the segmentation fails

    Returns
    -------
    blobs_in_frame : list
        List of <Blob object> segmented in the current frame
    max_number_of_blobs : int
        Maximum number of blobs found in the whole video so far in the segmentation process

    See Also
    --------
    Video
    Blob
    segment_frame
    blob_extractor
    """
    blobs_in_frame = []
    ret, frame = cap.read()

    try:
        if video.resolution_reduction != 1 and ret:
            frame = cv2.resize(frame, None,
                                fx = video.resolution_reduction,
                                fy = video.resolution_reduction,
                                interpolation = cv2.INTER_CUBIC)
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avIntensity = np.float32(np.mean(frameGray))
        segmentedFrame = segment_frame(frameGray/avIntensity,
                                        segmentation_thresholds['min_threshold'],
                                        segmentation_thresholds['max_threshold'],
                                        video.bkg, video.ROI, video.subtract_bkg)
        # Fill holes in the segmented frame to avoid duplication of contours
        segmentedFrame = ndimage.binary_fill_holes(segmentedFrame).astype('uint8')
        # Find contours in the segmented image
        bounding_boxes, miniframes, centroids, \
        areas, pixels, contours, estimated_body_lengths = blob_extractor(segmentedFrame,
                                                                        frameGray,
                                                                        segmentation_thresholds['min_area'],
                                                                        segmentation_thresholds['max_area'])
    except:
        logger.info("An error occurred while reading frame number : %i" %frame_number)
        logger.info("ret: %s" %str(ret))
        logger.info("frame: %s" %str(frame))
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

def segment_episode(video, segmentation_thresholds, path = None, episode_start_end_frames = None):
    """Gets list of blobs segmented in every frame of the episode of the video
    given by `path` (if the video is splitted in different files) or by
    `episode_start_end_frames` (if the video is given in a single file)

    Parameters
    ----------
    video : <Video object>
        Object collecting all the parameters of the video and paths for saving and loading
    segmentation_thresholds : dict
        Dictionary with the thresholds used for the segmentation: `min_threshold`,
        `max_threshold`, `min_area`, `max_area`
    path : string
        Path to the video file from where to get the VideoCapture (OpenCV) object
    episode_start_end_frames : tuple
        Tuple (starting_frame, ending_frame) indicanting the start and end of the episode
        when the video is given in a single file

    Returns
    -------
    blobs_in_episode : list
        List of `blobs_in_frame` of the episode of the video being segmented
    max_number_of_blobs : int
        Maximum number of blobs found in the episode of the video being segmented

    See Also
    --------
    Video
    Blob
    get_videoCapture
    segment_frame
    blob_extractor
    """
    blobs_in_episode = []

    cap, number_of_frames_in_episode = get_videoCapture(video, path, episode_start_end_frames)
    max_number_of_blobs = 0
    frame_number = 0
    while frame_number < number_of_frames_in_episode:
        global_frame_number = episode_start_end_frames[0] + frame_number
        if video.tracking_interval is None or global_frame_number >= video.tracking_interval[0] and global_frame_number <= video.tracking_interval[1]:
            blobs_in_frame, max_number_of_blobs = get_blobs_in_frame(cap, video,
                                                                    segmentation_thresholds,
                                                                    max_number_of_blobs,
                                                                    frame_number)
        else:
            ret, frame = cap.read()
            blobs_in_frame = []
        #store all the blobs encountered in the episode
        blobs_in_episode.append(blobs_in_frame)
        frame_number += 1
    cap.release()
    #cv2.destroyAllWindows()
    gc.collect()
    return blobs_in_episode, max_number_of_blobs

def segment(video):
    """Segment the video giving a list of `blobs` for every frame in the video.
    If a video is given as a set of files (episodes), those files are used to
    parallelise the segmentation process. If a video is given in a single file
    the list of indices `video.episodes_start_end` is used for the parallelisation

    Parameters
    ----------
    video : <Video object>
        Object collecting all the parameters of the video and paths for saving and loading

    Returns
    -------
    blobs_in_video : list
        List of `blobs_in_frame` for all the frames of the video

    See Also
    --------
    segment_episode
    """
    # avoid computing with all the cores in very large videos. It fills the RAM.
    # num_cores = multiprocessing.cpu_count()
    num_cores = int(np.ceil(multiprocessing.cpu_count() / 2))
    # num_cores = 1
    # if video.number_of_episodes < num_cores:
    #     num_cores = 1
    #init variables to store data
    blobs_in_video = []
    maximum_number_of_blobs_in_episode = []
    segmentation_thresholds = {'min_threshold': video.min_threshold,
                                'max_threshold': video.max_threshold,
                                'min_area': video.min_area,
                                'max_area': video.max_area}
    if not video.paths_to_video_segments:
        logger.info('There is only one path, segmenting by frame indices')
        #Spliting episodes_start_end in sublists for parallel processing
        episodes_start_end_sublists = [video.episodes_start_end[i:i+num_cores]
                                        for i in range(0,len(video.episodes_start_end),num_cores)]
        for episodes_start_end_sublist in tqdm(episodes_start_end_sublists, desc = 'Segmentation progress'):
            OupPutParallel = Parallel(n_jobs=num_cores)(
                                delayed(segment_episode)(video, segmentation_thresholds, None, episode_start_end_frames)
                                for episode_start_end_frames in episodes_start_end_sublist)
            blobs_in_episode = [out[0] for out in OupPutParallel]
            maximum_number_of_blobs_in_episode.append([out[1] for out in OupPutParallel])
            blobs_in_video.append(blobs_in_episode)
    else:
        #splitting videoPaths list into sublists
        pathsSubLists = [video.paths_to_video_segments[i:i+num_cores]
                            for i in range(0,len(video.paths_to_video_segments),num_cores)]
        episodes_start_end_sublists = [video.episodes_start_end[i:i+num_cores]
                                        for i in range(0,len(video.episodes_start_end),num_cores)]

        for pathsSubList, episodes_start_end_sublist in tqdm(zip(pathsSubLists, episodes_start_end_sublists), desc = 'Segmentation progress'):
            OupPutParallel = Parallel(n_jobs=num_cores)(
                                delayed(segment_episode)(video, segmentation_thresholds, path, episode_start_end_frames)
                                for path, episode_start_end_frames in zip(pathsSubList, episodes_start_end_sublist))
            blobs_in_episode = [out[0] for out in OupPutParallel]
            maximum_number_of_blobs_in_episode.append([out[1] for out in OupPutParallel])
            blobs_in_video.append(blobs_in_episode)

    video._maximum_number_of_blobs = max(flatten(maximum_number_of_blobs_in_episode))
    #blobs_in_video is flattened to obtain a list of blobs per episode and then the list of all blobs
    blobs_in_video = flatten(flatten(blobs_in_video))
    return blobs_in_video

def resegment(video, frame_number, list_of_blobs, new_segmentation_thresholds):
    """Updates the `list_of_blobs` for a particular `frame_number` by performing
    a segmentation with `new_segmentation_thresholds`. This function is called for
    the frames in which the number of blobs is higher than the number of animals
    stated by the user.

    Parameters
    ----------
    video : <Video object>
        Object collecting all the parameters of the video and paths for saving and loading
    frame_number : int
        Number of the frame to update with the new segmentation
    list_of_blobs : <ListOfBlobs object>
        Object containing the list of blobs segmented in the video in each frame
    new_segmentation_thresholds : dict
        Dictionary with the thresholds used for the new segmentation: `min_threshold`,
        `max_threshold`, `min_area`, `max_area`

    Returns
    -------
    number_of_blobs_in_frame : int
        Number of blobs found in the frame

    See Also
    --------
    get_videoCapture
    get_blobs_in_frame
    """
    episode_number = video.in_which_episode(frame_number)
    if not video.paths_to_video_segments:
        cap, _ = get_videoCapture(video, None, video.episodes_start_end[episode_number])
        cap.set(1,frame_number)
    else:
        path = video.paths_to_video_segments[episode_number]
        cap, _ = get_videoCapture(video, path, None)
        start = video.episodes_start_end[episode_number][0]
        cap.set(1,frame_number - start)

    blobs_in_resegmanted_frame, \
    number_of_blobs_in_frame = get_blobs_in_frame(cap, video, new_segmentation_thresholds,
                                                0, frame_number)
    list_of_blobs.blobs_in_video[frame_number] = blobs_in_resegmanted_frame
    cap.release()
    cv2.destroyAllWindows()
    return number_of_blobs_in_frame