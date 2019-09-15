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
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)

import os
import sys
import gc

import numpy as np
import multiprocessing
import cv2
import h5py
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy import ndimage
from confapp import conf

from idtrackerai.blob import Blob
from idtrackerai.utils.py_utils import flatten, set_mkl_to_single_thread, set_mkl_to_multi_thread
from idtrackerai.utils.segmentation_utils import segment_frame, blob_extractor

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


def get_blobs_in_frame(cap, video, segmentation_thresholds,
                       max_number_of_blobs, frame_number, global_frame_number,
                       frame_number_in_video_path,
                       bounding_box_images_path, episode, video_path, pixels_path,
                       save_pixels, save_segmentation_image):
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
        where the segmentation fails. This frame is the frame of the episode if the video
        is chuncked.
    global_frame_number : int
        This is the frame number in the whole video. It will be different to the frame_number
        if the video is chuncked.


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
    if conf.SIGMA_GAUSSIAN_BLURRING is not None:
        frame = cv2.GaussianBlur(frame, (0, 0), conf.SIGMA_GAUSSIAN_BLURRING)

    try:
        if video.resolution_reduction != 1 and ret:
            frame = cv2.resize(frame, None,
                               fx=video.resolution_reduction,
                               fy=video.resolution_reduction,
                               interpolation=cv2.INTER_AREA)
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape)>2 else frame
        avIntensity = np.float32(np.mean(np.ma.array(frameGray,
                                                     mask=video.ROI==0)))
        segmentedFrame = segment_frame(frameGray / avIntensity,
                                       segmentation_thresholds['min_threshold'],
                                       segmentation_thresholds['max_threshold'],
                                       video.bkg, video.ROI, video.subtract_bkg)
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1,4)
        # ax[0].imshow(frameGray)
        # ax[1].imshow(video.bkg)
        # ax[2].imshow(segmentedFrame)
        # Fill holes in the segmented frame to avoid duplication of contours
        # segmentedFrame = ndimage.binary_fill_holes(segmentedFrame).astype('uint8')
        # plt.show()
        # input("Press Enter to continue...")
        # Find contours in the segmented image
        bounding_boxes, miniframes, centroids, \
            areas, pixels, contours, estimated_body_lengths = \
            blob_extractor(segmentedFrame, frameGray,
                           segmentation_thresholds['min_area'],
                           segmentation_thresholds['max_area'],
                           save_pixels, save_segmentation_image)
    except Exception as e:
        print(e)
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
        if save_segmentation_image == 'DISK':
            with h5py.File(bounding_box_images_path, 'a') as f1:
                f1.create_dataset(str(global_frame_number) + '-' + str(i),
                                  data=miniframes[i])
            miniframes[i] = None
        if save_pixels == 'DISK':
            with h5py.File(pixels_path, 'a') as f2:
                f2.create_dataset(str(global_frame_number) + '-' + str(i),
                                  data=pixels[i])
            pixels[i] = None

        blob = Blob(centroids[i],
                    contours[i],
                    areas[i],
                    bounding_box,
                    bounding_box_image=miniframes[i],
                    bounding_box_images_path=bounding_box_images_path,
                    estimated_body_length=estimated_body_lengths[i],
                    number_of_animals=video.number_of_animals,
                    frame_number=global_frame_number,
                    pixels=pixels[i],
                    pixels_path=pixels_path,
                    in_frame_index=i,
                    video_height=video.height,
                    video_width=video.width,
                    video_path=video_path,
                    frame_number_in_video_path=frame_number_in_video_path,
                    resolution_reduction=video.resolution_reduction)
        blobs_in_frame.append(blob)

    if len(centroids) > max_number_of_blobs:
        max_number_of_blobs = len(centroids)

    return blobs_in_frame, max_number_of_blobs


def frame_in_intervals(frame_number, intervals):
    """Returns True if a frame is inside of one of the frame intervals. Otherwise
    returns False

    Parameters
    ----------
    frame_number : int
        Number of the frame to be checked.
    intervals : list
        List of intervals where to check for the frame
    """
    for interval in intervals:
        if frame_number >= interval[0] and frame_number <= interval[1]:
            return True
    return False


def segment_episode(video, segmentation_thresholds,
                    path=None, episode_start_end_frames=None,
                    save_pixels=None,
                    save_segmentation_image=None):
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
    episode = video.in_which_episode(episode_start_end_frames[0])
    if video.paths_to_video_segments is None:
        video_path = video.video_path
    else:
        video_path = video.paths_to_video_segments[episode]
    bounding_box_images_path = None
    if save_segmentation_image == 'DISK':
        bounding_box_images_path = os.path.join(video._segmentation_data_folder,
                                                'episode_images_{}.hdf5'.format(str(episode)))
        if os.path.isfile(bounding_box_images_path):
            os.remove(bounding_box_images_path)

    pixels_path = None
    if save_pixels == 'DISK':
        pixels_path = os.path.join(video._segmentation_data_folder,
                                   'episode_pixels_{}.hdf5'.format(str(episode)))
        if os.path.isfile(pixels_path):
            os.remove(pixels_path)

    blobs_in_episode = []
    cap, number_of_frames_in_episode = get_videoCapture(video, path, episode_start_end_frames)
    max_number_of_blobs = 0
    frame_number = 0
    while frame_number < number_of_frames_in_episode:

        global_frame_number = episode_start_end_frames[0] + frame_number
        if video.paths_to_video_segments is None:
            frame_number_in_video_path = global_frame_number
        else:
            frame_number_in_video_path = frame_number
        if video.tracking_interval is None or frame_in_intervals(global_frame_number, video.tracking_interval):
            blobs_in_frame, max_number_of_blobs = \
                get_blobs_in_frame(cap, video, segmentation_thresholds,
                                   max_number_of_blobs, frame_number,
                                   global_frame_number,
                                   frame_number_in_video_path,
                                   bounding_box_images_path,
                                   episode, video_path, pixels_path,
                                   save_pixels, save_segmentation_image)
        else:
            ret, _ = cap.read()
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
    num_cpus = int(multiprocessing.cpu_count())
    num_jobs = conf.NUMBER_OF_JOBS_FOR_SEGMENTATION
    # num_jobs = 1
    if conf.NUMBER_OF_JOBS_FOR_SEGMENTATION is None:
        num_jobs = 1
    elif conf.NUMBER_OF_JOBS_FOR_SEGMENTATION < 0:
        num_jobs = (num_cpus + 1 + num_jobs)

    #init variables to store data
    blobs_in_video = []
    maximum_number_of_blobs_in_episode = []
    segmentation_thresholds = {'min_threshold': video.min_threshold,
                                'max_threshold': video.max_threshold,
                                'min_area': video.min_area,
                                'max_area': video.max_area}

    set_mkl_to_single_thread()
    if not video.paths_to_video_segments:
        logger.info('There is only one path, segmenting by frame indices')
        #Spliting episodes_start_end in sublists for parallel processing
        episodes_start_end_sublists = [video.episodes_start_end[i:i+num_jobs]
                                        for i in range(0,len(video.episodes_start_end),num_jobs)]

        for episodes_start_end_sublist in tqdm(episodes_start_end_sublists, desc='Segmentation progress'):
            OupPutParallel = Parallel(n_jobs=conf.NUMBER_OF_JOBS_FOR_SEGMENTATION)(
                                delayed(segment_episode)(video, segmentation_thresholds, None, episode_start_end_frames,
                                                         conf.SAVE_PIXELS, conf.SAVE_SEGMENTATION_IMAGE)
                                for episode_start_end_frames in episodes_start_end_sublist)
            blobs_in_episode = [out[0] for out in OupPutParallel]
            maximum_number_of_blobs_in_episode.append([out[1] for out in OupPutParallel])
            blobs_in_video.append(blobs_in_episode)
    else:
        #splitting videoPaths list into sublists
        pathsSubLists = [video.paths_to_video_segments[i: i + num_jobs]
                         for i in range(0,len(video.paths_to_video_segments), num_jobs)]
        episodes_start_end_sublists = [video.episodes_start_end[i:i+num_jobs]
                                       for i in range(0,len(video.episodes_start_end), num_jobs)]

        for pathsSubList, episodes_start_end_sublist in tqdm(list(zip(pathsSubLists, episodes_start_end_sublists)), desc='Segmentation progress'):
            OupPutParallel = Parallel(n_jobs=conf.NUMBER_OF_JOBS_FOR_SEGMENTATION)(
                                delayed(segment_episode)(video, segmentation_thresholds, path, episode_start_end_frames,
                                                         conf.SAVE_PIXELS, conf.SAVE_SEGMENTATION_IMAGE)
                                for path, episode_start_end_frames in zip(pathsSubList, episodes_start_end_sublist))
            blobs_in_episode = [out[0] for out in OupPutParallel]
            maximum_number_of_blobs_in_episode.append([out[1] for out in OupPutParallel])
            blobs_in_video.append(blobs_in_episode)
    set_mkl_to_multi_thread()
    video._maximum_number_of_blobs = max(flatten(maximum_number_of_blobs_in_episode))
    #blobs_in_video is flattened to obtain a list of blobs per episode and then the list of all blobs
    blobs_in_video = flatten(flatten(blobs_in_video))
    return blobs_in_video


def resegment(video, frame_number, list_of_blobs,
              new_segmentation_thresholds):
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
    if video.paths_to_video_segments is None:
        video_path = video.video_path
    else:
        video_path = video.paths_to_video_segments[episode]
    if not video.paths_to_video_segments:
        cap, _ = get_videoCapture(video, None, video.episodes_start_end[episode_number])
        cap.set(1, frame_number)
        frame_number_in_video_path = frame_number
    else:
        path = video.paths_to_video_segments[episode_number]
        cap, _ = get_videoCapture(video, path, None)
        start = video.episodes_start_end[episode_number][0]
        cap.set(1,frame_number - start)
        frame_number_in_video_path = frame_number-start

    episode = video.in_which_episode(frame_number)
    bounding_box_images_path = None
    if conf.SAVE_SEGMENTATION_IMAGE == 'DISK':
        bounding_box_images_path = os.path.join(video._segmentation_data_folder,
                                    's_images_{}.hdf5'.format(str(episode)))
        with h5py.File(bounding_box_images_path, 'a') as f:
            images_to_delete = [d_name for d_name in f.keys()
                                if d_name.split('-')[0] == str(frame_number)]
            for im in images_to_delete:
                del f[im]

    pixels_path = None
    if conf.SAVE_PIXELS == 'DISK':
        pixels_path = os.path.join(video._segmentation_data_folder,
                                   'episode_pixels_{}.hdf5'.format(str(episode)))
        with h5py.File(pixels_path, 'a') as f:
            pixels_to_delete = [d_name for d_name in f.keys()
                                if d_name.split('-')[0] == str(frame_number)]
            for px in pixels_to_delete:
                del f[px]


    blobs_in_resegmanted_frame, \
        number_of_blobs_in_frame = \
        get_blobs_in_frame(cap, video, new_segmentation_thresholds,
                           0, frame_number, frame_number,
                           frame_number_in_video_path,
                           bounding_box_images_path, episode,
                           video_path, pixels_path,
                           conf.SAVE_PIXELS, conf.SAVE_SEGMENTATION_IMAGE)
    list_of_blobs.blobs_in_video[frame_number] = blobs_in_resegmanted_frame
    cap.release()
    cv2.destroyAllWindows()
    return number_of_blobs_in_frame
