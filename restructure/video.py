from __future__ import absolute_import, division, print_function
#from collections import namedtuple
import itertools
import numpy as np
import os
import glob
try:
    import cPickle as pickle
except:
    import pickle

from natsort import natsorted
import cv2

AVAILABLE_VIDEO_EXTENSION = ['.avi', '.mp4']
SUPPORTED_ANIMAL_TYPES = ['fish', 'fly', 'ant']
FRAMES_PER_EPISODE = 500 #long videos are divided into chunks. This is the number of frame per chunk

class Video(object):
    def __init__(self, video_path = None, animal_type = None, num_animals = None, bkg = None, subtract_bkg = False, ROI = None, apply_ROI = False):
        self._video_path = video_path #string: path to the video
        self._animal_type = animal_type #string: type of animals to be tracked in the video
        self._num_animals = num_animals #int: number of animals in the video
        self._episodes_start_end = None #list of lists: starting and ending frame per chunk [video is split for parallel computation]
        self.bkg = bkg #matrix [shape = shape of a frame] background used to do bkg subtraction
        self.subtract_bkg = subtract_bkg #boolean: True if the user specifies to subtract the background
        self.ROI = ROI #matrix [shape = shape of a frame] 255 are valid (part of the ROI) pixels and 0 are invalid according to openCV convention
        self.apply_ROI = apply_ROI #boolean: True if the user applies a ROI to the video
        self._has_preprocessing_parameters = False #boolean: True once the preprocessing parameters (max/min area, max/min threshold) are set and saved
        self._max_number_of_blobs = None #int: the maximum number of blobs detected in the video
        self._blobs_path = None #string: path to the saved list of blob objects
        self._has_been_preprocessed = None #boolean: True if a video has been fragmented in a past session
        self._global_fragments_path = None #string: path to saved list of global fragments
        self._has_been_pretrained = None
        self._pretraining_path = None

    @property
    def video_path(self):
        return self._video_path

    @video_path.setter
    def video_path(self, value):
        video_name, video_extension = os.path.splitext(value)
        if video_extension in AVAILABLE_VIDEO_EXTENSION:
            self._video_path = value
            #get video folder
            self._video_folder = os.path.dirname(self._video_path)
            #collect some info on the video (resolution, number of frames, ..)
            self.get_info()
            #create a folder in which preprocessing data will be saved
            self.create_preprocessing_folder()
            #give a unique name (wrt the video)
            self._path_to_video_object = os.path.join(self._preprocessing_folder, 'video_object.npy')
        else:
            raise ValueError("Supported video extensions are ", AVAILABLE_VIDEO_EXTENSION)

    @property
    def animal_type(self):
        return self._animal_type

    @property
    def num_animals(self):
        return self._num_animals

    @animal_type.setter
    def animal_type(self, value):
        if value in SUPPORTED_ANIMAL_TYPES:
            self._animal_type = value
        else:
            raise ValueError("The supported animal types are " , SUPPORTED_ANIMAL_TYPES)

    def check_split_video(self):
        """If the video is divided in chunks retrieves the path to each chunk"""
        paths_to_video_segments = scanFolder(self.video_path)

        if len(paths_to_video_segments) > 1:
            return paths_to_video_segments
        else:
            return None

    def get_info(self):
        self._paths_to_video_segments = self.check_split_video()
        cap = cv2.VideoCapture(self._video_path)
        self._width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        self._height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        if self._paths_to_video_segments is None:
            self._num_frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            self.get_episodes()
        else:
            chunks_lengths = [int(cv2.VideoCapture(chunk).get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)) for chunk in self._paths_to_video_segments]
            self._episodes_start_end = [(np.sum(chunks_lengths[:i-1], dtype = np.int), np.sum(chunks_lengths[:i])-1) for i in range(1,len(chunks_lengths)+1)]
            self._num_frames = np.sum(chunks_lengths)
            self._num_episodes = len(self._paths_to_video_segments)
        cap.release()

    def create_preprocessing_folder(self):
        """If it does not exist creates a folder called preprocessing
        in the video folder"""
        self._preprocessing_folder = os.path.join(self._video_folder, 'preprocessing')
        if not os.path.isdir(self._preprocessing_folder):
            os.makedirs(self._preprocessing_folder)
            print("the folder " + self._preprocessing_folder + " has been created")

    def create_training_and_session_folder(self):
        """Creates a folder named training in video_folder and a folder session_num
        where num is the session number and it is created everytime one starts
        training a network for a certain video_path
        """
        self._training_path = os.path.join(self._video_folder, 'training')
        if not os.path.isdir(self._training_path):
            self._session_path = os.path.join(self._training_path, 'session_1')
        else:
            self._sessions_paths = glob.glob(self._training_path +"/session*")
            last_session_index = get_last_training_session_index(self._sessions_paths)
            new_session_index = str(last_session_index + 1)
            self._session_path = os.path.join(self._training_path + "/session_" + new_session_index)

        os.makedirs(self._session_path)
        print("the folder " + self._training_path + " has been created")

    def create_pretraining_folder(self):
        """Creates a folder named pretraining in video_folder where the model
        trained during the pretraining is stored
        """
        self._pretraining_path = os.path.join(self._session_path, 'pretraining')
        if not os.path.isdir(self._pretraining_path):
            os.makedirs(self._pretraining_path)


    def get_episodes(self):
        """Split video in episodes (chunks) of 500 frames
        for parallelisation"""
        starting_frames = np.arange(0, self._num_frames, FRAMES_PER_EPISODE)
        ending_frames = np.hstack((starting_frames[1:] - 1, self._num_frames - 1))
        self._episodes_start_end =zip(starting_frames, ending_frames)
        self._num_episodes = len(starting_frames)

    def in_which_episode(self, frame_index):
        """Check to which episode a frame index belongs in time"""
        episode_number = [i for i, episode_start_end in enumerate(self._episodes_start_end) if episode_start_end[0] <= frame_index and episode_start_end[1] >= frame_index]
        if episode_number:
            return episode_number[0]
        else:
            return None

    @property
    def min_threshold(self):
        return self._min_threshold

    @property
    def max_threshold(self):
        return self._max_threshold

    @property
    def min_area(self):
        return self._min_area

    @property
    def max_area(self):
        return self._max_area

    @property
    def resize(self):
        return self._resize

    @property
    def blobs_path(self):
        """get the path to save the blob collection after segmentation.
        It checks that the segmentation has been succesfully performed"""
        if self._has_been_segmented:
            self._blobs_path = os.path.join(self._preprocessing_folder, 'blobs_collection.npy')
        return self._blobs_path

    @property
    def global_fragments_path(self):
        """get the path to save the list of global fragments after
        fragmentation"""
        if self._has_been_preprocessed:
            self._global_fragments_path = os.path.join(self._preprocessing_folder, 'global_fragments.npy')
        return self._global_fragments_path

    def save(self):
        """save class"""
        print("saving video object")
        np.save(self._path_to_video_object, self)

def get_num_frame(path):
    cap = cv2.VideoCapture(path)
    num_frame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    cap.release()
    return num_frame

def scanFolder(path):
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    folder = os.path.dirname(path)
    paths = glob.glob(folder + "/*" + extension)
    filename = filename.split("_")[:-1][0]
    paths = natsorted([path for path in paths if filename in path])
    return paths

def get_last_training_session_index(subFolders):
    """gets path to the last training session
    """
    if len(subFolders) == 0:
        lastIndex = 0
    else:
        subFolders = natsorted(subFolders)[::-1]
        lastIndex = int(subFolders[0].split('_')[-1])
    return lastIndex


if __name__ == "__main__":

    video = Video()
    video.video_path = '/home/lab/Desktop/TF_models/IdTrackerDeep/videos/Cafeina5pecesShort/Caffeine5fish_20140206T122428_1.avi'
