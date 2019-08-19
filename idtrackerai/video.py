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
import time
import glob
from tempfile import mkstemp
from shutil import move, rmtree

import numpy as np
import cv2
from natsort import natsorted
from confapp import conf


from idtrackerai.postprocessing.assign_them_all import close_trajectories_gaps

if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.video")

class Video(object):
    """
    Attributes
    ----------

    video_path : str
        path to the video (if the video is split in portions, the path to one
        of the portions)
    number_of_animals : int
        number of animals to track
    episodes_start_end : list
        list of lists of the form [start, end] to parallelise processes
    original_bkg : ndarray
        model of the background with size the frame size
    bkg : ndarray
        resized bkg according to :attr:`.Video.resolution_reduction`
    subtract_bkg : bool
        True if bkg subtraction is performed
    original_ROI : ndarray
        ROI of the size of frame
    ROI : ndarray
        resized ROI according to :attr:`.Video.resolution_reduction`
    apply_ROI : bool
        True if a non-trivial ROI has been specified
    min_threshold : int
        minimum intensity threshold
    max_threshold : int
        maximum intensity threshold
    min_area : int
        minimum blob area
    max_area : int
        maximum blob area
    resegmentation_parameters : list
        segmentation parameters specific to a frame
    has_preprocessing_parameters : bool
        True if the preprocessing has been concluded succesfully
    maximum_number_of_blobs : int
        Maximum number of blobs segmented from a single frame
    resolution_reduction : float
        resolution reduction to be applied to the original frame
    erosion_kernel_size : int
        size of the kernel used to erode blobs while solving the crossings. See
        :mod:`~assign_them_all`
    """

    def __init__(self, video_path = None, open_multiple_files=False):
        logger.debug("Video object init")
        self._open_multiple_files = open_multiple_files
        self._video_path = video_path #string: path to the video
        self._number_of_animals = None #int: number of animals in the video
        self._episodes_start_end = None #list of lists: starting and ending frame per chunk [video is split for parallel computation]
        self._original_bkg = None #matrix [shape = shape of a frame] background used to do bkg subtraction
        self._bkg = None
        self._resolution_reduction = 1.
        self._subtract_bkg = None #boolean: True if the user specifies to subtract the background
        self._original_ROI = None #matrix [shape = shape of a frame] 255 are valid (part of the ROI) pixels and 0 are invalid according to openCV convention
        self._ROI = None
        self._apply_ROI = None #boolean: True if the user applies a ROI to the video
        self._min_threshold = conf.MIN_THRESHOLD_DEFAULT
        self._max_threshold = conf.MAX_THRESHOLD_DEFAULT
        self._min_area = conf.MIN_AREA_DEFAULT
        self._max_area = conf.MAX_AREA_DEFAULT
        self._resize = 1
        self._resegmentation_parameters = []
        self._tracking_interval = None
        self._has_preprocessing_parameters = False #boolean: True once the preprocessing parameters (max/min area, max/min threshold) are set and saved
        self._maximum_number_of_blobs = 0 #int: the maximum number of blobs detected in the video
        self._blobs_path = None #string: path to the saved list of blob objects
        self._blobs_path_segmented = None
        self._blobs_path_interpolated = None
        self._has_been_segmented = None
        self._has_been_preprocessed = None #boolean: True if a video has been fragmented in a past session
        self._preprocessing_folder = None
        self._images_folder = None
        self._global_fragments_path = None #string: path to saved list of global fragments
        self._has_been_pretrained = None
        self._pretraining_folder = None
        self._knowledge_transfer_model_folder = None
        self._identity_transfer = None
        self._tracking_with_knowledge_transfer = False
        self._percentage_of_accumulated_images = None
        self._first_accumulation_finished = None
        self._second_accumulation_finished = None
        self._has_been_assigned = None # ?
        self._has_duplications_solved = None
        self._has_crossings_solved = None
        self._has_trajectories = None
        self._has_trajectories_wo_gaps = None
        self._embeddings_folder = None # If embeddings are computed, the will be saved in this path
        self._first_frame_first_global_fragment = []
        self._segmentation_time = 0.
        self._crossing_detector_time = 0.
        self._fragmentation_time = 0.
        self._protocol1_time = 0.
        self._protocol2_time = 0.
        self._protocol3_pretraining_time = 0.
        self._protocol3_accumulation_time = 0.
        self._identify_time = 0.
        self._create_trajectories_time = 0.
        self._there_are_crossings = True
        self._track_wo_identities = False # Track without identities
        self._number_of_channels = 1
        self.individual_videos_folder = None
        if conf.SIGMA_GAUSSIAN_BLURRING is not None:
            self.sigma_gaussian_blurring = conf.SIGMA_GAUSSIAN_BLURRING

        # Flag to decide which type of interpolation is done. This flag is updated when we update a blob centroid
        self._is_centroid_updated = False

        logger.debug(f'Video(open_multiple_files={self.open_multiple_files})' )

    @property
    def is_centroid_updated(self):
        return self._is_centroid_updated

    @is_centroid_updated.setter
    def is_centroid_updated(self, value):
        self._is_centroid_updated = value

    @property
    def number_of_channels(self):
        return self._number_of_channels

    @property
    def episodes_start_end(self):
        return self._episodes_start_end

    @property
    def has_preprocessing_parameters(self):
        return self._has_preprocessing_parameters

    @property
    def preprocessing_folder(self):
        return self._preprocessing_folder

    @property
    def images_folder(self):
        return self._images_folder

    @property
    def has_been_preprocessed(self):
        return self._has_been_preprocessed

    @property
    def has_been_segmented(self):
        return self._has_been_segmented

    @property
    def crossings_detector_folder(self):
        return self._crossings_detector_folder

    @property
    def has_been_pretrained(self):
        return self._has_been_pretrained

    @property
    def previous_session_folder(self):
        return self._previous_session_folder

    @property
    def pretraining_folder(self):
        return self._pretraining_folder

    @property
    def first_accumulation_finished(self):
        return self._first_accumulation_finished

    @property
    def second_accumulation_finished(self):
        return self._second_accumulation_finished

    @property
    def accumulation_folder(self):
        return self._accumulation_folder

    @property
    def percentage_of_accumulated_images(self):
        return self._percentage_of_accumulated_images

    @property
    def has_been_assigned(self):
        return self._has_been_assigned

    @property
    def has_duplications_solved(self):
        return self._has_duplications_solved

    @property
    def has_crossings_solved(self):
        return self._has_crossings_solved

    @property
    def has_trajectories(self):
        return self._has_trajectories

    @property
    def has_trajectories_wo_gaps(self):
        return self._has_trajectories_wo_gaps

    @property
    def embeddings_folder(self):
        return self._embeddings_folder

    @property
    def session_folder(self):
        return self._session_folder

    @property
    def apply_ROI(self):
        return self._apply_ROI

    @property
    def original_ROI(self):
        return self._original_ROI

    @property
    def ROI(self):
        return self._ROI

    @property
    def subtract_bkg(self):
        return self._subtract_bkg

    @property
    def original_bkg(self):
        return self._original_bkg

    @property
    def bkg(self):
        return self._bkg

    @property
    def resolution_reduction(self):
        return self._resolution_reduction

    @resolution_reduction.setter
    def resolution_reduction(self, value):
        self._resolution_reduction = value
        fake_frame = np.ones((self._original_height, self._original_width)).astype('uint8')
        fake_frame = cv2.resize(fake_frame, None,
                                fx = value,
                                fy = value,
                                interpolation = cv2.INTER_AREA)
        self._height = fake_frame.shape[0]
        self._width = fake_frame.shape[1]
        if self.subtract_bkg:
            self._bkg = cv2.resize(self.original_bkg, None,
                                            fx = value,
                                            fy = value,
                                            interpolation = cv2.INTER_AREA)
        if self.apply_ROI or self.original_ROI is not None:
            self._ROI = cv2.resize(self.original_ROI, None,
                                            fx = value,
                                            fy = value,
                                            interpolation = cv2.INTER_AREA)

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
    def resegmentation_parameters(self):
        return self._resegmentation_parameters

    @property
    def tracking_interval(self):
        """ Tuple with the starting and ending frame on which the tracking will
        be performed.
        """
        return self._tracking_interval

    @property
    def resize(self):
        return self._resize

    @property
    def blobs_path(self):
        """get the path to save the blob collection after segmentation.
        It checks that the segmentation has been succesfully performed"""
        if self.preprocessing_folder is None: return None
        self._blobs_path = os.path.join(self.preprocessing_folder, 'blobs_collection.npy')
        return self._blobs_path

    @property
    def blobs_path_segmented(self):
        """get the path to save the blob collection after segmentation.
        It checks that the segmentation has been succesfully performed"""
        self._blobs_path_segmented = os.path.join(self.preprocessing_folder, 'blobs_collection_segmented.npy')
        return self._blobs_path_segmented

    @property
    def blobs_path_interpolated(self):
        self._blobs_path_interpolated = os.path.join(self.preprocessing_folder, 'blobs_collection_interpolated.npy')
        return self._blobs_path_interpolated

    @property
    def global_fragments_path(self):
        """get the path to save the list of global fragments after
        fragmentation"""
        self._global_fragments_path = os.path.join(self.preprocessing_folder, 'global_fragments.npy')
        return self._global_fragments_path

    @property
    def fragments_path(self):
        """get the path to save the list of global fragments after
        fragmentation"""
        self._fragments_path = os.path.join(self.preprocessing_folder, 'fragments.npy')
        return self._fragments_path

    @property
    def erosion_kernel_size(self):
        return self._erosion_kernel_size

    @property
    def path_to_video_object(self):
        return self._path_to_video_object

    @property
    def git_commit(self):
        return '0'
        return self._git_commit

    def save(self):
        """save class"""
        logger.info("saving video object in %s" %self.path_to_video_object)
        logger.debug(f'Video.save(open_multiple_files={self.open_multiple_files})')
        np.save(self.path_to_video_object, self)

    @property
    def knowledge_transfer_model_folder(self):
        return self._knowledge_transfer_model_folder

    @knowledge_transfer_model_folder.setter
    def knowledge_transfer_model_folder(self, new_kt_model_path):
        if new_kt_model_path:
            subfolders = glob.glob(os.path.join(new_kt_model_path, "*"))
            if os.path.join(new_kt_model_path, "conv") in subfolders and os.path.join(new_kt_model_path, "softmax") in subfolders:
                self._knowledge_transfer_model_folder = new_kt_model_path
            else:
                raise ValueError("The model folders " + os.path.join(new_kt_model_path, "conv") + " and " + os.path.join(new_kt_model_path, "softmax") + " are missing")
        else:
            self._knowledge_transfer_model_folder = None

    @property
    def identity_transfer(self):
        return self._identity_transfer

    def is_identity_transfer_possible(self):
        return self.number_of_animals == self.knowledge_transfer_info_dict['number_of_animals']

    def check_and_set_identity_transfer_if_possible(self):
        if conf.KNOWLEDGE_TRANSFER_FOLDER_IDCNN is None:
            raise ValueError("Need to provide a value (path) for the varialbe KNOWLEDGE_TRANSFER_FOLDER_IDCNN in the local_settings.py file")
        self.knowledge_transfer_model_folder = conf.KNOWLEDGE_TRANSFER_FOLDER_IDCNN
        self.knowledge_transfer_info_dict = np.load(os.path.join(self.knowledge_transfer_model_folder, 'info.npy'), allow_pickle=True).item()
        if self.is_identity_transfer_possible():
            logger.info("Tracking with identity transfer. The IDENTIFICATION_IMAGE_SIZE will be matched\
                        to the input_image_size of the transferred network")
            self._identity_transfer = True
            self._tracking_with_knowledge_transfer = True
            self._knowledge_transfer_model_folder = conf.KNOWLEDGE_TRANSFER_FOLDER_IDCNN
            conf.IDENTIFICATION_IMAGE_SIZE = self.knowledge_transfer_info_dict['input_image_size']
        else:
            logger.info("Tracking with identity transfer failed. The number of animals in the video\
                        needs to be the same as the number of animals in the transferred network")
            self._identity_transfer = False
            self._tracking_with_knowledge_transfer = False
            conf.KNOWLEDGE_TRANSFER_FOLDER_IDCNN = None

    @property
    def tracking_with_knowledge_transfer(self):
        return self._tracking_with_knowledge_transfer

    @property
    def track_wo_identities(self):
        """
        Flag track without identities

        returns: Boolean
        """
        return self._track_wo_identities

    @property
    def video_path(self):
        return self._video_path

    @video_path.setter
    def video_path(self, value):
        video_name, video_extension = os.path.splitext(value)
        if video_extension in conf.AVAILABLE_VIDEO_EXTENSION:
            self._video_path = value
            #get video folder
            self._video_folder = os.path.dirname(self.video_path)
            #collect some info on the video (resolution, number of frames, ..)
            if not hasattr(self,'number_of_frames'):
                self.get_info()
            self.modified = time.strftime("%c")
        else:
            raise ValueError("Supported video extensions are ", conf.AVAILABLE_VIDEO_EXTENSION)

    @property
    def video_folder(self):
        return self._video_folder

    @property
    def number_of_animals(self):
        return self._number_of_animals

    @property
    def maximum_number_of_blobs(self):
        return self._maximum_number_of_blobs

    @property
    def number_of_frames(self):
        return self._number_of_frames

    @property
    def number_of_episodes(self):
        return self._number_of_episodes

    @property
    def open_multiple_files(self):
        return self._open_multiple_files

    def check_split_video(self):
        """If the video is divided in segments retrieves their paths
        """
        paths_to_video_segments = scanFolder(self.video_path)
        if len(paths_to_video_segments) > 1 and self.open_multiple_files:
            return paths_to_video_segments
        else:
            return None

    @property
    def paths_to_video_segments(self):
        return self._paths_to_video_segments

    @property
    def original_width(self):
        return self._original_width

    @property
    def original_height(self):
        return self._original_height

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def frames_per_second(self):
        return self._frames_per_second

    @property
    def fragment_identifier_to_index(self):
        return self._fragment_identifier_to_index

    @property
    def first_frame_first_global_fragment(self):
        return self._first_frame_first_global_fragment

    @property
    def median_body_length(self):
        return self._median_body_length

    @property
    def median_body_length_full_resolution(self):
        return self.median_body_length/self.resolution_reduction

    @property
    def model_area(self):
        return self._model_area

    @property
    def maximum_number_of_images_in_global_fragments(self):
        return self._maximum_number_of_images_in_global_fragments

    @property
    def number_of_unique_images_in_global_fragments(self):
        return self._number_of_unique_images_in_global_fragments

    @property
    def there_are_crossings(self):
        return self._there_are_crossings

    @property
    def ratio_accumulated_images(self):
        return self._ratio_accumulated_images

    @property
    def segmentation_time(self):
        return self._segmentation_time

    @property
    def crossing_detector_time(self):
        return self._crossing_detector_time

    @property
    def fragmentation_time(self):
        return self._fragmentation_time

    @property
    def protocol1_time(self):
        return self._protocol1_time

    @property
    def protocol2_time(self):
        return self._protocol2_time

    @property
    def protocol3_pretraining_time(self):
        return self._protocol3_pretraining_time

    @property
    def protocol3_accumulation_time(self):
        return self._protocol3_accumulation_time

    @property
    def identify_time(self):
        return self._identify_time

    @property
    def create_trajectories_time(self):
        return self._create_trajectories_time

    def check_paths_consistency_with_video_path(self, new_video_path):
        if self.video_path != new_video_path:
            self.update_paths(new_video_path)

    def update_paths(self, new_video_object_path):
        if new_video_object_path == '': raise ValueError("The path to the video object is an empty string")
        new_session_path = os.path.split(new_video_object_path)[0]
        old_session_path = self.session_folder
        video_name = os.path.split(self._video_path)[1]
        self._video_folder = os.path.split(new_session_path)[0]
        self.video_path = os.path.join(self.video_folder,video_name)
        attributes_to_modify = {key: getattr(self, key) for key in self.__dict__
        if isinstance(getattr(self, key), str)
        and old_session_path in getattr(self, key) }

        for key in attributes_to_modify:
            new_value = attributes_to_modify[key].replace(old_session_path, new_session_path)
            setattr(self, key, new_value)

        if self.paths_to_video_segments is not None and len(self.paths_to_video_segments) != 0:
            logger.info("Updating paths_to_video_segments")
            new_paths_to_video_segments = []
            for path in self.paths_to_video_segments:
                new_path = os.path.join(self.video_folder, os.path.split(path)[1])
                new_paths_to_video_segments.append(new_path)
            self._paths_to_video_segments = new_paths_to_video_segments

        logger.info("Updating checkpoint files")
        folders_to_check = ['crossings_detector_folder', 'pretraining_folder', 'accumulation_folder']
        for folder in folders_to_check:
            if hasattr(self, folder) and getattr(self, folder) is not None:
                if folder == folders_to_check[0]:
                    checkpoint_path = os.path.join(self.crossings_detector_folder, 'checkpoint')
                    if os.path.isfile(checkpoint_path):
                        self.update_tensorflow_checkpoints_file(checkpoint_path, old_session_path, new_session_path)
                    else:
                        logger.warn('No checkpoint found in %s ' %folder)
                else:
                    for sub_folder in ['conv', 'softmax']:
                        checkpoint_path = os.path.join(getattr(self, folder), sub_folder, 'checkpoint')
                        if os.path.isfile(checkpoint_path):
                            self.update_tensorflow_checkpoints_file(checkpoint_path, old_session_path, new_session_path)
                        else:
                            logger.warn('No checkpoint found in %s ' %os.path.join(getattr(self,folder),sub_folder))

        logger.info("Saving video object")
        self.save()
        logger.info("Done")

    @staticmethod
    def update_tensorflow_checkpoints_file(checkpoint_path, current_session_name, new_session_name):
        #checkpoint_file = open(checkpoint_path, "r")
        fh, abs_path = mkstemp()
        with os.fdopen(fh,'w') as new_file:
            with open(checkpoint_path) as old_file:
                for line in old_file:
                    splitted_line = line.split('"')
                    string_to_replace = splitted_line[1]
                    new_string = string_to_replace.replace(current_session_name, new_session_name)
                    splitted_line[1] = new_string
                    new_line = '"'.join(splitted_line)
                    new_file.write(new_line)

        os.remove(checkpoint_path)
        move(abs_path, checkpoint_path)

    def rename_session_folder(self, new_session_name):
        assert new_session_name != ''
        new_session_name = 'session_' + new_session_name
        current_session_name = os.path.split(self.session_folder)[1]
        logger.info("Updating checkpoint files")
        folders_to_check = ['video_folder',
                            'preprocessing_folder',
                            'logs_folder',
                            'previous_session_folder'
                            'crossings_detector_folder',
                            'pretraining_folder',
                            'accumulation_folder']

        for folder in folders_to_check:
            if hasattr(self, folder) and getattr(self, folder) is not None:
                if folder == folders_to_check[0]:
                    checkpoint_path = os.path.join(self.crossings_detector_folder, 'checkpoint')
                    if os.path.isfile(checkpoint_path):
                        self.update_tensorflow_checkpoints_file(checkpoint_path, current_session_name, new_session_name)
                    else:
                        logger.warn('No checkpoint found in %s ' %folder)
                else:
                    for sub_folder in ['conv', 'softmax']:
                        checkpoint_path = os.path.join(getattr(self,folder), sub_folder, 'checkpoint')
                        if os.path.isfile(checkpoint_path):
                            self.update_tensorflow_checkpoints_file(checkpoint_path, current_session_name, new_session_name)
                        else:
                            logger.warn('No checkpoint found in %s ' %os.path.join(getattr(self, folder), sub_folder))

        attributes_to_modify = {key: getattr(self, key) for key in self.__dict__
        if isinstance(getattr(self, key), str)
        and current_session_name in getattr(self, key) }
        logger.info("Modifying folder name from %s to %s "  %(current_session_name, new_session_name))
        os.rename(self.session_folder,
                os.path.join(self.video_folder, new_session_name))
        logger.info("Updating video object")

        for key in attributes_to_modify:
            new_value = attributes_to_modify[key].replace(current_session_name, new_session_name)
            setattr(self, key, new_value)

        logger.info("Saving video object")
        self.save()

    def get_info(self):
        """Retrieves basic information concerning the video. If the video is
        recorded as a single file, it generates "episodes" for parallelisation
        during segmentation
        """
        self._paths_to_video_segments = self.check_split_video()
        cap = cv2.VideoCapture(self.video_path)
        self._width  = self._original_width = int(cap.get(3))
        self._height = self._original_height = int(cap.get(4))

        try:
            self._frames_per_second = int(cap.get(5))
        except:
            self._frames_per_second = None
            logger.info("Cannot read frame per second")
        if self._paths_to_video_segments is None:
            self._number_of_frames = int(cap.get(7))
            self.get_episodes()
        else:
            chunks_lengths = [int(cv2.VideoCapture(chunk).get(7)) for chunk in self._paths_to_video_segments]
            self._episodes_start_end = [(np.sum(chunks_lengths[:i-1], dtype = np.int), np.sum(chunks_lengths[:i])-1) for i in range(1,len(chunks_lengths)+1)]
            self._number_of_frames = np.sum(chunks_lengths)
            self._number_of_episodes = len(self._paths_to_video_segments)
        cap.release()

    @property
    def identification_image_size(self):
        return self._identification_image_size

    def compute_identification_image_size(self, maximum_body_length):
        """Uses an estimate of the body length of the animals in order to
        compute the size of the square image that is generated from every
        blob to identify the animals
        """
        if conf.IDENTIFICATION_IMAGE_SIZE is None:
            identification_image_size = int(maximum_body_length / np.sqrt(2))
            identification_image_size = identification_image_size + identification_image_size % 2
            self._identification_image_size = (identification_image_size, identification_image_size, self.number_of_channels)
        else:
            self._identification_image_size = conf.IDENTIFICATION_IMAGE_SIZE

    def init_processes_time_attributes(self):
        self.generate_trajectories_time = 0
        self.solve_impossible_jumps_time = 0
        self.solve_duplications_time = 0
        self.assignment_time = 0
        self.second_accumulation_time = 0
        self.pretraining_time = 0
        self.assignment_time = 0
        self.first_accumulation_time = 0
        self.preprocessing_time = 0

    def create_session_folder(self, name = ''):
        """Creates a folder named training in video_folder and a folder session_num
        where num is the session number and it is created everytime one starts
        training a network for a certain video_path
        """
        if name == '':
            self._session_folder = os.path.join(self.video_folder, 'session')
        else:
            self._session_folder = os.path.join(self.video_folder, 'session_' + name)

        if not os.path.isdir(self._session_folder):
            os.makedirs(self._session_folder)
            self._previous_session_folder = ''
        else:
            self._previous_session_folder = self.session_folder

        self._path_to_video_object = os.path.join(self.session_folder, 'video_object.npy')
        logger.info("the folder %s has been created" %self.session_folder)

    def create_images_folders(self):
        """Create a folder named images inside of the session_folder
        """
        ## for RAM optimization
        self._segmentation_data_folder = os.path.join(self.session_folder, 'segmentation_data')
        self._identification_images_folder = os.path.join(self.session_folder, 'identification_images')
        self.identification_images_file_path = os.path.join(self._identification_images_folder, 'i_images.hdf5')
        if not os.path.isdir(self._segmentation_data_folder):
            os.makedirs(self._segmentation_data_folder)
            logger.info("the folder %s has been created" %self._segmentation_data_folder)

        if not os.path.isdir(self._identification_images_folder):
            os.makedirs(self._identification_images_folder)
            logger.info("the folder %s has been created" %self._identification_images_folder)


    def create_preprocessing_folder(self):
        """If it does not exist creates a folder called preprocessing
        in the video folder"""
        self._preprocessing_folder = os.path.join(self.session_folder, 'preprocessing')
        if not os.path.isdir(self.preprocessing_folder):
            os.makedirs(self.preprocessing_folder)
            logger.info("the folder %s has been created" %self._preprocessing_folder)


    def create_crossings_detector_folder(self):
        """If it does not exist creates a folder called crossing_detector
        in the video folder"""
        logger.info('setting path to save crossing detector model')
        self._crossings_detector_folder = os.path.join(self.session_folder, 'crossings_detector')
        if not os.path.isdir(self.crossings_detector_folder):
            logger.info("the folder %s has been created" %self.crossings_detector_folder)
            os.makedirs(self.crossings_detector_folder)

    def create_pretraining_folder(self, delete = False):
        """Creates a folder named pretraining in video_folder where the model
        trained during the pretraining is stored
        """
        self._pretraining_folder = os.path.join(self.session_folder, 'pretraining')
        if not os.path.isdir(self.pretraining_folder):
            os.makedirs(self.pretraining_folder)
        elif delete:
            rmtree(self.pretraining_folder)
            os.makedirs(self.pretraining_folder)


    def create_accumulation_folder(self, iteration_number = 0, delete = False):
        """Folder in which the model generated while accumulating is stored
        (after pretraining)
        """
        accumulation_folder_name = 'accumulation_' + str(iteration_number)
        self._accumulation_folder = os.path.join(self.session_folder, accumulation_folder_name)
        if not os.path.isdir(self.accumulation_folder):
            os.makedirs(self.accumulation_folder)
        elif delete:
            rmtree(self.accumulation_folder)
            os.makedirs(self.accumulation_folder)

    def create_individual_videos_folder(self):
        """Create folder where to save the individual videos
        """
        self.individual_videos_folder = os.path.join(self.session_folder, 'individual_videos')
        if not os.path.exists(self.individual_videos_folder):
            os.makedirs(self.individual_videos_folder)

    def init_accumulation_statistics_attributes(self, attributes = None):
        if attributes is None:
            attributes = ['number_of_accumulated_global_fragments',
                        'number_of_non_certain_global_fragments',
                        'number_of_randomly_assigned_global_fragments',
                        'number_of_nonconsistent_global_fragments',
                        'number_of_nonunique_global_fragments',
                        'number_of_acceptable_global_fragments',
                        'validation_accuracy','validation_individual_accuracies',
                        'training_accuracy','training_individual_accuracies',
                        'ratio_of_accumulated_images']
        self.accumulation_statistics_attributes_list = attributes
        [setattr(self, attribute, []) for attribute in self.accumulation_statistics_attributes_list]

    def store_accumulation_step_statistics_data(self, new_values):
        [getattr(self, attr).append(value) for attr, value in zip(self.accumulation_statistics_attributes_list, new_values)]

    def store_accumulation_statistics_data(self, accumulation_trial, number_of_possible_accumulation = conf.MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS + 1):
        if not hasattr(self, 'accumulation_statistics'): self.accumulation_statistics = [None] * number_of_possible_accumulation
        self.accumulation_statistics[accumulation_trial] = [getattr(self, stat_attr)
                                                            for stat_attr in self.accumulation_statistics_attributes_list]

    def create_trajectories_folder(self):
        """Folder in which trajectories files are stored
        """
        self.trajectories_folder = os.path.join(self.session_folder,'trajectories')
        if not os.path.isdir(self.trajectories_folder):
            logger.info("Creating trajectories folder...")
            os.makedirs(self.trajectories_folder)
            logger.info("the folder %s has been created" %self.trajectories_folder)

    def create_trajectories_wo_identities_folder(self):
        """Folder in which trajectories without identites are stored
        """
        self.trajectories_wo_identities_folder = os.path.join(self.session_folder,'trajectories_wo_identities')
        if not os.path.isdir(self.trajectories_wo_identities_folder):
            logger.info("Creating trajectories folder...")
            os.makedirs(self.trajectories_wo_identities_folder)
            logger.info("the folder %s has been created" %self.trajectories_wo_identities_folder)

    def create_trajectories_wo_gaps_folder(self):
        """Folder in which trajectories files are stored
        """
        self.trajectories_wo_gaps_folder = os.path.join(self.session_folder, 'trajectories_wo_gaps')
        if not os.path.isdir(self.trajectories_wo_gaps_folder):
            logger.info("Creating trajectories folder...")
            os.makedirs(self.trajectories_wo_gaps_folder)
            logger.info("the folder %s has been created" %self.trajectories_wo_gaps_folder)

    def create_embeddings_folder(self):
        """If it does not exist creates a folder called embedding
        in the video folder"""
        self._embeddings_folder = os.path.join(self.session_folder, 'embeddings')
        if not os.path.isdir(self.embeddings_folder):
            os.makedirs(self.embeddings_folder)
            logger.info("the folder %s has been created" %self.embeddings_folder)

    def get_episodes(self):
        """Split video in episodes (chunks) of FRAMES_PER_EPISODE frames
        for parallelisation"""
        starting_frames = np.arange(0, self.number_of_frames, conf.FRAMES_PER_EPISODE)
        ending_frames = np.hstack((starting_frames[1:]-1, self.number_of_frames))
        self._episodes_start_end = list(zip(starting_frames, ending_frames))
        self._number_of_episodes = len(starting_frames)

    def in_which_episode(self, frame_number):
        """Check to which episode a frame index belongs
        """
        episode_number = [i for i, episode_start_end in enumerate(self._episodes_start_end)
                            if episode_start_end[0] <= frame_number
                            and episode_start_end[1] >= frame_number]
        if episode_number:
            return episode_number[0]
        else:
            return None

    def copy_attributes_between_two_video_objects(self, video_object_source, list_of_attributes, is_property = None):
        for i, attribute in enumerate(list_of_attributes):
            if not hasattr(video_object_source, attribute):
                logger.warning("attribute %s does not exist" %attribute)
            else:
                if is_property is not None:
                    attribute_is_property = is_property[i]
                    if attribute_is_property:
                        setattr(self, '_' + attribute, getattr(video_object_source, attribute))
                    else:
                        setattr(self, attribute, getattr(video_object_source, attribute))
                else:
                    setattr(self, '_' + attribute, getattr(video_object_source, attribute))

    def compute_overall_P2(self, fragments):
        weighted_P2 = 0
        number_of_individual_blobs = 0

        for fragment in fragments:
            if fragment.is_an_individual:
                if fragment.assigned_identities[0] != 0:
                    weighted_P2 += fragment.P2_vector[fragment.assigned_identities[0] - 1] * fragment.number_of_images
                number_of_individual_blobs += fragment.number_of_images

        self.overall_P2 = weighted_P2 / number_of_individual_blobs

    def delete_accumulation_folders(self):
        for path in glob.glob(os.path.join(self.session_folder, '*')):
            if 'accumulation_' in path or 'pretraining' in path:
                rmtree(path, ignore_errors=True)

    def delete_data(self):
        logger.info("Data policy: {}".format(conf.DATA_POLICY))
        if conf.DATA_POLICY in ['trajectories', 'validation',
                           'knowledge_transfer', 'idmatcher.ai']:

            if os.path.isdir(self._segmentation_data_folder):
                logger.info("Deleting segmentation images")
                rmtree(self._segmentation_data_folder, ignore_errors=True)
            if os.path.isfile(self.global_fragments_path):
                logger.info("Deleting global fragments")
                os.remove(self.global_fragments_path)
            if os.path.isfile(self.blobs_path_segmented):
                logger.info("Deleting blobs segmented")
                os.remove(self.blobs_path_segmented)
            if hasattr(self, '_crossings_detector_folder') and os.path.isdir(self.crossings_detector_folder):
                logger.info("Deleting crossing detector folder")
                rmtree(self.crossings_detector_folder, ignore_errors=True)


        if conf.DATA_POLICY in ['trajectories', 'validation',
                           'knowledge_transfer']:
            if os.path.isdir(self._identification_images_folder):
                logger.info("Deleting identification images")
                rmtree(self._identification_images_folder, ignore_errors=True)

        if conf.DATA_POLICY in ['trajectories', 'validation']:
            logger.info("Deleting CNN models folders")
            self.delete_accumulation_folders()

        if conf.DATA_POLICY == 'trajectories':
            if os.path.isdir(self.preprocessing_folder):
                logger.info("Deleting preprocessing data")
                rmtree(self.preprocessing_folder, ignore_errors=True)


    def interpolate(self, list_of_blobs=None, list_of_fragments=None, identity=None, start=None, end=None):
        """
        Does the blobs positions interpolation

        :param ListOfBlobs list_of_blobs:
        :param identity:
        :param int start:
        :param int end:
        :param ListOfFragments list_of_fragments:
        """

        if self.is_centroid_updated:
            assert identity is not None and start is not None and end is not None and list_of_blobs is not None

            list_of_blobs.interpolate_from_user_generated_centroids(self, identity, start, end)

        else:
            assert list_of_fragments is not None and list_of_blobs is not None

            close_trajectories_gaps(self, list_of_blobs, list_of_fragments)


    def get_first_frame(self, list_of_blobs):
        if self.number_of_animals != 1:
            return self.first_frame_first_global_fragment[self.accumulation_trial]
        elif self.number_of_animals == 1:
            return 0
        else:
            for blobs_in_frame in list_of_blobs.blobs_in_video:
                if len(blobs_in_frame) != 0:
                    return blobs_in_frame[0].frame_number


def scanFolder(path):
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    folder = os.path.dirname(path)
    paths = glob.glob(folder + "/*" + extension)
    return natsorted(paths)
