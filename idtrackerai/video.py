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
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H.,
# de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of
# unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P:
# gonzalo.polavieja@neuro.fchampalimaud.org)


from typing import Dict
import glob
import logging
import os
from shutil import move, rmtree
from tempfile import mkstemp

import cv2
import numpy as np
from confapp import conf
from natsort import natsorted


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

    def __init__(self, video_path=None, open_multiple_files=False):
        logger.debug("Video object init")
        # General video properties
        self._video_paths = None
        self._original_width = None
        self._original_height = None
        self._width = None
        self._height = None
        self._frames_per_second = None
        self._number_of_animals = None
        self._number_of_frames = None
        self._number_of_episodes = None
        self._episodes_start_end = None
        self._user_defined_parameters = None  # has a setter

        # Get some other video properties
        self._open_multiple_files = open_multiple_files
        self.video_path = video_path  # has a setter
        self._video_paths = self.get_video_paths(
            self.video_path, self._open_multiple_files
        )
        self._get_info_from_video_file()
        (
            self._number_of_frames,
            self._episodes_start_end,
            self._number_of_episodes,
        ) = self.get_num_frames_and_processing_episodes(self._video_paths)

        self._number_of_channels = 1  # Used to create identification images
        self._maximum_number_of_blobs = (
            0  # int: the maximum number of blobs detected in the video
        )
        self._frames_with_more_blobs_than_animals = None
        self._median_body_length = None
        self._model_area = None
        self._fragment_identifier_to_index = None
        self._identity_transfer = None
        self._tracking_with_knowledge_transfer = False
        self._percentage_of_accumulated_images = None
        self._first_frame_first_global_fragment = []
        self._identities_groups = {}
        self._there_are_crossings = True
        self._accumulation_trial = 0
        self._knowledge_transfer_info_dict = None
        self._setup_points = []
        if conf.SIGMA_GAUSSIAN_BLURRING is not None:
            self.sigma_gaussian_blurring = conf.SIGMA_GAUSSIAN_BLURRING

        # Paths and folders
        self._blobs_path = (
            None  # string: path to the saved list of blob objects
        )
        self._blobs_path_segmented = None
        self._blobs_path_interpolated = None
        self._accumulation_folder = None
        self._preprocessing_folder = None
        self._images_folder = None
        self._global_fragments_path = (
            None  # string: path to saved list of global fragments
        )
        self._pretraining_folder = None
        self.individual_videos_folder = None

        # Flag to decide which type of interpolation is done. This flag
        # is updated when we update a blob centroid
        self._is_centroid_updated = False
        self._estimated_accuracy = None

        # Processes states
        self._has_preprocessing_parameters = False
        self._has_animals_detected = False  # animal detection and segmentation
        self._has_crossings_detected = False  # crossings detection
        self._has_been_fragmented = False  # fragmentation
        self._has_protocol1_finished = False  # protocols cascade
        self._has_protocol2_finished = False  # protocols cascade
        self._has_protocol3_pretraining_finished = False  # protocols cascade
        self._has_protocol3_accumulation_finished = False  # protocols cascade
        self._has_protocol3_finished = False  # protocols cascade
        self._has_residual_identification = False  # residual identification
        self._has_impossible_jumps_solved = False  # post-processing
        self._has_crossings_solved = False  # crossings interpolation
        self._has_trajectories = False  # trajectories generation
        self._has_trajectories_wo_gaps = False  # trajectories generation

        # Timers
        self._detect_animals_time = 0.0
        self._crossing_detector_time = 0.0
        self._fragmentation_time = 0.0
        self._protocol1_time = 0.0
        self._protocol2_time = 0.0
        self._protocol3_pretraining_time = 0.0
        self._protocol3_accumulation_time = 0.0
        self._identify_time = 0.0
        self._create_trajectories_time = 0.0

        logger.debug(f"Video(open_multiple_files={self.open_multiple_files})")

    # General video properties
    @property
    def number_of_channels(self):
        return self._number_of_channels

    @property
    def episodes_start_end(self):
        return self._episodes_start_end

    @property
    def video_path(self):
        return self._video_path

    @video_path.setter
    def video_path(self, video_path):
        assert os.path.exists(video_path)
        video_name, video_extension = os.path.splitext(video_path)
        if video_extension in conf.AVAILABLE_VIDEO_EXTENSION:
            self._video_path = video_path
            # get video folder
            self._video_folder = os.path.dirname(self.video_path)
        else:
            raise ValueError(
                "Supported video extensions are ",
                conf.AVAILABLE_VIDEO_EXTENSION,
            )

    @property
    def video_folder(self):
        return self._video_folder

    @property
    def number_of_frames(self):
        return self._number_of_frames

    @property
    def number_of_episodes(self):
        return self._number_of_episodes

    @property
    def open_multiple_files(self):
        return self._open_multiple_files

    @property
    def video_paths(self):
        return self._video_paths

    @property
    def original_width(self):
        return self._original_width

    @property
    def original_height(self):
        return self._original_height

    @property
    def width(self):
        return np.round(
            self.original_width
            * self.user_defined_parameters["resolution_reduction"]
        ).astype(int)

    @property
    def height(self):
        return np.round(
            self.original_height
            * self.user_defined_parameters["resolution_reduction"]
        ).astype(int)

    @property
    def frames_per_second(self):
        return self._frames_per_second

    @property
    def fragment_identifier_to_index(self):
        return self._fragment_identifier_to_index

    # Processing parameters properties

    @property
    def user_defined_parameters(self):
        if self._user_defined_parameters is not None:
            return self._user_defined_parameters
        else:
            raise Exception("No _user_defined_parameters given")

    @user_defined_parameters.setter
    def user_defined_parameters(self, value: Dict):
        self._user_defined_parameters = value

    # During processing properties

    @property
    def maximum_number_of_blobs(self):
        return self._maximum_number_of_blobs

    @property
    def percentage_of_accumulated_images(self):
        return self._percentage_of_accumulated_images

    @property
    def erosion_kernel_size(self):
        return self._erosion_kernel_size

    @property
    def frames_with_more_blobs_than_animals(self):
        return self._frames_with_more_blobs_than_animals

    @property
    def accumulation_trial(self):
        return self._accumulation_trial

    @property
    def estimated_accuracy(self):
        return self._estimated_accuracy

    @property
    def identification_image_size(self):
        return self._identification_image_size

    @property
    def knowledge_transfer_info_dict(self):
        return self._knowledge_transfer_info_dict

    @property
    def first_frame_first_global_fragment(self):
        return self._first_frame_first_global_fragment

    @property
    def median_body_length(self):
        return self._median_body_length

    @property
    def median_body_length_full_resolution(self):
        return (
            self.median_body_length
            / self.user_defined_parameters["resolution_reduction"]
        )

    @property
    def model_area(self):
        return self._model_area

    # @property
    # def maximum_number_of_images_in_global_fragments(self):
    #     return self._maximum_number_of_images_in_global_fragments

    # @property
    # def number_of_unique_images_in_global_fragments(self):
    #     return self._number_of_unique_images_in_global_fragments

    @property
    def there_are_crossings(self):
        return self._there_are_crossings

    @property
    def ratio_accumulated_images(self):
        return self._ratio_accumulated_images

    # Processing steps
    @property
    def has_animals_detected(self):
        return self._has_animals_detected

    @property
    def has_crossings_detected(self):
        return self._has_crossings_detected

    @property
    def has_been_fragmented(self):
        return self._has_been_fragmented

    @property
    def has_protocol1_finished(self):
        return self._has_protocol1_finished

    @property
    def has_protocol2_finished(self):
        return self._has_protocol2_finished

    @property
    def has_protocol3_pretraining_finished(self):
        return self._has_protocol3_pretraining_finished

    @property
    def has_protocol3_accumulation_finished(self):
        return self._has_protocol3_accumulation_finished

    @property
    def has_protocol3_finished(self):
        return self._has_protocol3_finished

    @property
    def has_residual_identification(self):
        return self._has_residual_identification

    @property
    def has_impossible_jumps_solved(self):
        return self._has_impossible_jumps_solved

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
    def has_preprocessing_parameters(self):
        return self._has_preprocessing_parameters

    # Comptational times
    @property
    def detect_animals_time(self):
        return self._detect_animals_time

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

    # Paths and folders
    @property
    def preprocessing_folder(self):
        return self._preprocessing_folder

    @property
    def images_folder(self):
        return self._images_folder

    @property
    def crossings_detector_folder(self):
        return self._crossings_detector_folder

    @property
    def previous_session_folder(self):
        return self._previous_session_folder

    @property
    def pretraining_folder(self):
        return self._pretraining_folder

    @property
    def accumulation_folder(self):
        return self._accumulation_folder

    @property
    def session_folder(self):
        return self._session_folder

    @property
    def blobs_path(self):
        """get the path to save the blob collection after segmentation.
        It checks that the segmentation has been succesfully performed"""
        if self.preprocessing_folder is None:
            return None
        self._blobs_path = os.path.join(
            self.preprocessing_folder, "blobs_collection.npy"
        )
        return self._blobs_path

    @property
    def blobs_path_segmented(self):
        """get the path to save the blob collection after segmentation.
        It checks that the segmentation has been succesfully performed"""
        self._blobs_path_segmented = os.path.join(
            self.preprocessing_folder, "blobs_collection_segmented.npy"
        )
        return self._blobs_path_segmented

    @property
    def blobs_path_interpolated(self):
        self._blobs_path_interpolated = os.path.join(
            self.preprocessing_folder, "blobs_collection_interpolated.npy"
        )
        return self._blobs_path_interpolated

    @property
    def global_fragments_path(self):
        """get the path to save the list of global fragments after
        fragmentation"""
        self._global_fragments_path = os.path.join(
            self.preprocessing_folder, "global_fragments.npy"
        )
        return self._global_fragments_path

    @property
    def fragments_path(self):
        """get the path to save the list of global fragments after
        fragmentation"""
        self._fragments_path = os.path.join(
            self.preprocessing_folder, "fragments.npy"
        )
        return self._fragments_path

    @property
    def path_to_video_object(self):
        return self._path_to_video_object

    @property
    def ground_truth_path(self):
        return os.path.join(self.video_folder, "_groundtruth.npy")

    @property
    def segmentation_data_foler(self):
        return self._segmentation_data_folder

    # Validation properties
    @property
    def identities_groups(self):
        return self._identities_groups

    @property
    def is_centroid_updated(self):
        return self._is_centroid_updated

    @is_centroid_updated.setter
    def is_centroid_updated(self, value):
        self._is_centroid_updated = value

    # Methods

    def save(self):
        """save class"""
        logger.info("saving video object in %s" % self.path_to_video_object)
        logger.debug(
            f"Video.save(open_multiple_files={self.open_multiple_files})"
        )
        np.save(self.path_to_video_object, self)

    @staticmethod
    def load(video_object_path):
        video_object = np.load(video_object_path, allow_pickle=True).item()
        video_object.update_paths(video_object_path)
        return video_object

    @staticmethod
    def get_video_paths(video_path, open_multiple_files):
        """If the video is divided in segments retrieves their paths"""
        video_paths = scanFolder(video_path)
        if len(video_paths) > 1 and open_multiple_files:
            return video_paths
        else:
            return [video_path]

    def check_paths_consistency_with_video_path(self, new_video_path):
        if self.video_path != new_video_path:
            self.update_paths(new_video_path)

    def update_paths(self, new_video_object_path):
        if new_video_object_path == "":
            raise ValueError("The path to the video object is an empty string")
        new_session_path = os.path.split(new_video_object_path)[0]
        old_session_path = self.session_folder
        video_name = os.path.split(self._video_path)[1]
        self._video_folder = os.path.split(new_session_path)[0]
        video_path = os.path.join(self.video_folder, video_name)
        if os.path.isfile(video_path):
            self.video_path = video_path
        else:
            logger.warning(
                f"video_path: {video_path} does not exists. "
                f"The original video_path {self.video_path}. "
                f"We will keep the original video_path"
            )

        attributes_to_modify = {
            key: getattr(self, key)
            for key in self.__dict__
            if isinstance(getattr(self, key), str)
            and old_session_path in getattr(self, key)
        }

        for key in attributes_to_modify:
            new_value = attributes_to_modify[key].replace(
                old_session_path, new_session_path
            )
            setattr(self, key, new_value)

        if self.video_paths is not None and len(self.video_paths) != 0:
            logger.info("Updating video_paths")
            new_paths_to_video_segments = []
            for path in self.video_paths:
                new_path = os.path.join(
                    self.video_folder, os.path.split(path)[1]
                )
                new_paths_to_video_segments.append(new_path)
            self._video_paths = new_paths_to_video_segments

        logger.info("Saving video object")
        self.save()
        logger.info("Done")

    def rename_session_folder(self, new_session_name):
        assert new_session_name != ""
        new_session_name = "session_" + new_session_name
        current_session_name = os.path.split(self.session_folder)[1]
        logger.info("Updating checkpoint files")
        folders_to_check = [
            "video_folder",
            "preprocessing_folder",
            "logs_folder",
            "previous_session_folder" "crossings_detector_folder",
            "pretraining_folder",
            "accumulation_folder",
        ]

        for folder in folders_to_check:
            if hasattr(self, folder) and getattr(self, folder) is not None:
                if folder == folders_to_check[0]:
                    checkpoint_path = os.path.join(
                        self.crossings_detector_folder, "checkpoint"
                    )
                    if os.path.isfile(checkpoint_path):
                        self.update_tensorflow_checkpoints_file(
                            checkpoint_path,
                            current_session_name,
                            new_session_name,
                        )
                    else:
                        logger.warn("No checkpoint found in %s " % folder)
                else:
                    for sub_folder in ["conv", "softmax"]:
                        checkpoint_path = os.path.join(
                            getattr(self, folder), sub_folder, "checkpoint"
                        )
                        if os.path.isfile(checkpoint_path):
                            self.update_tensorflow_checkpoints_file(
                                checkpoint_path,
                                current_session_name,
                                new_session_name,
                            )
                        else:
                            logger.warn(
                                "No checkpoint found in %s "
                                % os.path.join(
                                    getattr(self, folder), sub_folder
                                )
                            )

        attributes_to_modify = {
            key: getattr(self, key)
            for key in self.__dict__
            if isinstance(getattr(self, key), str)
            and current_session_name in getattr(self, key)
        }
        logger.info(
            "Modifying folder name from %s to %s "
            % (current_session_name, new_session_name)
        )
        os.rename(
            self.session_folder,
            os.path.join(self.video_folder, new_session_name),
        )
        logger.info("Updating video object")

        for key in attributes_to_modify:
            new_value = attributes_to_modify[key].replace(
                current_session_name, new_session_name
            )
            setattr(self, key, new_value)

        logger.info("Saving video object")
        self.save()

    def _get_info_from_video_file(self):

        widths, heights, frames_per_seconds = [], [], []
        for path in self._video_paths:
            cap = cv2.VideoCapture(path)
            widths.append(int(cap.get(3)))
            heights.append(int(cap.get(4)))

            try:
                frames_per_seconds.append(int(cap.get(5)))
            except cv2.error:
                logger.warning(f"Cannot read frame per second for {path}")
                frames_per_seconds.append(None)
            cap.release()

        assert len(set(widths)) == 1
        assert len(set(heights)) == 1
        assert len(set(frames_per_seconds)) == 1

        self._width = self._original_width = widths[0]
        self._height = self._original_height = heights[0]
        self._frames_per_second = frames_per_seconds[0]

    def compute_identification_image_size(self, maximum_body_length):
        """Uses an estimate of the body length of the animals in order to
        compute the size of the square image that is generated from every
        blob to identify the animals
        """
        if self.user_defined_parameters["identification_image_size"] is None:
            identification_image_size = int(maximum_body_length / np.sqrt(2))
            identification_image_size = (
                identification_image_size + identification_image_size % 2
            )
            self._identification_image_size = (
                identification_image_size,
                identification_image_size,
                self.number_of_channels,
            )
        else:
            self._identification_image_size = self.user_defined_parameters[
                "identification_image_size"
            ]

    def create_session_folder(self, name=""):
        """Creates a folder named training in video_folder and a folder session_num
        where num is the session number and it is created everytime one starts
        training a network for a certain video_path
        """
        if name == "":
            session_name = "session"
        else:
            session_name = "session_" + name

        self._session_folder = os.path.join(self.video_folder, session_name)
        logger.info(f"Creating session folder at {self._session_folder}")
        if not os.path.isdir(self._session_folder):
            os.makedirs(self._session_folder)
            self._previous_session_folder = ""
        else:
            self._previous_session_folder = self.session_folder

        self._path_to_video_object = os.path.join(
            self.session_folder, "video_object.npy"
        )
        logger.info("the folder %s has been created" % self.session_folder)

    def create_images_folders(self):
        """Create a folder named images inside of the session_folder"""
        ## for RAM optimization
        self._segmentation_data_folder = os.path.join(
            self.session_folder, "segmentation_data"
        )
        self._identification_images_folder = os.path.join(
            self.session_folder, "identification_images"
        )
        self.identification_images_file_paths = [
            os.path.join(
                self._identification_images_folder,
                "id_images_{}.hdf5".format(e),
            )
            for e in range(self.number_of_episodes)
        ]
        if not os.path.isdir(self._segmentation_data_folder):
            os.makedirs(self._segmentation_data_folder)
            logger.info(
                "the folder %s has been created"
                % self._segmentation_data_folder
            )

        if not os.path.isdir(self._identification_images_folder):
            os.makedirs(self._identification_images_folder)
            logger.info(
                "the folder %s has been created"
                % self._identification_images_folder
            )

    def create_preprocessing_folder(self):
        """If it does not exist creates a folder called preprocessing
        in the video folder"""
        self._preprocessing_folder = os.path.join(
            self.session_folder, "preprocessing"
        )
        if not os.path.isdir(self.preprocessing_folder):
            os.makedirs(self.preprocessing_folder)
            logger.info(
                "the folder %s has been created" % self._preprocessing_folder
            )

    def create_crossings_detector_folder(self):
        """If it does not exist creates a folder called crossing_detector
        in the video folder"""
        logger.info("setting path to save crossing detector model")
        self._crossings_detector_folder = os.path.join(
            self.session_folder, "crossings_detector"
        )
        if not os.path.isdir(self.crossings_detector_folder):
            logger.info(
                "the folder %s has been created"
                % self.crossings_detector_folder
            )
            os.makedirs(self.crossings_detector_folder)

    def create_pretraining_folder(self, delete=False):
        """Creates a folder named pretraining in video_folder where the model
        trained during the pretraining is stored
        """
        self._pretraining_folder = os.path.join(
            self.session_folder, "pretraining"
        )
        if not os.path.isdir(self.pretraining_folder):
            os.makedirs(self.pretraining_folder)
        elif delete:
            rmtree(self.pretraining_folder)
            os.makedirs(self.pretraining_folder)

    def create_accumulation_folder(self, iteration_number=0, delete=False):
        """Folder in which the model generated while accumulating is stored
        (after pretraining)
        """
        accumulation_folder_name = "accumulation_" + str(iteration_number)
        self._accumulation_folder = os.path.join(
            self.session_folder, accumulation_folder_name
        )
        if not os.path.isdir(self.accumulation_folder):
            os.makedirs(self.accumulation_folder)
        elif delete:
            rmtree(self.accumulation_folder)
            os.makedirs(self.accumulation_folder)

    def create_individual_videos_folder(self):
        """Create folder where to save the individual videos"""
        self.individual_videos_folder = os.path.join(
            self.session_folder, "individual_videos"
        )
        if not os.path.exists(self.individual_videos_folder):
            os.makedirs(self.individual_videos_folder)

    def init_accumulation_statistics_attributes(self, attributes=None):
        if attributes is None:
            attributes = [
                "number_of_accumulated_global_fragments",
                "number_of_non_certain_global_fragments",
                "number_of_randomly_assigned_global_fragments",
                "number_of_nonconsistent_global_fragments",
                "number_of_nonunique_global_fragments",
                "number_of_acceptable_global_fragments",
                "ratio_of_accumulated_images",
            ]
        self.accumulation_statistics_attributes_list = attributes
        [
            setattr(self, attribute, [])
            for attribute in self.accumulation_statistics_attributes_list
        ]

    def store_accumulation_step_statistics_data(self, new_values):
        [
            getattr(self, attr).append(value)
            for attr, value in zip(
                self.accumulation_statistics_attributes_list, new_values
            )
        ]

    def store_accumulation_statistics_data(
        self,
        accumulation_trial,
        number_of_possible_accumulation=conf.MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS
        + 1,
    ):
        if not hasattr(self, "accumulation_statistics"):
            self.accumulation_statistics = [
                None
            ] * number_of_possible_accumulation
        self.accumulation_statistics[accumulation_trial] = [
            getattr(self, stat_attr)
            for stat_attr in self.accumulation_statistics_attributes_list
        ]

    def create_trajectories_folder(self):
        """Folder in which trajectories files are stored"""
        self.trajectories_folder = os.path.join(
            self.session_folder, "trajectories"
        )
        if not os.path.isdir(self.trajectories_folder):
            logger.info("Creating trajectories folder...")
            os.makedirs(self.trajectories_folder)
            logger.info(
                "the folder %s has been created" % self.trajectories_folder
            )

    def create_trajectories_wo_identification_folder(self):
        """Folder in which trajectories without identites are stored"""
        self.trajectories_wo_identification_folder = os.path.join(
            self.session_folder, "trajectories_wo_identification"
        )
        if not os.path.isdir(self.trajectories_wo_identification_folder):
            logger.info("Creating trajectories folder...")
            os.makedirs(self.trajectories_wo_identification_folder)
            logger.info(
                "the folder %s has been created"
                % self.trajectories_wo_identification_folder
            )

    def create_trajectories_wo_gaps_folder(self):
        """Folder in which trajectories files are stored"""
        self.trajectories_wo_gaps_folder = os.path.join(
            self.session_folder, "trajectories_wo_gaps"
        )
        if not os.path.isdir(self.trajectories_wo_gaps_folder):
            logger.info("Creating trajectories folder...")
            os.makedirs(self.trajectories_wo_gaps_folder)
            logger.info(
                "the folder %s has been created"
                % self.trajectories_wo_gaps_folder
            )

    @staticmethod
    def get_num_frames_and_processing_episodes(video_paths):
        logger.info("Getting video episodes and number of frames")
        if len(video_paths) == 1:  # single video file
            cap = cv2.VideoCapture(video_paths[0])
            number_of_frames = int(cap.get(7))
            start = list(range(0, number_of_frames, conf.FRAMES_PER_EPISODE))
            end = start[1:] + [number_of_frames]
            episodes_start_end = list(zip(start, end))
            number_of_episodes = len(episodes_start_end)
        else:  # multiple video files
            logger.info("Tracking multiple files:")
            logger.info(f"{video_paths}")
            num_frames_in_video_segments = [
                int(cv2.VideoCapture(video_segment).get(7))
                for video_segment in video_paths
            ]
            end = list(np.cumsum(num_frames_in_video_segments))
            start = [0] + end[:-1]
            episodes_start_end = list(zip(start, end))
            number_of_frames = np.sum(num_frames_in_video_segments)
            number_of_episodes = len(episodes_start_end)
        logger.info(f"The video has {number_of_frames} frames")
        logger.info(f"The video has {number_of_episodes} episodes")
        return number_of_frames, episodes_start_end, number_of_episodes

    def in_which_episode(self, frame_number):
        """Check to which episode a frame index belongs"""
        episode_number = [
            i
            for i, episode_start_end in enumerate(self._episodes_start_end)
            if episode_start_end[0] <= frame_number
            and episode_start_end[1] >= frame_number
        ]
        if episode_number:
            return episode_number[0]
        else:
            return None

    def copy_attributes_between_two_video_objects(
        self, video_object_source, list_of_attributes, is_property=None
    ):
        for i, attribute in enumerate(list_of_attributes):
            if not hasattr(video_object_source, attribute):
                logger.warning("attribute %s does not exist" % attribute)
            else:
                if is_property is not None:
                    attribute_is_property = is_property[i]
                    if attribute_is_property:
                        setattr(
                            self,
                            "_" + attribute,
                            getattr(video_object_source, attribute),
                        )
                    else:
                        setattr(
                            self,
                            attribute,
                            getattr(video_object_source, attribute),
                        )
                else:
                    setattr(
                        self,
                        "_" + attribute,
                        getattr(video_object_source, attribute),
                    )

    def compute_estimated_accuracy(self, fragments):
        weighted_P2 = 0
        number_of_individual_blobs = 0

        for fragment in fragments:
            if fragment.is_an_individual:
                if fragment.assigned_identities[0] != 0:
                    weighted_P2 += (
                        fragment.P2_vector[fragment.assigned_identities[0] - 1]
                        * fragment.number_of_images
                    )
                number_of_individual_blobs += fragment.number_of_images

        self._estimated_accuracy = weighted_P2 / number_of_individual_blobs

    def delete_accumulation_folders(self):
        for path in glob.glob(os.path.join(self.session_folder, "*")):
            if "accumulation_" in path or "pretraining" in path:
                rmtree(path, ignore_errors=True)

    def delete_data(self):
        logger.info("Data policy: {}".format(conf.DATA_POLICY))
        if conf.DATA_POLICY in [
            "trajectories",
            "validation",
            "knowledge_transfer",
            "idmatcher.ai",
        ]:

            if os.path.isdir(self._segmentation_data_folder):
                logger.info("Deleting segmentation images")
                rmtree(self._segmentation_data_folder, ignore_errors=True)
            if os.path.isfile(self.global_fragments_path):
                logger.info("Deleting global fragments")
                os.remove(self.global_fragments_path)
            if os.path.isfile(self.blobs_path_segmented):
                logger.info("Deleting blobs segmented")
                os.remove(self.blobs_path_segmented)
            if hasattr(self, "_crossings_detector_folder") and os.path.isdir(
                self.crossings_detector_folder
            ):
                logger.info("Deleting crossing detector folder")
                rmtree(self.crossings_detector_folder, ignore_errors=True)

        if conf.DATA_POLICY in [
            "trajectories",
            "validation",
            "knowledge_transfer",
        ]:
            if os.path.isdir(self._identification_images_folder):
                logger.info("Deleting identification images")
                rmtree(self._identification_images_folder, ignore_errors=True)

        if conf.DATA_POLICY in ["trajectories", "validation"]:
            logger.info("Deleting CNN models folders")
            self.delete_accumulation_folders()

        if conf.DATA_POLICY == "trajectories":
            if os.path.isdir(self.preprocessing_folder):
                logger.info("Deleting preprocessing data")
                rmtree(self.preprocessing_folder, ignore_errors=True)

    def get_first_frame(self, list_of_blobs):
        if self.user_defined_parameters["number_of_animals"] != 1:
            return self.first_frame_first_global_fragment[
                self.accumulation_trial
            ]
        elif self.user_defined_parameters["number_of_animals"] == 1:
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
