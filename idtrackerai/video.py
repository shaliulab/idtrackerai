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
    A class containing the main features of the video.

    This class includes properties of the video by itself, user defined
    parameters for the tracking, and other properties that are generated
    throughout the tracking process.

    We use this class as a storage of data coming from different processes.
    However, this is bad practice and it will change in the future.
    """

    def __init__(self, video_path, open_multiple_files):
        """Initializes a video object

        Parameters
        ----------
        video_path : str
            Path to a video file
        open_multiple_files : bool
            Flag to indicate that multiple files must be loaded
        """
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

        # User defined parameters
        self._user_defined_parameters = None  # has a setter
        self._open_multiple_files = open_multiple_files
        self._setup_points = []  # updated later

        # TODO: Should be part of the user defined parameters and not in constants
        if conf.SIGMA_GAUSSIAN_BLURRING is not None:
            self.sigma_gaussian_blurring = conf.SIGMA_GAUSSIAN_BLURRING

        # Get some other video features
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

        # TODO: HARDCODED _number_of_channels. Change if color information is used.
        # Currently idtracker.ai does not rely on color. All color videos
        # are converted to gray scale so the number of channels is forced to
        # always be one. This should be changed if color images are used for
        # identification, as this attributed is used to created the
        # identification images.
        self._number_of_channels = 1  # Used to create identification images

        # Attributes computed by other processes in the tracking
        # During segmentation
        self._maximum_number_of_blobs = 0  # initialized to 0. updated later
        self._frames_with_more_blobs_than_animals = None  # updated later
        # During crossing detection
        self._median_body_length = None  # updated later
        self._model_area = None  # updated later
        self._there_are_crossings = True  # updated later
        # During fragmentation
        self._fragment_identifier_to_index = None  # updated later
        # During tracking (protocol cascade)
        self._identity_transfer = None  # updated later
        self._tracking_with_knowledge_transfer = False  # updated later
        self._percentage_of_accumulated_images = None  # updated later
        self._first_frame_first_global_fragment = []  # updated later
        self._accumulation_trial = 0  # updated later
        self._knowledge_transfer_info_dict = None  # updated later
        # During validation (in validation GUI)
        self._identities_groups = {}  # updated later

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
        """Number of channels in the video"""
        return self._number_of_channels

    @property
    def episodes_start_end(self):
        """List of lists [frame_start, frame_end].

        Indicates the starting and ending frames of each video episode.
        Video episodes are used for parallelisation of some processes.
        """
        return self._episodes_start_end

    @property
    def video_path(self):
        """Path to the video selected for tracking."""
        return self._video_path

    @video_path.setter
    def video_path(self, video_path):
        """Sets the `_video_path` and `_video_folder` private attributes.

        Parameters
        ----------
        video_path : str
            Path to the video selected for tracking

        Raises
        ------
        ValueError
            If the video extension is not in AVAILABLE_VIDEO_EXTENSION.
            See idtrackerai.constants.
        """
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
        """Directory where video was stored. Parent of video_path.

        Returns
        -------
        str
            Path to the video folder where the video to be tracked was stored.
        """
        return self._video_folder

    @property
    def number_of_frames(self):
        """Total number of frames in the video to be tracked.

        Returns
        -------
        int
            Total number of frames in the video to be tracked. It considers
            all frames in all episodes. If the video consists of different
            files, the sum of the number of frames of all files is considered.

        See Also
        --------
        :method:`~idtrackerai.video.Video.get_num_frames_and_processing_episodes`
        """
        return self._number_of_frames

    @property
    def number_of_episodes(self):
        """Number of episodes in which the video is splitted for parallel
        processing.

        Returns
        -------
        int
            Number of parts in which the videos is splited.

        See Also
        --------
        :int:`~idtrackerai.constants.FRAMES_PER_EPISODE`
        """
        return self._number_of_episodes

    @property
    def open_multiple_files(self):
        """Flag indicate whether the video is composed of multiple files.

        Returns
        -------
        bool
            Some videos are stored with multiple files. This flag indicates
            whether the video to be tracked consisted of multiple files.

        See Also
        --------
        :method:`~idtrackerai.video.Video.get_video_paths`
        """
        return self._open_multiple_files

    @property
    def video_paths(self):
        """List of paths (str) indicate each of the files that compose the
        video

        Returns
        -------
        List[str]
            List of paths to the different files the video is composed of.
            If the video is a single file, the list will have length 1.

        See Also
        --------
        :method:`~idtrackerai.video.Video.get_video_paths`
        """
        return self._video_paths

    @property
    def original_width(self):
        """Original video width in pixels.

        Returns
        -------
        int
            Original video width in pixels. It does not consider the resolution
            reduction factor defined by the user.
        """
        return self._original_width

    @property
    def original_height(self):
        """Original video height in pixels.

        Returns
        -------
        int
            Original video width in pixels. It does not consider the resolution
            reduction factor defined by the user.
        """
        return self._original_height

    @property
    def width(self):
        """Video width in pixels after applying the resolution reduction
        factor.

        Returns
        -------
        int
            Video width in pixels after applying the resolution reduction
            factor defined by the user.
        """
        return np.round(
            self.original_width
            * self.user_defined_parameters["resolution_reduction"]
        ).astype(int)

    @property
    def height(self):
        """Video height in pixels after applying the resolution reduction
        factor.

        Returns
        -------
        int
            Video height in pixels after applying the resolution reduction
            factor.
        """
        return np.round(
            self.original_height
            * self.user_defined_parameters["resolution_reduction"]
        ).astype(int)

    @property
    def frames_per_second(self):
        """Video frame rate in frames per second.

        Returns
        -------
        int
            Video frame rate in frames per second obtained by OpenCV from the
            video file.
        """
        return self._frames_per_second

    # TODO: Not used. Check if necessary. Otherwise delete.
    @property
    def fragment_identifier_to_index(self):
        return self._fragment_identifier_to_index

    # Processing parameters properties

    @property
    def user_defined_parameters(self):
        """Dictionary with all the user-defined parameters for tracking.

        Returns
        -------
        dict
            Dictionary containing all the parameters defined by the user to
            perform the tracking.

        Raises
        ------
        Exception
            When `_users_defined_parameters is None`

        """
        if self._user_defined_parameters is not None:
            return self._user_defined_parameters
        else:
            raise Exception("No _user_defined_parameters given")

    @user_defined_parameters.setter
    def user_defined_parameters(self, value: Dict):
        """Sets the value of the private attribute `_user_defined_parameters`

        See Also:
        ---------
        base_idtrackerai.py in idtrackerai-app
        """
        self._user_defined_parameters = value

    # During processing properties

    # TODO: move to animal_detection
    @property
    def maximum_number_of_blobs(self):
        """Maximum number of blobs in a frame found during animals_detection"""
        return self._maximum_number_of_blobs

    # TODO: move to accumulation_manager.py
    @property
    def percentage_of_accumulated_images(self):
        return self._percentage_of_accumulated_images

    # TODO: move to constants.py
    @property
    def erosion_kernel_size(self):
        return self._erosion_kernel_size

    # TODO: move to animals_detection.py
    @property
    def frames_with_more_blobs_than_animals(self):
        return self._frames_with_more_blobs_than_animals

    # TODO: move to accumulation_manager.py
    @property
    def accumulation_trial(self):
        return self._accumulation_trial

    # TODO: move to tracker.py
    @property
    def estimated_accuracy(self):
        return self._estimated_accuracy

    # TODO: move to crossings_detection.py
    @property
    def identification_image_size(self):
        return self._identification_image_size

    # TODO: Probably not used. Check and delete
    @property
    def knowledge_transfer_info_dict(self):
        return self._knowledge_transfer_info_dict

    # TODO: move tracker.py
    @property
    def first_frame_first_global_fragment(self):
        return self._first_frame_first_global_fragment

    # TODO: move to crossings_detection.py where it is computed
    @property
    def median_body_length(self):
        """Median body length in pixels considering the resolution reduction
        factor
        """
        return self._median_body_length

    # TODO: move to crossings_detection.py
    @property
    def median_body_length_full_resolution(self):
        """Median body length in pixels in full frame resolution
        (i.e. without considering the resolution reduction factor)
        """
        return (
            self.median_body_length
            / self.user_defined_parameters["resolution_reduction"]
        )

    # TODO: move to crossings_detection.py
    @property
    def model_area(self):
        return self._model_area

    # TODO: move to crossings_detection.py
    @property
    def there_are_crossings(self):
        return self._there_are_crossings

    # TODO: move to accumulation_manager.py
    @property
    def ratio_accumulated_images(self):
        return self._ratio_accumulated_images

    # Processing steps
    # Flags to indicate whether the different processes have finished or not
    # It was used in the passed for the resume feature, but it is not active
    # in the current version
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

    # Attributes to store computational times of the different processses
    # TODO: each process class should have its own attribute to store this.
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
    # TODO: The different processes should create and store the path to the
    # folder where they save the data
    @property
    def preprocessing_folder(self):
        return self._preprocessing_folder

    @property
    def images_folder(self):
        return self._images_folder

    @property
    def crossings_detector_folder(self):
        return self._crossings_detector_folder

    # TODO: Probably not used, check and remove if not used.
    @property
    def previous_session_folder(self):
        return self._previous_session_folder

    @property
    def pretraining_folder(self):
        return self._pretraining_folder

    @property
    def accumulation_folder(self):
        return self._accumulation_folder

    # TODO: This should probably be the only path that should be stored in
    # Video.
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

    # Validation
    @property
    def identities_groups(self):
        """Groups of identities stored during the validation of the tracking
        in the validation GUI. This is useful to group identities in different
        classes depending on the experiment.

        This feature was coded becuase some users require indicating classes
        of individuals but we do not use it in the lab.
        """
        return self._identities_groups

    @property
    def is_centroid_updated(self):
        """Indicates whether the (x, y) centroid of some blobs has been updated
        during the validation process in the validation GUI.
        """
        return self._is_centroid_updated

    @is_centroid_updated.setter
    def is_centroid_updated(self, value):
        self._is_centroid_updated = value

    # Methods
    def save(self):
        """Saves the instantiated Video object.

        This is not good practices, as we are saving an object. We should be
        saving a dictionary and reconstruct the object from it in the load
        method.
        """
        # TODO: Do not save full objects. Save ad dictionary and reconstruct
        # the object in the load method.
        logger.info("saving video object in %s" % self.path_to_video_object)
        logger.debug(
            f"Video.save(open_multiple_files={self.open_multiple_files})"
        )
        np.save(self.path_to_video_object, self)

    @staticmethod
    def load(video_object_path):
        """Load a video object stored in a .npy file.

        In the future it should load a json file with information about the
        video and reconstruct the Video object from it.
        """
        video_object = np.load(video_object_path, allow_pickle=True).item()
        video_object.update_paths(video_object_path)
        return video_object

    @staticmethod
    def get_video_paths(video_path, open_multiple_files):
        """If the video is divided in episodes retrieves their paths"""
        video_paths = scan_folder(video_path)
        if len(video_paths) > 1 and open_multiple_files:
            return video_paths
        else:
            return [video_path]

    def update_paths(self, new_video_object_path):
        """Update paths of objects (e.g. blobs_path, preprocessing_folder...)
        according to the new location of the new video object given
        by `new_video_object_path`.

        Parameters
        ----------
        new_video_object_path : str
            Path to a video_object.npy
        """
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

    # TODO: Probably not used. Check and remove if not used.
    def rename_session_folder(self, new_session_name):
        assert new_session_name != ""
        new_session_name = "session_" + new_session_name
        current_session_name = os.path.split(self.session_folder)[1]
        logger.info("Updating checkpoint files")
        folders_to_check = [
            "video_folder",
            "preprocessing_folder",
            "logs_folder",
            "previous_session_folder",
            "crossings_detector_folder",
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
        """Gets some information about the video from the video file itself.

        Assign values to private attributes
        `_width`, `_height` and `_frames_per_second`.
        """

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

    # TODO: move to crossings_detection.py
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

    # Methods to create folders where to store data
    # TODO: Some of these methods should go to the classes corresponding to
    # the process.
    def create_session_folder(self, name=""):
        """Creates a folder where all the results of the tracking session
        will be stored.

        Parameters
        ----------
        name : str, optional
            Name of the tracking session, by default ""
        """
        if name == "":
            session_name = "session"
        else:
            session_name = "session_" + name

        self._session_folder = os.path.join(self.video_folder, session_name)
        logger.info(f"Creating session folder at {self._session_folder}")

        # TODO: `_previons_session_folder` is probably not used. Remove.
        if not os.path.isdir(self._session_folder):
            os.makedirs(self._session_folder)
            self._previous_session_folder = ""
        else:
            self._previous_session_folder = self.session_folder

        self._path_to_video_object = os.path.join(
            self.session_folder, "video_object.npy"
        )
        logger.info("the folder %s has been created" % self.session_folder)

    # TODO: It should be fragmented and moved to animals_detection.py and
    # crossings_detection.py. One for segmentation_data and other to
    # identification_images.
    def create_images_folders(self):
        """Creates folders to store segmentation images and identification
        images.
        """
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

    # Some methods related to the accumulation process
    # TODO: Move to accumulation_manager.py
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

    # TODO: Move to accumulation_manager.py
    def store_accumulation_step_statistics_data(self, new_values):
        [
            getattr(self, attr).append(value)
            for attr, value in zip(
                self.accumulation_statistics_attributes_list, new_values
            )
        ]

    # TODO: Move to accumulation_manager.py
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

    @staticmethod
    def get_num_frames_and_processing_episodes(video_paths):
        """Gets the number of frames in the video, the eposides of the video,
        and the number of episodes.

        Episodes are used to compute processes in parallel for different
        parts of the video

        Parameters
        ----------
        video_paths : list
            List of paths to each video file.

        Returns
        -------
        number_of_frames: int
            Total number of frames in the video
        episodes_start_end: list
            List of tuples (start, end) indicating the begining and end of
            each video episode
        number_of_episodes: int
            Number of episodes in the video.
        """
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
        """Given a `frame_number` of the whole video it returns the episode
        number.

        Parameters
        ----------
        frame_number : int
            Frame number considering all frames of the video.

        Returns
        -------
        int
            Episode number where the `frame_number` corresponds to.
        """
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

    # TODO: move to tracker.py
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

    def _delete_accumulation_folders(self):
        for path in glob.glob(os.path.join(self.session_folder, "*")):
            if "accumulation_" in path or "pretraining" in path:
                rmtree(path, ignore_errors=True)

    # TODO: DATA_POLICY should be a input argument to this function
    def delete_data(self):
        """Deletes some folders with data, to make the outcome lighter.

        Which folders are deleted depends on the constant DATA_POLICY
        """
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
            self._delete_accumulation_folders()

        if conf.DATA_POLICY == "trajectories":
            if os.path.isdir(self.preprocessing_folder):
                logger.info("Deleting preprocessing data")
                rmtree(self.preprocessing_folder, ignore_errors=True)

    # TODO: to list_of_global_fragments.py, list_of_blobs.py, or tracker.py
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


def scan_folder(path):
    """Returns all directories with same extension as the file indicated by
    `path`.

    Parameters
    ----------
    path : str
        Path to a video file

    Returns
    -------
    list
        List of paths with same extension as the file indicated by `path`
    """
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    folder = os.path.dirname(path)
    paths = glob.glob(folder + "/*" + extension)
    return natsorted(paths)
