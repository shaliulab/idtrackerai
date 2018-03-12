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
import cv2
import itertools
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from idtrackerai.preprocessing.fishcontour import FishContour
if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.blob")

class Blob(object):
    """ Object representing a blob (collection of pixels) segmented from a frame

    Attributes
    ----------

    frame_number : int
        Object containing all the parameters of the video.
    number_of_animals : int
        Number of animals to be tracked
    centroid : tuple
        Centroid as (x,y) in frame coordinate
    contour: list
        List of tuples (x,y) in frame coordinate of the pixels on the boundary of the blob
    area : int
        Number of pixels composing the blob
    bounding_box_in_frame_coordinates: list
        List of tuples [(x, y), (x + width, y + height)] of the bounding box rectangle enclosing the blob's pixels
    bounding_box_image: ndarray
        Image obtained by clipping the frame to the bounding box
    estimated_body_length: float
        Body length estimated from the bounding box
    _image_for_identification: ndarray
        Image fed to the network to get the identity of the animal the blob is associated to
    pixels : list
        List of ravelled pixels belonging to the blob (wrt the full frame)
    _is_an_individual : bool
        If True the blob is associated to a single animal
    _is_a_crossing : bool
        If True the blob is associated to two or more animals touching
    _was_a_crossing : bool
        If True the blob has been generated by splitting a crossing in postprocessing
    _is_a_misclassified_individual : bool
        This property can be modified only by the user during validation. It identifies a blob that was by mistaken associated to a crossing by the DeepCrossingDetector
    next : list
        List of blob objects segmented in self.frame_number + 1 whose list of pixels intersect self.pixels
    previous : list
        List of blob objects segmented in self.frame_number - 1 whose list of pixels intersect self.pixels
    _fragment_identifier : int
        Unique integer identifying the fragment (built by blob overlapping) to which self belongs to
    _blob_index : int
        Hierarchy of the blob
    _used_for_training : bool
        If True the image obtained from the blob has been used to train the idCNN
    _accumulation_step : int
        Accumulation step in which the image associated to the blob has been accumulated
    _generated_while_closing_the_gap : bool
        If True the blob has been generated while solving the crossings
    _user_generated_identity : int
        The identity corrected manually by the user during validation
    _identity_corrected_closing_gaps : int
        The identity given to the blob during in postprocessing
    _identity_corrected_solving_duplication : int
        The identity given to the blob while solving duplications
    _identity : int
        Identity associated to the blob
    is_identified : bool
        True if self.identity is not None
    final_identity : int
        Identity assigned to self after validation
    assigned_identity : int
        Identity assigned to self by the algorithm (ignoring eventual correction made by the user during validation)
    has_ambiguous_identity: bool
        True if during either accumulation of residual identification the blob has
        been associated with equal probability to two (or more) distinct identities
    nose_coordinates : tuple
        Coordinate of the nose of the blob (only for zebrafish)
    head_coordinates : tuple
        Coordinate of the centroid of the head of the blob (only for zebrafish)
    extreme1_coordinate : tuple

    extreme2_coordinates : tuple

    """
    def __init__(self, centroid, contour, area,
                bounding_box_in_frame_coordinates, bounding_box_image = None,
                estimated_body_length = None, pixels = None,
                number_of_animals = None, frame_number = None):
        self.frame_number = frame_number
        self.number_of_animals = number_of_animals
        self.centroid = np.array(centroid) # numpy array (int64): coordinates of the centroid of the blob in pixels
        self.contour = contour # openCV contour [[[x1,y1]],[[x2,y2]],...,[[xn,yn]]]
        self.area = area # int: number of pixels in the blob
        self.bounding_box_in_frame_coordinates = bounding_box_in_frame_coordinates #tuple of tuples: ((x1,y1),(x2,y2)) (top-left corner, bottom-right corner) in pixels
        self.bounding_box_image = bounding_box_image # numpy array (uint8): image of the fish cropped from the video according to the bounding_box_in_frame_coordinates
        self.estimated_body_length = estimated_body_length
        self._image_for_identification = None # numpy array (float32)
        self.pixels = pixels # list of int's: linearized pixels of the blob
        self._is_an_individual = False
        self._is_a_crossing = False
        self._was_a_crossing = False
        self._is_a_misclassified_individual = False
        self.next = [] # next blob object overlapping in pixels with current blob object
        self.previous = [] # previous blob object overlapping in pixels with the current blob object
        self._fragment_identifier = None # identity in individual fragment after fragmentation
        self._blob_index = None # index of the blob to plot the individual fragments
        self._used_for_training = None
        self._accumulation_step = None
        self._generated_while_closing_the_gap = False
        self._user_generated_identity = None
        self._identity_corrected_closing_gaps = None
        self._identity_corrected_solving_duplication = None
        self._identity = None

    @property
    def fragment_identifier(self):
        return self._fragment_identifier

    @property
    def is_an_individual(self):
        return self._is_an_individual

    @property
    def is_a_crossing(self):
        return self._is_a_crossing

    @property
    def was_a_crossing(self):
        return self._was_a_crossing

    def check_for_multiple_next_or_previous(self, direction = None):
        """Return True if self has multiple blobs in its past or future overlapping
        history of the blob

        Parameters
        ----------
        direction : str
            "previous" or "next". If "previous" the past overlapping history will
            be checked in order to find out if the blob will split in the past.
            Symmetrically, if "next" the future overlapping history of the blob
            will be checked

        Returns
        -------
        Bool
            If True the blob splits into two or multiple overlapping blobs in its
            "past" or "future" history, depending on the parameter "direction"

        """
        opposite_direction = 'next' if direction == 'previous' else 'previous'
        current = getattr(self, direction)[0]

        while len(getattr(current, direction)) == 1 :

            current = getattr(current, direction)[0]
            if len(getattr(current, direction)) > 1:# or\
                return True
                break

        return False

    def is_a_sure_individual(self):
        """A blob marked as a sure individual will be used to train the Deep
        Crossing Detector (the artificial neural network used to disriminate
        images associated with individual from ones associated to crossings).

        Returns
        -------
        Bool
            Blob is a sure individual if:

            * it overlaps with one and only one blob in both the immediate past and future frames;
            * it never splits in both its past and future overlapping history


        """
        if self.is_an_individual and len(self.previous) == 1 \
            and len(self.next) == 1 and len(self.next[0].previous) == 1 and\
            len(self.previous[0].next) == 1:
            has_multiple_previous = self.check_for_multiple_next_or_previous('previous')
            has_multiple_next = self.check_for_multiple_next_or_previous('next')
            if not has_multiple_previous and not has_multiple_next:
                return True
        else:
            return False

    def is_a_sure_crossing(self):
        """A blob marked as a sure crossing will be used to train the Deep
        Crossing Detector (the artificial neural network used to disriminate
        images associated with individual from ones associated to crossings).

        Returns
        -------
        Bool
            Blob is a sure crossing if:

            * it overlaps with one and only one blob in both the immediate past and future frames;
            * it splits in both its past and future overlapping history

        """
        if self.is_a_crossing and (len(self.previous) > 1 or len(self.next) > 1):
            return True
        elif self.is_a_crossing and len(self.previous) == 1 and len(self.next) == 1:
            has_multiple_previous = self.check_for_multiple_next_or_previous('previous')
            has_multiple_next = self.check_for_multiple_next_or_previous('next')
            if has_multiple_previous and has_multiple_next:
                return True
        else:
            return False

    @property
    def has_ambiguous_identity(self):
        return self.is_an_individual_in_a_fragment and self.identity is list

    def overlaps_with(self, other):
        """Given a second blob object, checks if the lists of pixels of the two
        blobs intersect

        Parameters
        ----------
        other : <Blob object>
            An instance of the class Blob

        Returns
        -------
        Bool
            True if the lists of pixels have non-empty intersection
        """
        overlaps = False
        intersection = np.intersect1d(self.pixels, other.pixels)
        if len(intersection) > 0:
            overlaps = True
        return overlaps

    def now_points_to(self, other):
        """Given two consecutive blob objects updates their respective overlapping
        histories

        Parameters
        ----------
        other : <Blob object>
            An instance of the class Blob
        """
        self.next.append(other)
        other.previous.append(self)

    def squared_distance_to(self, other):
        """Returns the squared distance from the centroid of self to the centroid
        of other

        Parameters
        ----------
        other : <Blob object> or tuple
            An instance of the class Blob or a tuple (x,y)

        Returns
        -------
        float
            Squared distance between centroids

        """
        if isinstance(other, Blob):
            return np.sum((np.asarray(self.centroid) - np.asarray(other.centroid))**2)
        elif isinstance(other, (tuple, list, np.ndarray)):
            return np.sum((np.asarray(self.centroid) - np.asarray(other))**2)

    def distance_from_countour_to(self, point):
        """Returns the distance between the point passed as input and the closes
        point belonging to the contour of the blob.

        Parameters
        ----------
        point : tuple
            (x,y)

        Returns
        -------
        float
            :math:`\min_{c\in \mbox{ blob.contour}}(d(c, point))`, where :math:`d`
            is the Euclidean distance.

        """
        return np.abs(cv2.pointPolygonTest(self.contour, point, True))

    @property
    def used_for_training(self):
        return self._used_for_training

    @property
    def accumulation_step(self):
        return self._accumulation_step

    @property
    def is_in_a_fragment(self):
        return len(self.previous) == len(self.next) == 1

    @property
    def is_an_individual_in_a_fragment(self):
        return self.is_an_individual and self.is_in_a_fragment

    @property
    def is_a_misclassified_individual(self):
        return self._is_a_misclassified_individual

    @property
    def blob_index(self):
        return self._blob_index

    @blob_index.setter
    def blob_index(self, new_blob_index):
        if self.is_an_individual_in_a_fragment:
            self._blob_index = new_blob_index

    @property
    def image_for_identification(self):
        return self._image_for_identification

    @property
    def nose_coordinates(self):
        return self._nose_coordinates

    @property
    def head_coordinates(self):
        return self._head_coordinates

    @property
    def extreme1_coordinate(self):
        return self._extreme1_coordinates

    @property
    def extreme2_coordinates(self):
        return self._extreme2_coordinates

    def in_a_global_fragment_core(self, blobs_in_frame):
        """A blob in a frame is in the core of a global fragment if in
        that frame there are as many blobs as number of animals to track

        Parameters
        ----------
        blobs_in_frame : list
            List of Blob objects representing the animals segmented in the frame
            self.frame_number

        Returns
        -------
        Bool
            True if the blob is in the core of a global fragment

        """
        return len(blobs_in_frame) == self.number_of_animals

    @property
    def identity(self):
        return self._identity

    @property
    def user_generated_identity(self):
        return self._user_generated_identity

    @user_generated_identity.setter
    def user_generated_identity(self, new_value):
        if self.is_a_crossing and self.assigned_identity is None:
            self._is_an_individual = True
            self._is_a_crossing = False
            self._is_a_misclassified_individual = True

    @property
    def identity_corrected_solving_duplication(self):
        return self._identity_corrected_solving_duplication

    @property
    def identity_corrected_closing_gaps(self):
        return self._identity_corrected_closing_gaps

    @property
    def final_identity(self):
        if hasattr(self, 'user_generated_identity') and self.user_generated_identity is not None:
            return self.user_generated_identity
        else:
            return self.assigned_identity

    @property
    def assigned_identity(self):
        if hasattr(self, 'identity_corrected_closing_gaps') and self.identity_corrected_closing_gaps is not None:
            return self.identity_corrected_closing_gaps
        elif hasattr(self, 'identity_corrected_solving_duplication') and self.identity_corrected_solving_duplication is not None:
            return self.identity_corrected_solving_duplication
        else:
            return self.identity

    @property
    def is_identified(self):
        return self._identity is not None

    @property
    def generated_while_closing_the_gap(self):
        return self._generated_while_closing_the_gap

    def compute_overlapping_with_previous_blob(self):
        number_of_previous_blobs = len(self.previous)
        if number_of_previous_blobs == 1:
            self.non_shared_information_with_previous = 1. - len(np.intersect1d(self.pixels, self.previous[0].pixels)) / np.mean([len(self.pixels), len(self.previous[0].pixels)])
            if self.non_shared_information_with_previous is np.nan:
                logger.debug("intersection both blobs %s" %str(len(np.intersect1d(self.pixels, self.previous[0].pixels))))
                logger.debug("mean pixels both blobs %s" %str(np.mean([len(self.pixels), len(self.previous[0].pixels)])))
                raise ValueError("non_shared_information_with_previous is nan")

    def apply_model_area(self, video, number_of_animals, model_area, identification_image_size, number_of_blobs):
        """Classify self as a crossing or individual blob according to its area

        Parameters
        ----------
        video : <Video object>
            Object containing all the parameters of the video.
        number_of_animals : int
            Number of animals to be tracked
        model_area : function
            Model of the area blobs representing individual animals
        identification_image_size : tuple
            Shape of the images used for the identification
        number_of_blobs : int
            Number of blobs segmented in the frame self.frame_number

        """
        if model_area(self.area) or number_of_blobs == number_of_animals: #Checks if area is compatible with the model area we built
            self.set_image_for_identification(video)
            self._is_an_individual = True
        else:
            self._is_a_crossing = True

    def set_image_for_identification(self, video):
        """Set the image that will be used to identitfy the animal with the idCNN

        Parameters
        ----------
        video : <Video object>
            Object containing all the parameters of the video.

        """
        self._image_for_identification, \
        self._extreme1_coordinates, \
        self._extreme2_coordinates, _ = self.get_image_for_identification(video)

    def get_image_for_identification(self, video, folder_to_save_for_paper_figure = '', image_size = None):
        """Compute the image that will be used to identify the animal with the idCNN

        Parameters
        ----------
        video : <Video object>
            Object containing all the parameters of the video.
        folder_to_save_for_paper_figure : str
            Path to the folder in which the image will be saved. If '' the image
            will not be saved.
        image_size : tuple
            Shape of the image

        Returns
        -------
        ndarray
            Stadardised image

        """
        if image_size is None:
            image_size = video.identification_image_size[0]

        return self._get_image_for_identification(video.height, video.width,
                                                self.bounding_box_image, self.pixels,
                                                self.bounding_box_in_frame_coordinates,
                                                image_size,
                                                folder_to_save_for_paper_figure = folder_to_save_for_paper_figure)

    @staticmethod
    def _get_image_for_identification(height, width, bounding_box_image, pixels, bounding_box_in_frame_coordinates, identification_image_size, folder_to_save_for_paper_figure = ''):
        """Short summary.

        Parameters
        ----------
        height : int
            Frame height
        width : int
            Frame width
        bounding_box_image : ndarray
            Images cropped from the frame by considering the bounding box associated to a blob
        pixels : list
            List of pixels associated to a blob
        bounding_box_in_frame_coordinates : list
            [(x, y), (x + bounding_box_width, y + bounding_box_height)]
        identification_image_size : tuple
            shape of the identification image
        folder_to_save_for_paper_figure : str
            folder to save the images for identification

        Returns
        -------
        ndarray
            Standardised image

        """
        if folder_to_save_for_paper_figure:
            save_preprocessing_step_image(bounding_box_image/255, folder_to_save_for_paper_figure, name = '0_bounding_box_image', min_max = [0, 1])
        bounding_box_image = remove_background_pixels(height, width, bounding_box_image, pixels, bounding_box_in_frame_coordinates, folder_to_save_for_paper_figure)
        pca = PCA()
        pxs = np.unravel_index(pixels,(height,width))
        pxs1 = np.asarray(zip(pxs[1],pxs[0]))
        pca.fit(pxs1)
        rot_ang = np.arctan(pca.components_[0][1]/pca.components_[0][0])*180/np.pi + 45 # we substract 45 so that the fish is aligned in the diagonal. This way we have smaller frames
        center = (pca.mean_[0], pca.mean_[1])
        center = full2miniframe(center, bounding_box_in_frame_coordinates)
        center = np.array([int(center[0]), int(center[1])])

        if folder_to_save_for_paper_figure:
            pxs_for_plot = np.array(pxs).T
            pxs_for_plot = np.array([pxs_for_plot[:, 0] - bounding_box_in_frame_coordinates[0][1], pxs_for_plot[:, 1] - bounding_box_in_frame_coordinates[0][0]])
            temp_image = np.zeros_like(bounding_box_image).astype('uint8')
            temp_image[pxs_for_plot[0,:], pxs_for_plot[1,:]] = 255
            slope = np.tan((rot_ang - 45) * np.pi / 180)
            X = [0, temp_image.shape[1]]
            Y = [-slope * center[0] + center[1], slope * (X[1] - center[0]) + center[1]]
            save_preprocessing_step_image(temp_image/255, folder_to_save_for_paper_figure, name = '4_blob_with_PCA_axes', min_max = [0, 1], draw_line = (X, Y))

        #rotate
        diag = np.sqrt(np.sum(np.asarray(bounding_box_image.shape)**2)).astype(int)
        diag = (diag, diag)
        M = cv2.getRotationMatrix2D(tuple(center), rot_ang, 1)
        minif_rot = cv2.warpAffine(bounding_box_image, M, diag, borderMode=cv2.BORDER_CONSTANT, flags = cv2.INTER_CUBIC)


        crop_distance = int(identification_image_size/2)
        x_range = xrange(center[0] - crop_distance, center[0] + crop_distance)
        y_range = xrange(center[1] - crop_distance, center[1] + crop_distance)
        image_for_identification = minif_rot.take(y_range, mode = 'wrap', axis=0).take(x_range, mode = 'wrap', axis=1)
        height, width = image_for_identification.shape

        rot_ang_rad = rot_ang * np.pi / 180
        h_or_t_1 = np.array([np.cos(rot_ang_rad), np.sin(rot_ang_rad)]) * rot_ang_rad
        h_or_t_2 = - h_or_t_1
        if folder_to_save_for_paper_figure:
            save_preprocessing_step_image(image_for_identification/255, folder_to_save_for_paper_figure, name = '5_blob_dilated_rotated', min_max = [0, 1])
        image_for_identification_standarised = ((image_for_identification - np.mean(image_for_identification))/np.std(image_for_identification)).astype('float32')
        if folder_to_save_for_paper_figure:
            save_preprocessing_step_image(image_for_identification_standarised, folder_to_save_for_paper_figure, name = '6_blob_dilated_rotated_normalized', min_max = [np.min(image_for_identification), np.max(image_for_identification)])

        return image_for_identification_standarised, tuple(h_or_t_1.astype('int')), tuple(h_or_t_2.astype('int')), image_for_identification

    def get_nose_and_head_coordinates(self):
        """Only for zebrafish: Compute the nose coordinate according to [1]_

        .. [1] Wang, Shuo Hong, et al. "Automated planar tracking the waving
            bodies of multiple zebrafish swimming in shallow water."
            PloS one 11.4 (2016): e0154714.
        """
        if self.is_an_individual:
            contour_cnt = FishContour.fromcv2contour(self.contour)
            noseFull, _, head_centroid_full = contour_cnt.find_nose_and_orientation()
            self._nose_coordinates = tuple(noseFull.astype('float32'))
            self._head_coordinates = tuple(head_centroid_full.astype('float32'))
        else:
            self._nose_coordinates = None
            self._head_coordinates = None

def remove_background_pixels(height, width, bounding_box_image, pixels, bounding_box_in_frame_coordinates, folder_to_save_for_paper_figure):
    """Removes the background pixels substiuting them with a homogeneous black
    background.

    Parameters
    ----------
    height : int
        Frame height
    width : int
        Frame width
    bounding_box_image : ndarray
        Images cropped from the frame by considering the bounding box associated to a blob
    pixels : list
        List of pixels associated to a blob
    bounding_box_in_frame_coordinates : list
        [(x, y), (x + bounding_box_width, y + bounding_box_height)]
    identification_image_size : tuple
        shape of the identification image
    folder_to_save_for_paper_figure : str
        folder to save the images for identification

    Returns
    -------
    ndarray
        Image with black background pixels

    """
    pxs = np.array(np.unravel_index(pixels,(height, width))).T
    pxs = np.array([pxs[:, 0] - bounding_box_in_frame_coordinates[0][1], pxs[:, 1] - bounding_box_in_frame_coordinates[0][0]])
    temp_image = np.zeros_like(bounding_box_image).astype('uint8')
    temp_image[pxs[0,:], pxs[1,:]] = 255
    if folder_to_save_for_paper_figure:
        new_thresholded_image = temp_image.copy()
        new_bounding_box_image = bounding_box_image.copy()
        new_bounding_box_image = cv2.cvtColor(new_bounding_box_image, cv2.COLOR_GRAY2RGB)
        contours, hierarchy = cv2.findContours(new_thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(new_bounding_box_image, contours, -1, (255,0,0), 1)
        save_preprocessing_step_image(new_bounding_box_image/255, folder_to_save_for_paper_figure, name = '1_blob_contour',  min_max = [0, 1])
    temp_image = cv2.dilate(temp_image, np.ones((3,3)).astype('uint8'), iterations = 1)
    if folder_to_save_for_paper_figure:
        new_thresholded_image = temp_image.copy()
        new_bounding_box_image = bounding_box_image.copy()
        new_bounding_box_image = cv2.cvtColor(new_bounding_box_image, cv2.COLOR_GRAY2RGB)
        contours, hierarchy = cv2.findContours(new_thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(new_bounding_box_image, contours, -1, (255,0,0), 1)
        save_preprocessing_step_image(new_bounding_box_image/255, folder_to_save_for_paper_figure, name = '2_blob_bw_dilated',  min_max = [0, 1])
    rows, columns = np.where(temp_image == 255)
    dilated_pixels = np.array([rows, columns])

    temp_image[dilated_pixels[0,:], dilated_pixels[1,:]] = bounding_box_image[dilated_pixels[0,:], dilated_pixels[1,:]]
    if folder_to_save_for_paper_figure:
        save_preprocessing_step_image(temp_image/255, folder_to_save_for_paper_figure, name = '3_blob_contour_dilated',  min_max = [0, 1])
    return temp_image

def full2miniframe(point, boundingBox):
    """Maps a point in the fullframe to the coordinate system defined by the image
    generated by considering the bounding box of the blob.
    Here it is use for centroids

    Parameters
    ----------
    point : tuple
        (x, y)
    boundingBox : list
        [(x, y), (x + bounding_box_width, y + bounding_box_height)]

    Returns
    -------
    tuple
        :math:`(x^\prime, y^\prime)`

    """
    return tuple(np.asarray(point) - np.asarray([boundingBox[0][0],boundingBox[0][1]]))

def save_preprocessing_step_image(image, save_folder, name = None, min_max = None, draw_line = None):
    from matplotlib import pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(1,1)
    ax.imshow(image, cmap = 'gray', vmin = min_max[0], vmax = min_max[1])
    if draw_line is not None:
        plt.plot(draw_line[0], draw_line[1], 'b-')
        ax.set_xlim([0, image.shape[1]])
        ax.set_ylim([0, image.shape[0]])
        ax.invert_yaxis()

    ax.set_xticks([])
    ax.set_yticks([])

    sns.despine(left = True, right = True, top = True, bottom = True)
    fig.savefig(os.path.join(save_folder,'%s.pdf' %name), transparent=True, bbox_inches='tight')
