from __future__ import absolute_import, division, print_function
import sys
sys.path.append('./utils')
sys.path.append('./preprocessing')

import itertools
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import logging

from get_portraits import get_portrait, get_body

logger = logging.getLogger("__main__.blob")

class Blob(object):
    def __init__(self, centroid, contour, area, bounding_box_in_frame_coordinates, bounding_box_image = None, estimated_body_length = None, portrait = None, pixels = None, number_of_animals = None, frame_number = None):
        self.frame_number = frame_number
        self.number_of_animals = number_of_animals
        self.centroid = np.array(centroid) # numpy array (int64): coordinates of the centroid of the blob in pixels
        self.contour = contour # openCV contour [[[x1,y1]],[[x2,y2]],...,[[xn,yn]]]
        self.area = area # int: number of pixels in the blob
        self.bounding_box_in_frame_coordinates = bounding_box_in_frame_coordinates #tuple of tuples: ((x1,y1),(x2,y2)) (top-left corner, bottom-right corner) in pixels
        self.bounding_box_image = bounding_box_image # numpy array (uint8): image of the fish cropped from the video according to the bounding_box_in_frame_coordinates
        self.estimated_body_length = estimated_body_length
        self._portrait = portrait # numpy array (float32)
        self.pixels = pixels # list of int's: linearized pixels of the blob
        self.reset_before_fragmentation('fragmentation')

    def reset_before_fragmentation(self, recovering_from):
        if recovering_from == 'fragmentation':
            self.next = [] # next blob object overlapping in pixels with current blob object
            self.previous = [] # previous blob object overlapping in pixels with the current blob object
            self._fragment_identifier = None # identity in individual fragment after fragmentation
            self._blob_index = None # index of the blob to plot the individual fragments

    @property
    def fragment_identifier(self):
        return self._fragment_identifier

    @property
    def user_generated_identity(self):
        return self._user_generated_identity

    @user_generated_identity.setter
    def user_generated_identity(self, new_identifier):
        self._user_generated_identity = new_identifier

    @property
    def is_a_fish(self):
        return self.portrait is not None

    @property
    def is_a_jump(self):
        is_a_jump = False
        if self.is_a_fish and len(self.next) == 0 and len(self.previous) == 0: # 1 frame jumps
            is_a_jump = True
        return is_a_jump

    @property
    def is_a_jumping_fragment(self):
        # this is a fragment of 2 frames that it is not considered a individual fragment but it is also not a single frame jump
        is_a_jumping_fragment = False
        if self.is_a_fish and len(self.next) == 0 and len(self.previous) == 1 and len(self.previous[0].previous) == 0 and len(self.previous[0].next) == 1: # 2 frames jumps
            is_a_jumping_fragment = True
        elif self.is_a_fish and len(self.next) == 1 and len(self.previous) == 0 and len(self.next[0].next) == 0 and len(self.next[0].previous) == 1: # 2 frames jumps
            is_a_jumping_fragment = True
        return is_a_jumping_fragment

    @property
    def is_a_ghost_crossing(self):
        return (self.is_a_fish and (len(self.next) != 1 or len(self.previous) != 1))

    @property
    def is_a_crossing(self):
        return self.portrait is None

    @property
    def has_ambiguous_identity(self):
        return self.is_a_fish_in_a_fragment and self.identity is list

    def overlaps_with(self, other):
        """Checks if pixels are disjoint
        """
        overlaps = False
        intersection = np.intersect1d(self.pixels, other.pixels)
        if len(intersection) > 0:
            overlaps = True
        return overlaps

    def now_points_to(self, other):
        self.next.append(other)
        other.previous.append(self)

    @property
    def is_in_a_fragment(self):
        return len(self.previous) == len(self.next) == 1

    @property
    def is_a_fish_in_a_fragment(self):
        return self.is_a_fish and self.is_in_a_fragment

    @property
    def blob_index(self):
        return self._blob_index

    @blob_index.setter
    def blob_index(self, new_blob_index):
        if self.is_a_fish_in_a_fragment:
            self._blob_index = new_blob_index

    @property
    def portrait(self):
        return self._portrait

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

    @property
    def assigned_during_accumulation(self):
        return self._assigned_during_accumulation

    def in_a_global_fragment_core(self, blobs_in_frame):
        '''a blob in a frame is a fish in the core of a global fragment if in
        that frame there are as many blobs as number of animals to track
        '''
        return len(blobs_in_frame) == self.number_of_animals

    @property
    def identity(self):
        return self._identity

    @identity.setter
    def identity(self, new_identifier):
        if type(new_identifier) is 'int':
            assert self.is_a_fish
        elif type(new_identifier) is 'list':
            assert self.is_a_crossing

        self._identity = new_identifier

    @property
    def is_identified(self):
        return self._identity is not None

    @property
    def frequencies_in_fragment(self):
        return self._frequencies_in_fragment

    @property
    def P1_vector(self):
        return self._P1_vector

    @property
    def P2_vector(self):
        return self._P2_vector

    def _along_transitions_in_individual_fragments(self, function):
        '''Crawls along an individual fragment and outputs a list with
        the result of function applied to all pairs of contiguous blobs
        '''
        output_along_fragment = []
        if self.is_a_fish_in_a_fragment:
            current = self
            while len(current.next) > 0 and current.next[0].fragment_identifier == self.fragment_identifier:
                output_along_fragment.append(function(current, current.next[0]))
                current = current.next[0]

            current = self

            while len(current.previous) > 0 and current.previous[0].fragment_identifier == self.fragment_identifier:
                output_along_fragment.append(function(current, current.previous[0]))
                current = current.previous[0]

        return output_along_fragment

    def _along_blobs_in_individual_fragment(self, function):
        '''Crawls along an individual fragment and outputs a list with
        the result of function applied to each blob
        '''
        output_along_fragment = []
        if self.is_a_fish_in_a_fragment:

            current = self
            output_along_fragment.append(function(current))
            while len(current.next) > 0 and current.next[0].fragment_identifier == self.fragment_identifier:
                current = current.next[0]
                output_along_fragment.append(function(current))

            current = self
            while len(current.previous) > 0 and current.previous[0].fragment_identifier == self.fragment_identifier:
                current = current.previous[0]
                output_along_fragment.append(function(current))

        return output_along_fragment


    def frame_by_frame_velocity(self):
        def distance_between_centroids(blob1, blob2):
            #This can be rewritten more elegantly with decorators
            #Also, it is faster if declared with global scope. Feel free to change
            return np.linalg.norm(blob1.centroid - blob2.centroid)
        return self._along_transitions_in_individual_fragments(distance_between_centroids)

    def distance_travelled_in_fragment(self):
        return sum(self.frame_by_frame_velocity())

    def compute_fragment_start_end(self):
        def frame_number_of_blob(blob):
            return blob.frame_number
        frame_numbers = self._along_blobs_in_individual_fragment(frame_number_of_blob)
        return [min(frame_numbers), max(frame_numbers)]

    def portraits_in_fragment(self):
        def return_portrait_blob(blob):
            return blob.portrait
        return self._along_blobs_in_individual_fragment(return_portrait_blob)

    def non_shared_information_in_fragment(self):
        def return_non_shared_information(blob):
            return blob.non_shared_information_with_previous
        return self._along_blobs_in_individual_fragment(return_non_shared_information)

    def identities_in_fragment(self):
        def blob_identity(blob):
            return blob._identity
        return self._along_blobs_in_individual_fragment(blob_identity)

    @staticmethod
    def get_coexisting_blobs(blob, blobs_in_video, fragment_identifiers_of_coexisting_fragments, coexisting_blobs):
        """Returns the list of blobs coexisting with blob"""
        #blob_ is the blob object in the same frame as blob
        for blob_ in blobs_in_video[blob.frame_number]:
            if blob_.fragment_identifier is not blob.fragment_identifier and \
                    blob_.fragment_identifier not in fragment_identifiers_of_coexisting_fragments and \
                    blob_.fragment_identifier is not None:
                coexisting_blobs.append(blob_)
                fragment_identifiers_of_coexisting_fragments.append(blob_.fragment_identifier)

        return coexisting_blobs, fragment_identifiers_of_coexisting_fragments

    def get_coexisting_blobs_in_fragment(self, blobs_in_video):
        coexisting_blobs = []
        fragment_identifiers_of_coexisting_fragments = []
        coexisting_blobs, fragment_identifiers_of_coexisting_fragments = self.get_coexisting_blobs(self, blobs_in_video, fragment_identifiers_of_coexisting_fragments, coexisting_blobs)
        if self.is_a_fish_in_a_fragment:
            current = self

            # while current.next[0].is_a_fish_in_a_fragment:
            while len(current.next) > 0 and current.next[0].fragment_identifier == self.fragment_identifier:
                current = current.next[0]
                coexisting_blobs, fragment_identifiers_of_coexisting_fragments = self.get_coexisting_blobs(current, blobs_in_video, fragment_identifiers_of_coexisting_fragments, coexisting_blobs)

            current = self

            # while current.previous[0].is_a_fish_in_a_fragment:
            while len(current.previous) > 0 and current.previous[0].fragment_identifier == self.fragment_identifier:
                current = current.previous[0]
                coexisting_blobs, fragment_identifiers_of_coexisting_fragments = self.get_coexisting_blobs(current, blobs_in_video, fragment_identifiers_of_coexisting_fragments, coexisting_blobs)

        return coexisting_blobs, fragment_identifiers_of_coexisting_fragments

    def get_fixed_identities_of_coexisting_fragments(self, blobs_in_video):
        identities = []
        fragment_identifiers_of_coexisting_fragments = []
        for b, blob in enumerate(blobs_in_video[self.frame_number]):
            if blob.fragment_identifier is not self.fragment_identifier and \
                    blob.fragment_identifier not in fragment_identifiers_of_coexisting_fragments and \
                    blob.fragment_identifier is not None \
                    and (blob.assigned_during_accumulation \
                    or (blob._is_a_duplication and blob._identity_corrected_solving_duplication is not None and blob._identity_corrected_solving_duplication != 0)\
                    or not blob._is_a_duplication):
                if blob._user_generated_identity is not None:
                    identities.append(blob._user_generated_identity)
                elif blob._identity_corrected_solving_duplication is not None:
                    identities.append(blob._identity_corrected_solving_duplication)
                elif blob.identity is not None:
                    identities.append(blob.identity)
                fragment_identifiers_of_coexisting_fragments.append(blob.fragment_identifier)

        if self.is_a_fish_in_a_fragment:
            current = self

            while len(current.next) > 0 and current.next[0].fragment_identifier == self.fragment_identifier:
                current = current.next[0]
                for b, blob in enumerate(blobs_in_video[current.frame_number]):
                    if blob.fragment_identifier is not current.fragment_identifier and \
                            blob.fragment_identifier not in fragment_identifiers_of_coexisting_fragments and \
                            blob.fragment_identifier is not None \
                            and (blob.assigned_during_accumulation \
                            or (blob._is_a_duplication and blob._identity_corrected_solving_duplication is not None and blob._identity_corrected_solving_duplication != 0)\
                            or not blob._is_a_duplication):
                        if blob._user_generated_identity is not None:
                            identities.append(blob._user_generated_identity)
                        elif blob._identity_corrected_solving_duplication is not None:
                            identities.append(blob._identity_corrected_solving_duplication)
                        elif blob.identity is not None:
                            identities.append(blob.identity)
                        fragment_identifiers_of_coexisting_fragments.append(blob.fragment_identifier)

            current = self
            while len(current.previous) > 0 and current.previous[0].fragment_identifier == self.fragment_identifier:
                current = current.previous[0]
                for blob in blobs_in_video[current.frame_number]:
                    if blob.fragment_identifier is not current.fragment_identifier and \
                            blob.fragment_identifier not in fragment_identifiers_of_coexisting_fragments and \
                            blob.fragment_identifier is not None \
                            and (blob.assigned_during_accumulation \
                            or (blob._is_a_duplication and blob._identity_corrected_solving_duplication is not None and blob._identity_corrected_solving_duplication != 0)\
                            or not blob._is_a_duplication):
                        if blob._user_generated_identity is not None:
                            identities.append(blob._user_generated_identity)
                        elif blob._identity_corrected_solving_duplication is not None:
                            identities.append(blob._identity_corrected_solving_duplication)
                        elif blob.identity is not None:
                            identities.append(blob.identity)
                        fragment_identifiers_of_coexisting_fragments.append(blob.fragment_identifier)
        return np.unique(np.asarray(identities)), fragment_identifiers_of_coexisting_fragments

    def set_identity_blob_in_fragment(self, identity_in_fragment, duplication_solved, P1_vector, frequencies_in_fragment):
        if not duplication_solved:
            self._identity = identity_in_fragment
        elif duplication_solved:
            self._identity_corrected_solving_duplication = identity_in_fragment
        self._frequencies_in_fragment = frequencies_in_fragment
        self._P1_vector = P1_vector

    def update_blob_assigned_during_accumulation(self):
            self._assigned_during_accumulation = True


    @staticmethod
    def update_identity_in_fragment_in_direction(current, \
                                            identity_in_fragment, \
                                            assigned_during_accumulation, \
                                            P1_vector, frequencies_in_fragment, \
                                            duplication_solved, direction = None, \
                                            fragment_identifier = None):
        if direction == 'next':
            opposite_direction = 'previous'
        elif direction == 'previous':
            opposite_direction = 'next'
        while len(getattr(current,direction)) > 0 and getattr(getattr(current,direction)[0],'fragment_identifier') == fragment_identifier: #NOTE maybe add this condition  #and current.fragment_identifier != fragment_identifier :
            current = getattr(current,direction)[0]
            current.set_identity_blob_in_fragment(identity_in_fragment, duplication_solved, P1_vector, frequencies_in_fragment)
            if assigned_during_accumulation:
                current.update_blob_assigned_during_accumulation()


    def update_identity_in_fragment(self, identity_in_fragment, \
                                    assigned_during_accumulation = False, \
                                    duplication_solved = False, \
                                    number_of_images_in_fragment = None):
        if self.is_a_fish_in_a_fragment:
            if assigned_during_accumulation:
                self.update_blob_assigned_during_accumulation()
            self._frequencies_in_fragment = np.zeros(self.number_of_animals)
            if identity_in_fragment != 0:
                self._frequencies_in_fragment[identity_in_fragment-1] = number_of_images_in_fragment ### NOTE Decide whether to use weighted frequencies or raw frequencies
                self._P1_vector = compute_P1_individual_fragment_from_frequencies(self._frequencies_in_fragment)
            elif hasattr(self,'ambiguous_identities') and len(self.ambiguous_identities) != 0:
                self._frequencies_in_fragment[self.ambiguous_identities-1] = number_of_images_in_fragment // len(self.ambiguous_identities)
                self._P1_vector = compute_P1_individual_fragment_from_frequencies(self._frequencies_in_fragment)

            self.set_identity_blob_in_fragment(identity_in_fragment, duplication_solved, self._P1_vector, self._frequencies_in_fragment)
            self.update_identity_in_fragment_in_direction(self, identity_in_fragment,
                                                            assigned_during_accumulation,
                                                            self._P1_vector,
                                                            self._frequencies_in_fragment,
                                                            duplication_solved,
                                                            direction = 'next',
                                                            fragment_identifier = self.fragment_identifier)
            self.update_identity_in_fragment_in_direction(self, identity_in_fragment,
                                                            assigned_during_accumulation,
                                                            self._P1_vector,
                                                            self._frequencies_in_fragment,
                                                            duplication_solved,
                                                            direction = 'previous',
                                                            fragment_identifier = self.fragment_identifier)

    def update_attributes_in_fragment(self, attributes, values):
        assert len(attributes) == len(values)
        [setattr(self, attribute, value) for attribute, value in zip(attributes, values)]
        current = self
        while len(current.next) > 0 and current.next[0].fragment_identifier == self.fragment_identifier:
            current = current.next[0]
            [setattr(current, attribute, value) for attribute, value in zip(attributes, values)]
        current = self
        while len(current.previous) > 0 and current.previous[0].fragment_identifier == self.fragment_identifier:
            current = current.previous[0]
            [setattr(current, attribute, value) for attribute, value in zip(attributes, values)]

    def compute_overlapping_with_previous_blob(self):
        number_of_previous_blobs = len(self.previous)
        if number_of_previous_blobs == 1:
            self.non_shared_information_with_previous = 1. - len(np.intersect1d(self.pixels, self.previous[0].pixels)) / np.mean([len(self.pixels), len(self.previous[0].pixels)])
            if self.non_shared_information_with_previous is np.nan:
                logger.debug("intersection both blobs %s" %str(len(np.intersect1d(self.pixels, self.previous[0].pixels))))
                logger.debug("mean pixels both blobs %s" %str(np.mean([len(self.pixels), len(self.previous[0].pixels)])))
                raise ValueError("non_shared_information_with_previous is nan")

    def apply_model_area(self, video, model_area, portraitSize):
        if model_area(self.area): #Checks if area is compatible with the model area we built
            if video.resolution_reduction == 1:
                height = video._height
                width = video._width
            else:
                height  = int(video._height * video.resolution_reduction)
                width  = int(video._width * video.resolution_reduction)

            if video.preprocessing_type == 'portrait':
                portrait, \
                self._nose_coordinates, \
                self._head_coordinates = get_portrait(self.bounding_box_image,
                                                    self.contour,
                                                    self.bounding_box_in_frame_coordinates,
                                                    portraitSize)
            elif video.preprocessing_type == 'body':
                portrait, \
                self._extreme1_coordinates, \
                self._extreme2_coordinates = get_body(height, width,
                                                    self.bounding_box_image,
                                                    self.pixels,
                                                    self.bounding_box_in_frame_coordinates,
                                                    portraitSize)
            elif video.preprocessing_type == 'body_blob':
                portrait, \
                self._extreme1_coordinates, \
                self._extreme2_coordinates = get_body(height, width,
                                                    self.bounding_box_image,
                                                    self.pixels,
                                                    self.bounding_box_in_frame_coordinates,
                                                    portraitSize, only_blob = True)
            self._portrait = ((portrait - np.mean(portrait))/np.std(portrait)).astype('float32')

def get_images_from_blobs_in_video(blobs_in_video):
    portraits_in_video = []
    fragments_identifier_used = []

    for blobs_in_frame in blobs_in_video:

        for blob in blobs_in_frame:
            if not blob.assigned_during_accumulation and blob.fragment_identifier not in fragments_identifier_used:
                images_in_fragment = blob.portraits_in_fragment()
                if len(images_in_fragment) > 0:
                    portraits_in_video.append(images_in_fragment)
                    fragments_identifier_used.append(blob.fragment_identifier)

    return np.concatenate(portraits_in_video, axis = 0)

def reset_blobs_fragmentation_parameters(blobs_in_video, recovering_from = 'fragmentation'):
    for blobs_in_frame in blobs_in_video:
        for blob in blobs_in_frame:
            blob.reset_before_fragmentation(recovering_from)
