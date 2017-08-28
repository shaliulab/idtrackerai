from __future__ import absolute_import, division, print_function
import sys
sys.path.append('./utils')
sys.path.append('./preprocessing')
from get_portraits import get_portrait, get_body
import itertools
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import logging

from statistics_for_assignment import compute_P1_individual_fragment_from_frequencies

# STD_TOLERANCE = 1 # tolerance to select a blob as being a single fish according to the area model
### NOTE set to 1 because we changed the model area to work with the median.
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
        if recovering_from == 'accumulation' or recovering_from == 'fragmentation':
            self._frequencies_in_fragment = np.zeros(self.number_of_animals).astype('int')
            self._P1_vector = np.zeros(self.number_of_animals)
            self._P2_vector = np.zeros(self.number_of_animals)
            self._assigned_during_accumulation = False
            self._user_generated_identity = None #in the validation part users can correct manually the identities
            self._identity = None
            self._identity_corrected_solving_duplication = None
            self._user_generated_centroids = []
            self._user_generated_identities = []
        if recovering_from == 'fragmentation':
            self.next = [] # next blob object overlapping in pixels with current blob object
            self.previous = [] # previous blob object overlapping in pixels with the current blob object
            self._fragment_identifier = None # identity in individual fragment after fragmentation
            self._blob_index = None # index of the blob to plot the individual fragments
            self._identity = None # identity assigned by the algorithm
            self._frequencies_in_fragment = np.zeros(self.number_of_animals).astype('int')
            self._P1_vector = np.zeros(self.number_of_animals)
            self._P2_vector = np.zeros(self.number_of_animals)
            self._assigned_during_accumulation = False
            self.fragment_identifier = None
            self._blob_index = None
            self.non_shared_information_with_previous = None
        if recovering_from == 'assignment':
            if not self.assigned_during_accumulation:
                self._identity = None
                self._frequencies_in_fragment = np.zeros(self.number_of_animals).astype('int')
                self._P1_vector = np.zeros(self.number_of_animals)
                self._P2_vector = np.zeros(self.number_of_animals)
            else:
                self._identity = int(np.argmax(self._P1_vector)) + 1
            if self._user_generated_identity is not None:
                self._user_generated_identity = None #in the validation part users can correct manually the identities
                self._user_generated_centroids = []
                self._user_generated_identities = []
        self._identity_corrected_solving_duplication = None
        self._is_a_duplication = False

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
    def fragment_identifier(self):
        return self._fragment_identifier

    @fragment_identifier.setter
    def fragment_identifier(self, new_fragment_identifier):
        if self.is_a_fish: #we only check if it is a fish because we also set fragments identifiers to the extremes of an individual fragment and to jumps and jumping fragments
            self._fragment_identifier = new_fragment_identifier

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
            while current.next[0].is_a_fish_in_a_fragment:
                output_along_fragment.append(function(current, current.next[0]))
                current = current.next[0]
            if len(current.next) == 1 and len(current.next[0].previous) == 1 and current.next[0].is_a_fish:
                output_along_fragment.append(function(current, current.next[0]))

            current = self

            while current.previous[0].is_a_fish_in_a_fragment:
                output_along_fragment.append(function(current, current.previous[0]))
                current = current.previous[0]
            if len(current.previous) == 1 and len(current.previous[0].next) == 1 and current.previous[0].is_a_fish:
                output_along_fragment.append(function(current, current.previous[0]))

        return output_along_fragment

    def _along_blobs_in_individual_fragment(self, function):
        '''Crawls along an individual fragment and outputs a list with
        the result of function applied to each blob
        '''
        output_along_fragment = []
        if self.is_a_fish_in_a_fragment:

            current = self
            output_along_fragment.append(function(current))
            while current.next[0].is_a_fish_in_a_fragment:
                current = current.next[0]
                output_along_fragment.append(function(current))
            if len(current.next) == 1 and len(current.next[0].previous) == 1 and current.next[0].is_a_fish:
                output_along_fragment.append(function(current.next[0]))

            current = self
            while current.previous[0].is_a_fish_in_a_fragment:
                current = current.previous[0]
                output_along_fragment.append(function(current))
            if len(current.previous) == 1 and len(current.previous[0].next) == 1 and current.previous[0].is_a_fish:
                output_along_fragment.append(function(current.previous[0]))

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

    def get_P1_vectors_coexisting_fragments(self, blobs_in_video):
        P1_vectors = []
        # if self.is_a_fish_in_a_fragment:
        fragment_identifiers_of_coexisting_fragments = []
        for b, blob in enumerate(blobs_in_video[self.frame_number]):
            if blob.fragment_identifier is not self.fragment_identifier and \
                    blob.fragment_identifier not in fragment_identifiers_of_coexisting_fragments and \
                    blob.fragment_identifier is not None:
                P1_vectors.append(blob.P1_vector)
                fragment_identifiers_of_coexisting_fragments.append(blob.fragment_identifier)

        if self.is_a_fish_in_a_fragment:
            current = self

            while current.next[0].is_a_fish_in_a_fragment:
                current = current.next[0]
                for b, blob in enumerate(blobs_in_video[current.frame_number]):
                    if blob.fragment_identifier is not current.fragment_identifier and \
                            blob.fragment_identifier not in fragment_identifiers_of_coexisting_fragments and \
                            blob.fragment_identifier is not None:
                        P1_vectors.append(blob.P1_vector)
                        fragment_identifiers_of_coexisting_fragments.append(blob.fragment_identifier)

            current = self

            while current.previous[0].is_a_fish_in_a_fragment:
                current = current.previous[0]
                for blob in blobs_in_video[current.frame_number]:
                    if blob.fragment_identifier is not current.fragment_identifier and \
                            blob.fragment_identifier not in fragment_identifiers_of_coexisting_fragments and \
                            blob.fragment_identifier is not None:
                        P1_vectors.append(blob.P1_vector)
                        fragment_identifiers_of_coexisting_fragments.append(blob.fragment_identifier)
        return np.asarray(P1_vectors)

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

            while current.next[0].is_a_fish_in_a_fragment:
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
            while current.previous[0].is_a_fish_in_a_fragment:
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
        while getattr(getattr(current,direction)[0],'is_a_fish_in_a_fragment'): #NOTE maybe add this condition  #and current.fragment_identifier != fragment_identifier :
            current = getattr(current,direction)[0]
            current.set_identity_blob_in_fragment(identity_in_fragment, duplication_solved, P1_vector, frequencies_in_fragment)
            if assigned_during_accumulation:
                current.update_blob_assigned_during_accumulation()

        if len(getattr(current,direction)) == 1 and len(getattr(getattr(current,direction)[0],opposite_direction)) == 1 and getattr(current,direction)[0].is_a_fish:
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
                print("frame_number:", self.frame_number)
                print("ambiguous_identities: ", self.ambiguous_identities)
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
        if len(current.next) > 0:
            while current.next[0].is_a_fish_in_a_fragment:
                current = current.next[0]
                [setattr(current, attribute, value) for attribute, value in zip(attributes, values)]
            if len(current.next) == 1 and len(current.next[0].previous) == 1 and current.next[0].is_a_fish:
                current = current.next[0]
                [setattr(current, attribute, value) for attribute, value in zip(attributes, values)]

        current = self
        if len(current.previous) > 0:
            while current.previous[0].is_a_fish_in_a_fragment:
                current = current.previous[0]
                [setattr(current, attribute, value) for attribute, value in zip(attributes, values)]
            if len(current.previous) == 1 and len(current.previous[0].next) == 1 and current.previous[0].is_a_fish:
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

def compute_fragment_identifier_and_blob_index(blobs_in_video, maximum_number_of_blobs):
    counter = 1
    possible_blob_indices = range(maximum_number_of_blobs)

    for blobs_in_frame in tqdm(blobs_in_video, desc = 'assigning fragment identifier'):
        used_blob_indices = [blob.blob_index for blob in blobs_in_frame if blob.blob_index is not None]
        missing_blob_indices =  list(set(possible_blob_indices).difference(set(used_blob_indices)))
        for blob in blobs_in_frame:
            if blob.fragment_identifier is None and blob.is_a_fish:
                blob.fragment_identifier = counter
                blob_index = missing_blob_indices.pop(0)
                blob._blob_index = blob_index
                blob.non_shared_information_with_previous = 1.

                if len(blob.next) == 1 and len(blob.next[0].previous) == 1 and blob.next[0].is_a_fish:
                    blob.next[0].fragment_identifier = counter
                    blob.next[0]._blob_index = blob_index
                    blob.next[0].compute_overlapping_with_previous_blob()

                    if blob.next[0].is_a_fish_in_a_fragment:
                        blob = blob.next[0]

                        while len(blob.next) == 1 and blob.next[0].is_a_fish_in_a_fragment:
                            blob = blob.next[0]
                            blob.fragment_identifier = counter
                            blob._blob_index = blob_index
                            # compute_overlapping_with_previous_blob
                            blob.compute_overlapping_with_previous_blob()

                        if len(blob.next) == 1 and len(blob.next[0].previous) == 1 and blob.next[0].is_a_fish:
                            blob.next[0].fragment_identifier = counter
                            blob.next[0]._blob_index = blob_index
                            blob.next[0].compute_overlapping_with_previous_blob()
                        # elif blob.next[0].is_a_ghost_crossing:
                        #     blob.next[0].fragment_identifier = counter
                        #     blob.next[0]._blob_index = blob_index
                        #     blob.next[0].compute_overlapping_with_previous_blob()

                counter += 1

def compute_crossing_fragment_identifier(list_of_blobs):
    """we define a crossing fragment as a crossing that in subsequent frames
    involves the same individuals"""
    crossing_identifier = 0

    for blobs_in_frame in list_of_blobs:
        for blob in blobs_in_frame:
            if blob.is_a_crossing and not hasattr(blob, 'crossing_identifier'):
                propagate_crossing_identifier(blob, crossing_identifier)
                crossing_identifier += 1
            elif blob.is_a_crossing and hasattr(blob, 'crossing_identifier'):
                blob.is_a_crossing_in_a_fragment = True
            else:
                blob.is_a_crossing_in_a_fragment = None

def propagate_crossing_identifier(blob, crossing_identifier):
    blob.is_a_crossing_in_a_fragment = True
    blob.crossing_identifier = crossing_identifier
    cur_blob = blob

    while len(cur_blob.next) == 1:
        cur_blob = cur_blob.next[0]
        cur_blob.crossing_identifier = crossing_identifier

    cur_blob = blob

    while len(cur_blob.previous) == 1:
        cur_blob = cur_blob.previous[0]
        cur_blob.crossing_identifier = crossing_identifier

def connect_blob_list(blobs_in_video):
    for frame_i in tqdm(xrange(1,len(blobs_in_video)), desc = 'Connecting blobs '):
        set_frame_number_to_blobs_in_frame(blobs_in_video[frame_i-1], frame_i-1)
        for (blob_0, blob_1) in itertools.product(blobs_in_video[frame_i-1], blobs_in_video[frame_i]):
            if blob_0.overlaps_with(blob_1):
                blob_0.now_points_to(blob_1)
    set_frame_number_to_blobs_in_frame(blobs_in_video[frame_i], frame_i)

def set_frame_number_to_blobs_in_frame(blobs_in_frame, frame_number):
    for blob in blobs_in_frame:
        blob.frame_number = frame_number

def all_blobs_in_a_fragment(blobs_in_frame):
    return all([blob.is_in_a_fragment for blob in blobs_in_frame])

def is_a_global_fragment(blobs_in_frame, num_animals):
    """Returns True iff:
    * number of blobs equals num_animals
    """
    return len(blobs_in_frame) == num_animals

def check_global_fragments(blobs_in_video, num_animals):
    """Returns an array with True iff:
    * each blob has a unique blob intersecting in the past and future
    * number of blobs equals num_animals
    """
    return [all_blobs_in_a_fragment(blobs_in_frame) and len(blobs_in_frame) == num_animals for blobs_in_frame in blobs_in_video]

def compute_portrait_size(video, maximum_body_length):
    if video.preprocessing_type == 'portrait':
        portrait_size = int(maximum_body_length/2)
        portrait_size =  portrait_size + portrait_size%2 #this is to make the portrait_size even
        video.portrait_size = (portrait_size, portrait_size, 1)
    elif video.preprocessing_type == 'body' or video.preprocessing_type == 'body_blob':
        portrait_size = int(np.sqrt(maximum_body_length ** 2 / 2))
        portrait_size = portrait_size + portrait_size%2  #this is to make the portrait_size
        video.portrait_size = (portrait_size, portrait_size, 1)

def apply_model_area(video, blob, model_area, portraitSize):
    if model_area(blob.area): #Checks if area is compatible with the model area we built
        if video.resolution_reduction == 1:
            height = video._height
            width = video._width
        else:
            height  = int(video._height * video.resolution_reduction)
            width  = int(video._width * video.resolution_reduction)

        if video.preprocessing_type == 'portrait':
            portrait, blob._nose_coordinates, blob._head_coordinates = get_portrait(blob.bounding_box_image, blob.contour, blob.bounding_box_in_frame_coordinates, portraitSize)
            blob._portrait = ((portrait - np.mean(portrait))/np.std(portrait)).astype('float32')
            # if not blob.in_a_global_fragment_core:
            #     blob.bounding_box_image = None
        elif video.preprocessing_type == 'body':
            portrait, blob._extreme1_coordinates, blob._extreme2_coordinates = get_body(height, width, blob.bounding_box_image, blob.pixels, blob.bounding_box_in_frame_coordinates, portraitSize)
            blob._portrait = ((portrait - np.mean(portrait))/np.std(portrait)).astype('float32')
            # if not blob.in_a_global_fragment_core:
            #     blob.bounding_box_image = None
        elif video.preprocessing_type == 'body_blob':
            portrait, blob._extreme1_coordinates, blob._extreme2_coordinates = get_body(height, width, blob.bounding_box_image, blob.pixels, blob.bounding_box_in_frame_coordinates, portraitSize, only_blob = True)
            blob._portrait = ((portrait - np.mean(portrait))/np.std(portrait)).astype('float32')
            # if not blob.in_a_global_fragment_core:
            #     blob.bounding_box_image = None

def apply_model_area_to_blobs_in_frame(video, blobs_in_frame, model_area, portraitSize):
    for blob in blobs_in_frame:
        apply_model_area(video, blob, model_area, portraitSize)

def apply_model_area_to_video(video, blobs_in_video, model_area, portraitSize):
    # Parallel(n_jobs=-1)(delayed(apply_model_area_to_blobs_in_frame)(frame, model_area) for frame in tqdm(blobs_in_video, desc = 'Fragmentation progress'))
    for blobs_in_frame in tqdm(blobs_in_video, desc = 'Applying model area'):
        apply_model_area_to_blobs_in_frame(video, blobs_in_frame, model_area, portraitSize)

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

def check_number_of_blobs(video, blobs):
    frames_with_more_blobs_than_animals = []
    for frame_number, blobs_in_frame in enumerate(blobs):

        if len(blobs_in_frame) > video.number_of_animals:
            frames_with_more_blobs_than_animals.append(frame_number)

    if len(frames_with_more_blobs_than_animals) > 0:
        logger.error('There are frames with more blobs than animals, this can be detrimental for the proper functioning of the system.')
        logger.error("Frames with more blobs than animals: %s" %str(frames_with_more_blobs_than_animals))
        raise ValueError('Please check your segmentaion')
    return frames_with_more_blobs_than_animals

class ListOfBlobs(object):
    def __init__(self, blobs_in_video = None, path_to_save = None):
        self.blobs_in_video = blobs_in_video
        self.path_to_save = path_to_save

    def generate_cut_points(self, num_chunks):
        n = len(self.blobs_in_video) // num_chunks
        self.cutting_points = np.arange(0,len(self.blobs_in_video),n)[1:]

    def cut_in_chunks(self):
        for frame in self.cutting_points:
            self.cut_at_frame(frame)

    def cut_at_frame(self, frame_number):
        for blob in self.blobs_in_video[frame_number-1]:
            blob.next = []
        for blob in self.blobs_in_video[frame_number]:
            blob.previous = []

    def reconnect(self):
        logger.info("Reconnecting list of blob objects")
        for frame_i in self.cutting_points:
            for (blob_0, blob_1) in itertools.product(self.blobs_in_video[frame_i-1], self.blobs_in_video[frame_i]):
                if blob_0.overlaps_with(blob_1):
                    blob_0.now_points_to(blob_1)

    def save(self):
        """save instance"""
        logger.info("saving blobs list at %s" %self.path_to_save)
        np.save(self.path_to_save, self)
        self.reconnect()

    @classmethod
    def load(cls, path_to_load_blob_list_file):
        logger.info("loading blobs list from %s" %path_to_load_blob_list_file)

        list_of_blobs = np.load(path_to_load_blob_list_file).item()
        logging.debug("cutting points %s" %list_of_blobs.cutting_points)
        list_of_blobs.reconnect()
        return list_of_blobs
