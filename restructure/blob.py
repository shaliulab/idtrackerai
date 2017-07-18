from __future__ import absolute_import, division, print_function
import sys
sys.path.append('./utils')
sys.path.append('./preprocessing')
from get_portraits import getPortrait, get_portrait_fly
import itertools
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

# STD_TOLERANCE = 1 # tolerance to select a blob as being a single fish according to the area model
### NOTE set to 1 because we changed the model area to work with the median.

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
        self._portrait = portrait # (numpy array (uint8),tuple(int,int),tuple(int,int)): (36x36 image of the animal,nose coordinates, head coordinates)
        self.pixels = pixels # list of int's: linearized pixels of the blob
        self.reset_before_fragmentation()

    def reset_before_fragmentation(self):
        self.next = [] # next blob object overlapping in pixels with current blob object
        self.previous = [] # previous blob object overlapping in pixels with the current blob object
        self._fragment_identifier = None # identity in individual fragment after fragmentation
        self._blob_index = None # index of the blob to plot the individual fragments
        self._identity = None # identity assigned by the algorithm
        self._frequencies_in_fragment = np.zeros(self.number_of_animals).astype('int')
        self._P1_vector = np.zeros(self.number_of_animals)
        self._P2_vector = np.zeros(self.number_of_animals)
        self._assigned_during_accumulation = False
        self._user_generated_identity = None #in the validation part users can correct manually the identities

    @property
    def user_generated_identity(self):
        return self._user_generated_identity

    @user_generated_identity.setter
    def user_generated_identity(self, new_identifier):
        #check if the identity is different from the one assigned by the algorithm
        if self._identity != new_identifier:
            self._user_generated_identity = new_identifier

    @property
    def is_a_fish(self):
        return self.portrait is not None

    @property
    def is_a_jump(self):
        return (self.is_a_fish and self.identity == 0 and len(self.next) == 0 and len(self.previous) == 0)

    @property
    def is_a_ghost_crossing(self):
        return (self.is_a_fish and self.identity == 0 and (len(self.next) != 1 or len(self.previous) != 1))

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
        if self.is_a_fish_in_a_fragment:
            self._fragment_identifier = new_fragment_identifier

    @property
    def assigned_during_accumulation(self):
        return self._assigned_during_accumulation

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
        output_along_segment = []
        if self.is_a_fish_in_a_fragment:
            current = self
            while current.next[0].is_a_fish_in_a_fragment:
                output_along_segment.append(function(current, current.next[0]))
                current = current.next[0]
            current = self

            while current.previous[0].is_a_fish_in_a_fragment:
                output_along_segment.append(function(current, current.previous[0]))
                current = current.previous[0]
        return output_along_segment

    def _along_blobs_in_individual_segment(self, function):
        '''Crawls along an individual fragment and outputs a list with
        the result of function applied to each blob
        '''
        output_along_segment = []
        if self.is_a_fish_in_a_fragment:
            current = self
            output_along_segment.append(function(current))
            while current.next[0].is_a_fish_in_a_fragment:
                current = current.next[0]
                output_along_segment.append(function(current))
            current = self

            while current.previous[0].is_a_fish_in_a_fragment:
                current = current.previous[0]
                output_along_segment.append(function(current))
        return output_along_segment


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
        frame_numbers = self._along_blobs_in_individual_segment(frame_number_of_blob)
        return [min(frame_numbers), max(frame_numbers)]

    def portraits_in_fragment(self):
        def return_portrait_blob(blob):
            return blob.portrait
        return self._along_blobs_in_individual_segment(return_portrait_blob)

    def identities_in_fragment(self):
        def blob_identity(blob):
            return blob._identity
        return self._along_blobs_in_individual_segment(blob_identity)

    def get_P1_vectors_coexisting_fragments(self, blobs_in_video):
        P1_vectors = []
        if self.is_a_fish_in_a_fragment:
            fragment_identifiers_of_coexisting_fragments = []
            for b, blob in enumerate(blobs_in_video[self.frame_number]):
                if blob.fragment_identifier is not self.fragment_identifier and \
                        blob.fragment_identifier not in fragment_identifiers_of_coexisting_fragments and \
                        blob.fragment_identifier is not None:
                    P1_vectors.append(blob.P1_vector)
                    fragment_identifiers_of_coexisting_fragments.append(blob.fragment_identifier)

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

    def update_identity_in_fragment(self, identity_in_fragment, assigned_during_accumulation = False):
        if self.is_a_fish_in_a_fragment:
            self._identity = identity_in_fragment
            if assigned_during_accumulation:
                self._assigned_during_accumulation = True
                self._P1_vector[identity_in_fragment-1] = 0.99999999999999
                self._P2_vector[identity_in_fragment-1] = 0.99999999999999
            current = self

            while current.next[0].is_a_fish_in_a_fragment:
                current = current.next[0]
                current._identity = identity_in_fragment
                if assigned_during_accumulation:
                    current._assigned_during_accumulation = True
                    current._P1_vector[identity_in_fragment-1] = 0.99999999999999
                    current._P2_vector[identity_in_fragment-1] = 0.99999999999999
                else:
                    current._P2_vector = self.P2_vector
            current = self

            while current.previous[0].is_a_fish_in_a_fragment:
                current = current.previous[0]
                current._identity = identity_in_fragment
                if assigned_during_accumulation:
                    current._assigned_during_accumulation = True
                    current._P1_vector[identity_in_fragment-1] = 0.99999999999999
                    current._P2_vector[identity_in_fragment-1] = 0.99999999999999
                else:
                    current._P2_vector = self.P2_vector

    def update_P1_in_fragment(self):
        current = self

        while current.next[0].is_a_fish_in_a_fragment:
            current = current.next[0]
            current._P1_vector = self.P1_vector
            current._frequencies_in_fragment = self.frequencies_in_fragment

        current = self

        while current.previous[0].is_a_fish_in_a_fragment:
            current = current.previous[0]
            current._P1_vector = self.P1_vector
            current._frequencies_in_fragment = self.frequencies_in_fragment

# def compute_fragment_identifier_and_blob_index(blobs_in_video, maximum_number_of_blobs):
#     counter = 1
#     for frame in tqdm(blobs_in_video, desc = 'assigning fragment identifier'):
#         for blob in frame:
#             if not blob.is_a_fish_in_a_fragment:
#                 blob.fragment_identifier = -1
#             elif blob.fragment_identifier is None:
#                 blob.fragment_identifier = counter
#                 while len(blob.next) == 1 and blob.next[0].is_a_fish_in_a_fragment:
#                     blob = blob.next[0]
#                     blob.fragment_identifier = counter
#                 counter += 1

def compute_fragment_identifier_and_blob_index(blobs_in_video, maximum_number_of_blobs):
    counter = 1
    possible_blob_indices = range(maximum_number_of_blobs)
    # print("possible_blob_indices ", possible_blob_indices)
    for blobs_in_frame in tqdm(blobs_in_video, desc = 'assigning fragment identifier'):
        used_blob_indices = [blob.blob_index for blob in blobs_in_frame if blob.blob_index is not None]
        # print("used_blob_indices ", used_blob_indices)
        missing_blob_indices =  list(set(possible_blob_indices).difference(set(used_blob_indices)))
        # print("missing_blob_indices ", missing_blob_indices)
        for blob in blobs_in_frame:
            if blob.fragment_identifier is None and blob.is_a_fish_in_a_fragment:
                blob.fragment_identifier = counter
                blob_index = missing_blob_indices.pop(0)
                blob._blob_index = blob_index
                while len(blob.next) == 1 and blob.next[0].is_a_fish_in_a_fragment:
                    blob = blob.next[0]
                    blob.fragment_identifier = counter
                    blob._blob_index = blob_index
                counter += 1

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

def apply_model_area(video, blob, model_area, portraitSize):
    if model_area(blob.area): #Checks if area is compatible with the model area we built
        if video.animal_type == 'fish':
            blob._portrait, blob._nose_coordinates, blob._head_coordinates = getPortrait(blob.bounding_box_image, blob.contour, blob.bounding_box_in_frame_coordinates, portraitSize)
            blob.bounding_box_image = None
        elif video.animal_type == 'fly':
            blob._portrait, blob._extreme1_coordinates, blob._extreme2_coordinates = get_portrait_fly(video, blob.bounding_box_image, blob.pixels, blob.bounding_box_in_frame_coordinates, portraitSize)
            blob.bounding_box_image = None

def apply_model_area_to_blobs_in_frame(video, blobs_in_frame, model_area, portraitSize):
    for blob in blobs_in_frame:
        apply_model_area(video, blob, model_area, portraitSize)

def apply_model_area_to_video(video, blobs_in_video, model_area, portraitSize):
    # Parallel(n_jobs=-1)(delayed(apply_model_area_to_blobs_in_frame)(frame, model_area) for frame in tqdm(blobs_in_video, desc = 'Fragmentation progress'))
    for blobs_in_frame in tqdm(blobs_in_video, desc = 'Fragmentation '):
        apply_model_area_to_blobs_in_frame(video, blobs_in_frame, model_area, portraitSize)

def get_images_from_blobs_in_video(blobs_in_video):
    portraits_in_video = []
    for blobs_in_frame in blobs_in_video:
        for blob in blobs_in_frame:
            if blob.is_a_fish_in_a_fragment and not blob.assigned_during_accumulation:
                portraits_in_video.append(blob.portrait)
    return np.asarray(portraits_in_video)

def reset_blobs_fragmentation_parameters(blobs_in_video):
    for blobs_in_frame in blobs_in_video:
        for blob in blobs_in_frame:
            blob.reset_before_fragmentation()

class ListOfBlobs(object):
    def __init__(self, blobs_in_video = None, path_to_save = None):
        self.blobs_in_video = blobs_in_video
        self.path_to_save = path_to_save

    def generate_cut_points(self, num_chunks):
        n = len(self.blobs_in_video) // num_chunks
        self.cutting_points = np.arange(0,len(self.blobs_in_video),n)

    def cut_in_chunks(self):
        for frame in self.cutting_points:
            self.cut_at_frame(frame)

    def cut_at_frame(self, frame_number):
        for blob in self.blobs_in_video[frame_number-1]:
            blob.next = []
        for blob in self.blobs_in_video[frame_number]:
            blob.previous = []

    def reconnect(self):
        # print("cutting points from reconnect ", self.cutting_points)
        for frame_i in self.cutting_points:
            for (blob_0, blob_1) in itertools.product(self.blobs_in_video[frame_i-1], self.blobs_in_video[frame_i]):
                if blob_0.overlaps_with(blob_1):
                    # print("Trying to reconnect")
                    blob_0.now_points_to(blob_1)

    def save(self):
        """save instance"""
        print("saving blobs list at ", self.path_to_save)
        np.save(self.path_to_save, self)
        self.reconnect()

    @classmethod
    def load(cls, path_to_load_blob_list_file):
        print("loading blobs list from ", path_to_load_blob_list_file)

        list_of_blobs = np.load(path_to_load_blob_list_file).item()
        print("cutting points", list_of_blobs.cutting_points)
        list_of_blobs.reconnect()
        return list_of_blobs
