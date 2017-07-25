from __future__ import absolute_import, print_function, division
import matplotlib
matplotlib.use('TKAgg')
from sklearn import mixture
import sys
sys.path.append('../')
sys.path.append('../preprocessing')
sys.path.append('../utils')
sys.path.append('../network')
import numpy as np
from tqdm import tqdm
import collections
from blob import ListOfBlobs

import matplotlib.pyplot as plt
from py_utils import get_spaced_colors_util
from assigner import assign
from id_CNN import ConvNetwork
from network_params import NetworkParams
from blob import Blob
from video_utils import segmentVideo, filterContoursBySize, getPixelsList, getBoundigBox

"""if the blob was not assigned durign the standard procedure it could be a jump or a crossing
these blobs are blobs that are not in a fragment. We distinguish the two cases in which
they are single fish (blob.is_a_fish == True) or crossings (blob.is_a_fish == False)
"""
VEL_PERCENTILE = 99 #percentile used to compute the jump threshold [for blobs that are fish but are isolated]
COVARIANCE_TYPE = "full" #covariance type in GMM to solve 2 animals crossings

def compute_model_velocity(blobs_in_video, number_of_animals, percentile = VEL_PERCENTILE):
    """computes the 2 * (percentile) of the distribution of velocities of identified fish.
    params
    -----
    blobs_in_video: list of blob objects
        collection of blobs detected in the video.
    number_of_animals int
    percentile int
    -----
    return
    -----
    float
    2* percentile(velocity distribution of identified animals)
    """
    distance_travelled_in_individual_fragments = []
    current_individual_fragment_identifier = -1

    for blobs_in_frame in tqdm( blobs_in_video, desc = "computing velocity model"):

        for blob in blobs_in_frame:
            if blob.is_a_fish_in_a_fragment and current_individual_fragment_identifier != blob.fragment_identifier:
                current_individual_fragment_identifier = blob.fragment_identifier
                distance_travelled_in_individual_fragments.extend(blob.frame_by_frame_velocity())

    # return 2 * np.percentile(distance_travelled_in_individual_fragments, percentile)
    return 2 * np.max(distance_travelled_in_individual_fragments)

def assign_jumps(images, video):
    """Restore the network associated to the model used to assign video.
    parameters
    ------
    images: ndarray (num_images, height, width)
        "images" collection of images to be assigned
    video: object
        "video" object associated to the tracked video. It contains information
        about the video, the animals tracked, and the status of the tracking.
    return
    -----
    assigner: object
        contains predictions (ndarray of shape [number of images]) of the network, the values in the last fully
        conencted layer (ndarray of shape [number of images, 100]), and the values of the softmax layer
        (ndarray of shape [number of images, number of animals in the tracked video])
    """
    assert video._has_been_assigned == True
    net_params = NetworkParams(video.number_of_animals,
                    learning_rate = 0.005,
                    keep_prob = 1.0,
                    use_adam_optimiser = False,
                    restore_folder = video._accumulation_folder,
                    save_folder = video._accumulation_folder,
                    image_size = video.portrait_size)
    net = ConvNetwork(net_params)
    net.restore()
    return assign(net, video, images, print_flag = True)

def compute_velocity_from_list_of_blobs(list_of_blobs):
    centroids = [blob.centroid for blob in list_of_blobs]
    print(centroids)
    velocity = [np.linalg.norm(centroids[i+1] - centroid) for i, centroid in enumerate(centroids[:-1])]
    print(velocity)
    return np.mean(velocity)


class Jump(object):
    def __init__(self, jumping_blob = None, number_of_animals = None, net_prediction = None, softmax_probs = None, velocity_threshold = None, number_of_frames = None):
        self._jumping_blob = jumping_blob
        self.possible_identities = range(1, number_of_animals + 1)
        self.prediction = int(net_prediction)
        self.softmax_probs = softmax_probs
        self.velocity_threshold = velocity_threshold
        self.number_of_frames = number_of_frames

    @property
    def jumping_blob(self):
        return self._jumping_blob

    @jumping_blob.setter
    def jumping_blob(self, jumping_blob):
        """by definition, a jumping blob is a blob which satisfied the model area,
        but does not belong to an individual fragment"""
        assert jumping_blob.is_a_fish
        assert not jumping_blob.is_in_a_fragment
        self._jumping_blob = jumping_blob

    def get_available_identities(self, blobs_in_video):
        blobs_in_frame_sure_identities = [blob.identity for blob in blobs_in_video[self.jumping_blob.frame_number] if blob.identity != 0]
        return set(self.possible_identities) - set(blobs_in_frame_sure_identities)

    def apply_model_velocity(self, blobs_in_video):
        print("checking velocity model for blob ", self.jumping_blob.identity, " in frame ", self.jumping_blob.frame_number)
        blobs_in_frame = blobs_in_video[self.jumping_blob.frame_number - 1]
        corresponding_blob_list = []
        corresponding_blob_list_past = [blob for blob in blobs_in_frame if blob.is_a_fish and blob.identity == self.jumping_blob.identity]
        if corresponding_blob_list_past:
            corresponding_blob_list.append(corresponding_blob_list_past[0])
        corresponding_blob_list.append(self.jumping_blob)
        print("self.jumping_blob.frame_number + 1 ", self.jumping_blob.frame_number + 1)
        print("self.number_of_frames ", self.number_of_frames)
        print("len(blobs_in_video) ", len(blobs_in_video))
        if self.jumping_blob.frame_number + 1 < self.number_of_frames:
            blobs_in_frame = blobs_in_video[self.jumping_blob.frame_number + 1]
            corresponding_blob_list_future = [blob for blob in blobs_in_frame if blob.is_a_fish and blob.identity == self.jumping_blob.identity]
            if corresponding_blob_list_future:
                corresponding_blob_list.append(corresponding_blob_list_future[0])
            print("corresponding_blob_list ", corresponding_blob_list)
        if len(corresponding_blob_list) > 1:
            velocity = compute_velocity_from_list_of_blobs(corresponding_blob_list)
            print("velocity, ", velocity)
            print("velocity_th, ", self.velocity_threshold)
            return velocity < self.velocity_threshold
        else:
            return False

    def check_id_availability(self, available_identities, sorted_assignments_indices):
        return [sorted_assignments_index + 1  for sorted_assignments_index in sorted_assignments_indices
            if (sorted_assignments_index + 1) in available_identities]

    def check_assigned_identity(self, blobs_in_video, available_identities, sorted_assignments_indices):
        if not self.apply_model_velocity(blobs_in_video):
            print("available_identities ", available_identities)
            print("removing ", self.jumping_blob.identity)
            available_identities.remove(self.jumping_blob.identity)
            print("new_available_identities ", available_identities)
            if len(list(available_identities)) > 0:
                self.jumping_blob.identity = self.check_id_availability(available_identities, sorted_assignments_indices)[0]
                print("self.check_id_availability(available_identities, sorted_assignments_indices), ", self.check_id_availability(available_identities, sorted_assignments_indices))
                print("self.jumping_blob.identity, ", self.jumping_blob.identity)
                self.check_assigned_identity(blobs_in_video, available_identities, sorted_assignments_indices)
            else:
                print("no more available_identities")
                self.jumping_blob.identity = 0
        else:
            print("it passes the velocity model")
            print("self.jumping_blob.identity, ", self.jumping_blob.identity)


    def assign_jump(self, blobs_in_video):
        available_identities = self.get_available_identities(blobs_in_video)

        if self.prediction in available_identities:
            self.jumping_blob._identity = self.prediction
        else:
            sorted_assignments_indices = np.argsort(np.array(self.softmax_probs))[::-1]
            new_identity = [sorted_assignments_index for sorted_assignments_index in sorted_assignments_indices
                if (sorted_assignments_index + 1) in available_identities][0]
            self.jumping_blob._identity = new_identity + 1

        if self.jumping_blob.frame_number >= 1:
            sorted_assignments_indices = np.argsort(np.array(self.softmax_probs))[::-1]
            self.check_assigned_identity(blobs_in_video, available_identities, sorted_assignments_indices)

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

class Duplication(object):
    def __init__(self, blobs_in_frame_with_duplication = None, duplicated_identities = None):
        self.blobs_in_frame = blobs_in_frame_with_duplication
        self.identities_to_be_reassigned = duplicated_identities
        #all non duplicated identities in the frame are not available. This list will be
        #updated later on
        self.non_available_identities = [blob.identity for blob in self.blobs_in_frame
                                        if blob.identity not in duplicated_identities]
        self.possible_identities = range(1, self.blobs_in_frame[0].number_of_animals)

    def assign_unique_identities(self):
        all_blobs_to_reassign = []

        for identity in self.identities_to_be_reassigned:
            self.available_identities = list(set(self.possible_identities) - set(self.non_available_identities))
            self.blobs_to_reassign = self.get_blobs_with_same_identity(identity)
            self.assign()
            all_blobs_to_reassign.extend(self.blobs_to_reassign)

        return all_blobs_to_reassign

    def get_blobs_with_same_identity(self, identity):
        """We do not reassign blobs used as references
        """
        return [blob for blob in self.blobs_in_frame
                if blob.identity == identity]

    @staticmethod
    def get_P2_matrix(blobs_list):
        return np.asarray([blob._P2_vector for blob in blobs_list])


    @staticmethod
    def sort_P2_matrix(P2_matrix):
        P2_ids = np.flip(np.argsort(P2_matrix, axis = 1), axis = 1) + 1
        corresponding_P2s = np.flip(np.sort(P2_matrix, axis = 1), axis = 1)
        return np.squeeze(np.asarray(P2_ids.T)), np.squeeze(np.asarray(corresponding_P2s.T))

    def assign(self):
        number_of_blobs_to_reassign = len(self.blobs_to_reassign)
        P2_matrix = self.get_P2_matrix(self.blobs_to_reassign)
        ids, P2s = self.sort_P2_matrix(P2_matrix)
        counter = 0
        assigned_identities = []

        while counter < number_of_blobs_to_reassign:
            for i in range(P2_matrix.shape[0]):
                ids_col = ids[i]
                P2s_col = P2s[i]
                max_P2_in_column = np.argmax(P2s_col)
                candidate_id = ids_col[max_P2_in_column]
                if candidate_id in self.available_identities and np.max(P2s_col) != 0:
                    self.blobs_to_reassign[max_P2_in_column]._identity = candidate_id
                    self.available_identities.remove(candidate_id)
                    counter += 1
                    assigned_identities.append(candidate_id)
                elif np.max(P2s_col) == 0 or i == ids.shape[0]:
                    self.blobs_to_reassign[max_P2_in_column]._identity = 0
            if candidate_id in assigned_identities:
                self.blobs_to_reassign[max_P2_in_column]._identity = 0

class TwoFishCrossing(object):
    def __init__(self, crossing_blob = None):
        self.blob = crossing_blob

    @staticmethod
    def fit_samples(samples, n_components = 2, covariance_type = COVARIANCE_TYPE):
        gmix = mixture.GMM(n_components= n_components, covariance_type = covariance_type)
        gmix.fit(samples)
        available_colors = get_spaced_colors_util(n_components, norm = True, black = False)
        return gmix.predict(samples)

    def clusters2blob_parameters(self, clusters, video):
        pixels = [self.pxs[np.where(clusters == i)[0]] for i in range(self.number_of_clusters)]
        areas = [len(pixel) for pixel in pixels]
        centroids = [np.sum(pixel, axis = 0) / pixel.shape[0] for pixel in pixels]
        centroids = [[cent[1], cent[0]] for cent in centroids]
        pixels =  [np.ravel_multi_index([pixel[:,0], pixel[:,1]],(video._height, video._width)) for pixel in pixels]
        return pixels, areas, centroids

    def get_candidate_blob(self, candidate_blob, previous_blob):
        if type(previous_blob.identity) is not list:
            candidate_blob._identity = int(previous_blob.identity)
            candidate_blob.previous = [previous_blob]
            candidate_blob.next = self.blob.next
            candidate_blob._portrait = 'crossing_solved'
            for next_blob in candidate_blob.next:
                next_blob.previous = [candidate_blob]
            print("____________________________________________")
            print("identity in crossing ", self.blob.identity)
            print("assigned identity ", candidate_blob.identity)
            print("is a crossing ", candidate_blob.is_a_crossing)
            print("____________________________________________")
        else:
            candidate_blob._identity = 0
            candidate_blob._portrait = 'crossing_solved'
        return candidate_blob

    def get_clusters(self, video):
        self.number_of_clusters = len(self.blob.identity)
        self.pxs = np.array(np.unravel_index(self.blob.pixels, (video._height, video._width))).T
        self.clusters = self.fit_samples(self.pxs, n_components = self.number_of_clusters)

    # def compute_assignment_score(self, overlapping):
    #     return np.sort(overlapping)[::-1], np.argsort(overlapping)[::-1]

    @staticmethod
    def naive_assignment(overlapping_percentages):
        return np.argmax(overlapping_percentages, axis = 0)

    def assign_clusters_identity(self, video):
        self.get_clusters(video)
        pixels, areas, centroids = self.clusters2blob_parameters(self.clusters, video)
        available_identities = self.blob.identity
        print("available identities ", available_identities)
        self.blobs_previous_frame = [blob for blob in self.blob.previous]
        overlapping_percentages = []
        candidate_blobs = []

        for i in range(self.number_of_clusters):
            candidate_blob = Blob(centroids[i],
                                self.blob.contour,
                                areas[i],
                                self.blob.bounding_box_in_frame_coordinates,
                                pixels = pixels[i],
                                number_of_animals = video.number_of_animals,
                                frame_number = self.blob.frame_number)

            overlapping = [len(np.intersect1d(candidate_blob.pixels, prev_blob.pixels))/candidate_blob.area for prev_blob in self.blobs_previous_frame]
            overlapping_percentages.append(overlapping)
            candidate_blobs.append(candidate_blob)

        overlapping_percentages = np.asarray(overlapping_percentages)
        print(overlapping_percentages)
        first_assign = self.naive_assignment(overlapping_percentages)
        print(first_assign)
        [available_identities.remove(self.blobs_previous_frame[i].identity) for i in np.unique(first_assign)]


        print(available_identities)

        uncrossed_blobs = [self.get_candidate_blob(candidate_blobs[j], self.blobs_previous_frame[i]) for j,i in enumerate(np.unique(first_assign))]
        if len(available_identities) > 0:
            prev_blob = [b for b in self.blobs_previous_frame is b.identity in available_identities]
            uncrossed_blobs.append(self.get_candidate_blob(candidate_blobs[1], prev_blob))
        return uncrossed_blobs

import cv2
from pprint import pprint
class Crossing(object):
    def __init__(self, crossing_blob, video):
        """Assigns identities to individual in crossing based on simple erosion
        algorithm
        parameters
        -----
        crossing_blob: object
            blob object associated to a crossing.
        video: object
            video object, used to read the height and width of the frame
        return
        -----
        blobs: list
            list of blobs objects (with identity) extracted from the crossing.
            These blobs have P1 and P2 vectors set to zero.
        """
        self.height = video._height
        self.width = video._width
        self.blob = crossing_blob
        self.bounding_box = self.blob.bounding_box_in_frame_coordinates

    def get_binary_image_from_pixels(self):
        pixels = np.array(np.unravel_index(self.blob.pixels, (self.height, self.width))).T
        image = np.zeros((self.height, self.width)).astype('uint8')
        image[pixels[:, 0], pixels[:,1]] = 255
        return image

    @staticmethod
    def erode(image, kernel_size = (3,3)):
        kernel = np.ones(kernel_size, np.uint8)
        return cv2.erode(image,kernel,iterations = 1)

    @staticmethod
    def get_distance_transform(image):
        return cv2.distanceTransform(image,  cv2.cv.CV_DIST_L2,  cv2.cv.CV_DIST_MASK_PRECISE)

    @staticmethod
    def normalize(image):
        return cv2.normalize(image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    @staticmethod
    def get_pixels_from_bw_image(image):
        return np.where(image == 255)[0]

    @staticmethod
    def get_pixels_from_contour(image, contour):
        temp_image = np.zeros_like(image)
        cv2.drawContours(temp_image, [contour], -1, color=255, thickness = -1)
        return np.asarray(np.where(temp_image == 255)).T

    @staticmethod
    def get_n_contours_by_area(contours, number_of_contours):
        contours = np.asarray(contours)
        areas = [cv2.contourArea(contour) for contour in contours]
        selected_contours_indices = np.argsort(areas)[::-1]
        return contours[selected_contours_indices[:number_of_contours]] \
        if len(selected_contours_indices) > number_of_contours \
        else contours

    @staticmethod
    def get_contours(image):
        contours, hierarchy = cv2.findContours(image.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return contours

    def separate_blobs(self):
        print(self.blob.number_of_animals_in_crossing)
        self.image = self.get_binary_image_from_pixels()
        self.image = self.erode(self.image)
        self.image = self.get_distance_transform(self.image)
        self.image = self.normalize(self.image)
        ret, self.image = cv2.threshold(self.image, .5, 1., cv2.cv.CV_THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        self.image = cv2.dilate(self.image, kernel, iterations=1)
        # cv2.imshow('img', self.image)
        # cv2.waitKey()
        contours = self.get_contours(self.image)
        self.contours = self.get_n_contours_by_area(contours, self.blob.number_of_animals_in_crossing)
        self.pixels = [self.get_pixels_from_contour(self.image, contour) for contour in self.contours]
        self.areas = [len(pixel) for pixel in self.pixels]
        self.centroids = [np.sum(pixel, axis = 0) / pixel.shape[0] for pixel in self.pixels]
        self.centroids = [[cent[1], cent[0]] for cent in self.centroids]
        self.pixels = [np.ravel_multi_index([pixel[:,0], pixel[:,1]],(self.height, self.width)) for pixel in self.pixels]
        self.bounding_boxes = [getBoundigBox(contour, self.width, self.height) for contour in self.contours]

    def assign_blobs(self):
        self.next_blobs = self.get_non_crossing_connected_blobs('next')
        self.prev_blobs = self.get_non_crossing_connected_blobs('previous')
        prev_next_blobs = [self.prev_blobs, self.next_blobs]
        #we get the identities in the crossing by considering the next and previous blobs
        self.identities = np.unique([[_blob.identity for _blob in _blobs] for _blobs in prev_next_blobs if _blobs is not None])
        #we compute the overlapping of each blob obtained from the crossing blob with all previous and next blobs
        self.overlapping = np.asarray([np.asarray([self.get_overlapping(pixel, blobs)
                                            for blobs in prev_next_blobs if blobs is not None])
                                            for i,pixel in enumerate(self.pixels)])
        #we consider the mean of previous and next overlapping percentages (if available)
        self.averaged_overlapping = np.mean(self.overlapping, axis = 1)
        print("----- ids and overlappings \n")
        print("identities ",self.identities)
        print("overlapping\n")
        pprint(self.overlapping)
        print("averaged overlapping\n")
        pprint(self.averaged_overlapping)
        if not np.sum(np.isnan(self.averaged_overlapping)):
            self.get_identities_of_crossing_blobs()
            return self.blobs
        else:
            return [self.blob]

    def get_overlapping(self, pixels, blobs):
        overlapping = [self.get_overlapping_percentage(pixels, blob.pixels) for blob in blobs]
        # identities = np.asarray([blob.identity for blob in blobs])
        # return corresponding_identities[np.argsort(overlapping)[::-1]], np.sort(overlapping)[::-1]
        return overlapping#, identities

    def get_non_crossing_connected_blobs(self, attribute = 'None'):
        blobs = [blob for blob in getattr(self.blob, attribute) if not blob.is_a_crossing]
        return blobs if len(blobs) > 0 else None

    def get_overlapping_percentage(self, pixels1, pixels2):
        return len(np.intersect1d(pixels1, pixels2))/len(pixels1)

    def get_identities_of_crossing_blobs(self):
        predictions = self.averaged_overlapping
        identities_index = np.argsort(predictions, axis = 1)
        self.identities = self.identities[identities_index]
        #case 1
        if len(self.identities[:,0]) == self.blob.number_of_animals_in_crossing and len(self.identities[0,:]) == self.blob.number_of_animals_in_crossing:
            self.blobs = [self.init_blob(i, identity) for i, identity in enumerate(self.identities[:,0])]
        else:
            self.blobs = [self.blob]


    def init_blob(self, index, identity):
        print("index ", index)
        print("identity ", identity)
        blob = Blob(self.centroids[index],
                    self.contours[index],
                    self.areas[index],
                    self.bounding_boxes[index],
                    pixels = self.pixels[index],
                    number_of_animals = video.number_of_animals,
                    frame_number = self.blob.frame_number)
        blob._identity = identity
        blob.previous = self.prev_blobs
        blob.next = self.blob.next
        blob._portrait = 'uncrossed blob'
        return blob





if __name__ == "__main__":
    from GUI_utils import frame_by_frame_identity_inspector
    NUM_CHUNKS_BLOB_SAVING = 10

    #load video and list of blobs
    # video = np.load('/home/chronos/Desktop/IdTrackerDeep/videos/8zebrafish_conflicto/session_4/video_object.npy').item()
    video = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/conflict8Short/session_1/video_object.npy').item()
    number_of_animals = video.number_of_animals
    # list_of_blobs_path = '/home/chronos/Desktop/IdTrackerDeep/videos/8zebrafish_conflicto/session_4/preprocessing/blobs_collection.npy'
    list_of_blobs_path = '/home/lab/Desktop/TF_models/IdTrackerDeep/videos/conflict8Short/session_1/preprocessing/blobs_collection_safe.npy'
    list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
    blobs = list_of_blobs.blobs_in_video
    if not hasattr(video, "velocity_threshold"):
        video.velocity_threshold = compute_model_velocity(blobs, number_of_animals, percentile = VEL_PERCENTILE)


    for blobs_in_frame in blobs:
        try:
            print("------------------------frame_number: ", blobs_in_frame[0].frame_number)
        except:
            print("--------")
        identities = [blob.identity for blob in blobs_in_frame if blob.identity != 0]
        print("identities in frame ", identities)
        duplicated_identities = set([x for x in identities if identities.count(x) > 1])
        print("duplicated identities ", duplicated_identities)
        if len(duplicated_identities) > 0:
            frame  = Duplication(blobs_in_frame_with_duplication = blobs_in_frame,
                                duplicated_identities = duplicated_identities)
            blobs_to_reassign = frame.assign_unique_identities()

            for blob in blobs_in_frame:
                for blob_d in blobs_to_reassign:
                    if blob is blob_d and blob.identity != blob_d.identity:
                        blob.update_identity_in_fragment(blob_d.identity)

    ''' Assign identities to jumps '''
    jump_blobs = [blob for blobs_in_frame in blobs for blob in blobs_in_frame
                    if blob.is_a_jump or blob.is_a_ghost_crossing]
    jump_images = [blob.portrait for blob in jump_blobs]
    #assign jumps by restoring the network
    assigner = assign_jumps(jump_images, video)

    for i, blob in enumerate(jump_blobs):
        jump = Jump(jumping_blob = blob,
                    number_of_animals = video.number_of_animals,
                    net_prediction = assigner._predictions[i],
                    softmax_probs = assigner._softmax_probs[i],
                    velocity_threshold = video.velocity_threshold,
                    number_of_frames = video._num_frames)
        jump.assign_jump(blobs)
        blob._identity = jump.jumping_blob.identity
        blob._P1_vector = assigner._softmax_probs[i]
        blob._P2_vector = None

    ''' Find who is in the crossings '''

    crossing_identifier = 0

    for frame_number, blobs_in_frame in enumerate(blobs):
        ''' from past to future '''
        print("---------------frame_number (from past): ", frame_number)
        for blob in blobs_in_frame:
            print('***new blob ')
            if blob.is_a_crossing:
                print('this blob is a crossing')
                blob._identity = list(flatten([previous_blob.identity for previous_blob in blob.previous]))
                blob.bad_crossing = False
                for previous_blob in blob.previous:
                    print("\nprevious_blob: is_a_fish - %i, is_a_crossing - %i" %(previous_blob.is_a_fish, previous_blob.is_a_crossing))
                    print("previous_blob identity: ", previous_blob.identity)
                    if previous_blob.is_a_crossing:
                        print("previous_blob_next_crossings ", [previous_blob_next.is_a_crossing for previous_blob_next in previous_blob.next])
                        for previous_blob_next in previous_blob.next:
                            print('--->', previous_blob_next.identity)
                        previous_has_more_than_one_crossing = sum([previous_blob_next.is_a_crossing for previous_blob_next in previous_blob.next]) > 1
                        print("previous_has_more_than_one_crossing, ", previous_has_more_than_one_crossing)
                        if previous_has_more_than_one_crossing:
                            blob.bad_crossing = True
                        if len(previous_blob.next) != 1: # the previous crossing_blob is splitting
                            for previous_blob_next in previous_blob.next:
                                print("previous_blob_next: is_a_fish - %i, is_a_crossing - %i" %(previous_blob_next.is_a_fish, previous_blob_next.is_a_crossing))
                                print("previous_blob_next identity: ", previous_blob_next.identity)
                                if previous_blob_next is not blob: # for every next of the previous that is not the current blob we remove the identities
                                    print(previous_blob_next.identity)
                                    if previous_blob_next.is_a_fish and previous_blob_next.identity != 0 and previous_blob_next.identity in blob._identity:
                                        blob._identity.remove(previous_blob_next.identity)
                                    else:
                                        print('we do nothing, probably a badly solved jump')

            blob.crossing_identifier = crossing_identifier
            crossing_identifier += 1

            print("blob.identity: ", blob.identity)

    for frame_number, blobs_in_frame in enumerate(blobs[::-1]):
        ''' from future to past '''
        for blob in blobs_in_frame:
            print("\nframe_number (from future): ", blob.frame_number)
            if blob.is_a_crossing:
                print(blob.is_a_crossing)
                has_more_than_one_crossing = sum([blob_previous.is_a_crossing for blob_previous in blob.previous]) > 1
                print("has_more_than_one_crossing, ", has_more_than_one_crossing)
                for blob_previous in blob.previous:
                    if blob_previous.is_a_crossing:
                        print("previous_blob.bad_crossing (before), ", blob_previous.bad_crossing)
                    if blob_previous.is_a_crossing and blob_previous.bad_crossing and has_more_than_one_crossing:
                        blob_previous.bad_crossing = True
                        print("previous_blob.bad_crossing(after), ", blob_previous.bad_crossing)
                blob._identity.extend(list(flatten([next_blob.identity for next_blob in blob.next])))
                blob._identity = list(np.unique(blob._identity))
                for next_blob in blob.next:
                    print("next_blob: is_a_fish - %i, is_a_crossing - %i" %(next_blob.is_a_fish, next_blob.is_a_crossing))
                    print("next_blob identity: ", next_blob.identity)
                    if next_blob.is_a_crossing:
                        if len(next_blob.previous) != 1: # the next crossing_blob is splitting
                            for next_blob_previous in next_blob.previous:
                                print("next_blob_previous: is_a_fish - %i, is_a_crossing - %i" %(next_blob_previous.is_a_fish, next_blob_previous.is_a_crossing))
                                if next_blob_previous is not blob:
                                    print(next_blob_previous.identity)
                                    if next_blob_previous.is_a_fish and next_blob_previous.identity != 0 and next_blob_previous.identity in blob._identity:
                                        blob._identity.remove(next_blob_previous.identity)
                                    elif next_blob_previous.is_a_crossing and not next_blob_previous.bad_crossing:
                                        [blob._identity.remove(identity) for identity in next_blob_previous.identity if identity in blob._identity]
                                    else:
                                        print('we do nothing, probably a badly solved jump')

                identities_to_remove_from_crossing = [blob_to_remove.identity for blob_to_remove in blobs_in_frame if blob_to_remove.is_a_fish]
                identities_to_remove_from_crossing.extend([0])
                [blob._identity.remove(identity) for identity in identities_to_remove_from_crossing if identity in blob._identity]
                if blob.bad_crossing:
                    blob.number_of_animals_in_crossing = None
                else:
                    blob.number_of_animals_in_crossing = len(blob.identity)
            print("blob.identity: ", blob.identity)
            print("frame number ", blob.frame_number)
            if blob.is_a_crossing:
                print("num animals ", blob.number_of_animals_in_crossing)

    #Solving 2-fish crossings
    for blobs_in_frame in blobs:
        for blob in blobs_in_frame:
            if blob.is_a_crossing:
                print(blob.frame_number)
                crossing = Crossing(blob, video)
                crossing.separate_blobs()
                uncrossed_blobs = crossing.assign_blobs()
                blobs_in_frame.remove(blob)
                blobs_in_frame.extend(uncrossed_blobs)
    #         if blob.is_a_crossing and blob.number_of_animals_in_crossing == 2 and blob.frame_number != 0:
    #             print("__________frame number ", blob.frame_number)
    #             crossing = TwoFishCrossing(crossing_blob = blob)
    #             uncrossed_blobs = crossing.assign_clusters_identity(video)
    #             blobs_in_frame.remove(blob)
    #             blobs_in_frame.extend(uncrossed_blobs)
    #
    # blobs_list = ListOfBlobs(blobs_in_video = blobs, path_to_save = video.blobs_path)
    # blobs_list.generate_cut_points(NUM_CHUNKS_BLOB_SAVING)
    # blobs_list.cut_in_chunks()
    # blobs_list.save()
    # frame_by_frame_identity_inspector(video, blobs)
