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
from get_portraits import get_body

import matplotlib.pyplot as plt
import cv2
from pprint import pprint
from py_utils import get_spaced_colors_util
from assigner import assign
from id_CNN import ConvNetwork
from network_params import NetworkParams
from blob import Blob
from video_utils import segmentVideo, filterContoursBySize, getPixelsList, getBoundigBox

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

def get_identities_in_crossing_forward(blob):
    blob._identity = list(flatten([previous_blob.identity for previous_blob in blob.previous]))
    blob.bad_crossing = False

    for previous_blob in blob.previous:
        if previous_blob.is_a_crossing:
            previous_has_more_than_one_crossing = sum([previous_blob_next.is_a_crossing for previous_blob_next in previous_blob.next]) > 1
            if previous_has_more_than_one_crossing:
                blob.bad_crossing = True
            if len(previous_blob.next) != 1: # the previous crossing_blob is splitting

                for previous_blob_next in previous_blob.next:
                    if previous_blob_next is not blob: # for every next of the previous that is not the current blob we remove the identities
                        if previous_blob_next.is_a_fish and previous_blob_next.identity != 0 and previous_blob_next.identity in blob._identity:
                            blob._identity.remove(previous_blob_next.identity)
    return blob

def get_identities_in_crossing_backward(blob, blobs_in_frame):
    has_more_than_one_crossing = sum([blob_previous.is_a_crossing for blob_previous in blob.previous]) > 1

    for blob_previous in blob.previous:
        if blob_previous.is_a_crossing and blob_previous.bad_crossing and has_more_than_one_crossing:
            blob_previous.bad_crossing = True
    blob._identity.extend(list(flatten([next_blob.identity for next_blob in blob.next])))
    blob._identity = list(np.unique(blob._identity))

    for next_blob in blob.next:
        if next_blob.is_a_crossing:
            if len(next_blob.previous) != 1: # the next crossing_blob is splitting
                for next_blob_previous in next_blob.previous:
                    if next_blob_previous is not blob:
                        if next_blob_previous.is_a_fish and next_blob_previous.identity != 0 and next_blob_previous.identity in blob._identity:
                            blob._identity.remove(next_blob_previous.identity)
                        elif next_blob_previous.is_a_crossing and not next_blob_previous.bad_crossing:
                            [blob._identity.remove(identity) for identity in next_blob_previous.identity if identity in blob._identity]

    identities_to_remove_from_crossing = [blob_to_remove.identity for blob_to_remove in blobs_in_frame if blob_to_remove.is_a_fish]
    identities_to_remove_from_crossing.extend([0])
    [blob._identity.remove(identity) for identity in identities_to_remove_from_crossing if identity in blob._identity]
    if blob.bad_crossing:
        blob.number_of_animals_in_crossing = None
    else:
        blob.number_of_animals_in_crossing = len(blob.identity)
    return blob

def give_me_identities_in_crossings(list_of_blobs):
    """Sweep through the video frame by frame to get the identities of individuals in each crossing whenever it is possible
    """
    for frame_number, blobs_in_frame in enumerate(tqdm(list_of_blobs, desc = "getting identities in crossing")):
        ''' from past to future '''
        for blob in blobs_in_frame:
            if blob.is_a_crossing:
                blob = get_identities_in_crossing_forward(blob)

    for frame_number, blobs_in_frame in enumerate(tqdm(blobs[::-1], desc = "getting identities in crossings")):
        ''' from future to past '''
        for blob in blobs_in_frame:
            if blob.is_a_crossing:
                blob = get_identities_in_crossing_backward(blob, blobs_in_frame)

    return list_of_blobs

def assign_crossing_identifier(list_of_blobs):
    """we define a crossing fragment as a crossing that in subsequent frames
    involves the same individuals"""
    crossing_identifier = 0
    # get crossings in a fragment
    for blobs_in_frame in list_of_blobs:
        for blob in blobs_in_frame:
            if blob.is_a_crossing and not hasattr(blob, 'crossing_identifier'):
                crossing_identifier = propagate_crossing_identifier(blob, crossing_identifier)
            else:
                blob.is_a_crossing_in_a_fragment = False
    return crossing_identifier

def propagate_crossing_identifier(blob, crossing_identifier):
    blob.is_a_crossing_in_a_fragment = True
    blob.crossing_identifier = crossing_identifier
    cur_blob = blob

    while len(cur_blob.next) == 1:
        cur_blob.next[0].is_a_crossing_in_a_fragment = True
        cur_blob.next[0].crossing_identifier = crossing_identifier
        cur_blob = cur_blob.next[0]

    cur_blob = blob

    while len(cur_blob.previous) == 1:
        cur_blob.previous[0].is_a_crossing_in_a_fragment = True
        cur_blob.previous[0].crossing_identifier = crossing_identifier
        cur_blob = cur_blob.previous[0]
    return crossing_identifier + 1

def get_crossing_and_statistics(list_of_blobs, max_crossing_identifier):
    number_of_crossing_frames = 0
    crossings = {i: [] for i in range(max_crossing_identifier)}

    for blobs_in_frame in list_of_blobs:
        for blob in blobs_in_frame:
            local_crossing = []
            if blob.is_a_crossing:
                print("frame number ", blob.frame_number)
                number_of_crossing_frames += 1
                crossings[blob.crossing_identifier].append(blob)

    crossings_lengths = [len(crossings[c]) for c in crossings]
    return crossings, len(crossings), number_of_crossing_frames, crossings_lengths

class Crossing(object):
    def __init__(self, crossing_blob, video, image_size):
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
        self.original_image = self.blob.bounding_box_image
        self.video = video
        self.image_size = image_size
        # print("start solving crossing in frame: " , self.blob.frame_number)

    def get_binary_image_from_pixels(self):
        pixels = np.array(np.unravel_index(self.blob.pixels, (self.height, self.width))).T
        image = np.zeros((self.height, self.width)).astype('uint8')
        image[pixels[:, 0], pixels[:,1]] = 255
        return image

    def get_binary_images_from_list_of_pixels(self, list_of_pixels):
        images = []

        for pixels in list_of_pixels:
            image = np.zeros((self.height, self.width)).astype('uint8')
            image[pixels[:, 0], pixels[:,1]] = 255
            images.append(image)

        return images

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
    def filter_contours_by_area(contours, min_area, min_area_ratio = .66):
        contours = np.asarray(contours)
        areas = [cv2.contourArea(contour) for contour in contours]
        return [contour for i, contour in enumerate(contours) if areas[i] > min_area * min_area_ratio]

    @staticmethod
    def get_contours(image):
        contours, hierarchy = cv2.findContours(image.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return contours

    def separate_blobs(self, erosion_kernel_size = (5,5), threshold_distance_transform = .25):
        self.image = self.get_binary_image_from_pixels()
        self.image = self.erode(self.image, kernel_size = erosion_kernel_size)
        self.image = self.get_distance_transform(self.image)
        self.image = self.normalize(self.image)
        ret, self.image = cv2.threshold(self.image, threshold_distance_transform, 1., cv2.cv.CV_THRESH_BINARY)
        contours = self.get_contours(self.image)
        self.contours = self.filter_contours_by_area(contours, self.video.min_area)
        self.pixels = [self.get_pixels_from_contour(self.image, contour) for contour in self.contours]
        images = self.get_binary_images_from_list_of_pixels(self.pixels)
        kernel = np.ones((5,5), np.uint8)
        self.pixels_coordinates = [self.get_pixels_from_contour(image, self.get_contours(cv2.dilate(image, kernel, iterations=1))[0]) for image in images]
        self.areas = [len(pixel) for pixel in self.pixels_coordinates]
        self.centroids = [np.sum(pixel, axis = 0) / pixel.shape[0] for pixel in self.pixels_coordinates]
        self.centroids = [[cent[1], cent[0]] for cent in self.centroids]
        self.pixels_ravelled = [np.ravel_multi_index([pixel[:,0], pixel[:,1]],(self.height, self.width)) for pixel in self.pixels_coordinates]
        self.bounding_boxes = [getBoundigBox(contour, self.width, self.height) for contour in self.contours]
        self.number_of_split_blobs = len(self.pixels)
        self.new_blobs = [self.create_blob_for_individual_crossing_discriminator(centroid, self.areas[i], self.pixels_ravelled[i], self.contours[i], video.number_of_animals, self.blob.frame_number)
            for i, centroid in enumerate(self.centroids)]

    def create_blob_for_individual_crossing_discriminator(self, centroid, area, pixels, contour, number_of_animals, frame_number):
        new_blob = Blob(centroid,
                        contour,
                        area,
                        None,
                        bounding_box_image = None,
                        estimated_body_length = None,
                        pixels = pixels,
                        number_of_animals = number_of_animals,
                        frame_number = frame_number)
        return self.generate_fish_crossing_image(new_blob)

    @staticmethod
    def get_blob_miniframe(blob, video):
        sNumber = video.in_which_episode(blob.frame_number)
        sFrame = blob.frame_number
        if video._paths_to_video_segments:
            cap = cv2.VideoCapture(video._paths_to_video_segments[sNumber])
        else:
            cap = cv2.VideoCapture(video.video_path)
        #Get frame from video file
        if video._paths_to_video_segments:
            start = video._episodes_start_end[sNumber][0]
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,sFrame - start)
        else:
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,blob.frame_number)
        ret, frame = cap.read()
        if ret:
            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bounding_box, estimated_body_length = getBoundigBox(blob.contour, video._width, video._height, crossing_detector = True)
            return frameGray[bounding_box[0][1]:bounding_box[1][1], bounding_box[0][0]:bounding_box[1][0]], bounding_box, estimated_body_length
        else:
            return None, None, None

    def generate_fish_crossing_image(self, blob):
        mini_frame, bounding_box, estimated_body_length = self.get_blob_miniframe(blob,self.video)
        blob.bounding_box_image = mini_frame
        blob.bounding_box_in_frame_coordinates = bounding_box
        blob.estimated_body_length = estimated_body_length
        if mini_frame is not None:
            image, _, _ = get_body(self.height, self.width, blob.bounding_box_image, blob.pixels, blob.bounding_box_in_frame_coordinates, self.image_size , only_blob = True)
            image = ((image - np.mean(image))/np.std(image)).astype('float32')
            blob._portrait = image
            return blob
        else:
            return None

    @staticmethod
    def get_images_from_list_of_blobs(list_of_blobs):
        return [getattr(blob, 'portrait') for blob in list_of_blobs if blob is not None]

    def generate_crossing_images(self, assigned_identities = []):
        self.separate_blobs()
        self.images = self.get_images_from_list_of_blobs(self.new_blobs)







if __name__ == "__main__":
    from GUI_utils import frame_by_frame_identity_inspector
    NUM_CHUNKS_BLOB_SAVING = 10

    #load video and list of blobs
    video = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/8zebrafish_conflicto/session_body_blob/video_object.npy').item()
    # video = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/Cafeina5pecesLarge/session_1/video_object.npy').item()
    number_of_animals = video.number_of_animals
    list_of_blobs_path = '/home/lab/Desktop/TF_models/IdTrackerDeep/videos/8zebrafish_conflicto/session_body_blob/preprocessing/blobs_collection.npy'
    # list_of_blobs_path = '/home/lab/Desktop/TF_models/IdTrackerDeep/videos/Cafeina5pecesLarge/session_1/preprocessing/blobs_collection.npy'
    list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
    blobs = list_of_blobs.blobs_in_video
    blobs = give_me_identities_in_crossings(blobs)
    max_crossing_identifier = assign_crossing_identifier(blobs)
    crossings, number_of_crossings, number_of_crossing_frames, crossing_lengths = get_crossing_and_statistics(blobs, max_crossing_identifier)

    crossing_blobs = []
    images_from_crossings = []

    for unsolved_crossing in tqdm(crossings.values(), desc = "Solving crossings"):
        for blob in unsolved_crossing:
            crossing = Crossing(blob, video, 294)
            crossing.generate_crossing_images()
            crossing_blobs.append(crossing)
            images_from_crossings.extend(crossing.images)

    # discriminate_crossing_and_fish_images()
    #
    # assign_fish_images()


    # for unsolved_crossing in tqdm(crossings.values(), desc = "Solving crossings"):
    #     print("***************************************************************************")
    #     print("length of unsolved_crossing before forward loop ", len(unsolved_crossing))
    #     for blob in unsolved_crossing:
    #         if blob.frame_number != 0:
    #             print(">>>>>>>> frame number ", blob.frame_number)
    #             print(">>>>>>>> number of inds in crossing ", blob.number_of_animals_in_crossing)
    #             crossing = Crossing(blob, video)
    #             crossing.separate_blobs()
    #             uncrossed_blobs = crossing.assign_blobs(next_or_previous_attribute = 'previous')
    #             if uncrossed_blobs is not None:
    #                 for _blob in blob.next:
    #                     _blob.previous.remove(blob)
    #                     _blob.previous.extend(uncrossed_blobs)
    #                 print("identities of uncrossed blobs ", [b.identity for b in uncrossed_blobs])
    #                 blobs[blob.frame_number].remove(blob)
    #                 blobs[blob.frame_number].extend(uncrossed_blobs)
    #                 print("____________________________________________________________________________")
    #                 print("removing .. ", len(unsolved_crossing))
    #                 unsolved_crossing.remove(blob)
    #                 print("removed .. ", len(unsolved_crossing))
    #                 print("new identities in frame ", [b.identity for b in blobs[blob.frame_number]])
    #             else:
    #                 print("no blobs found")
    #     print("***************************************************************************")
    #     print("length of unsolved_crossing before backward loop ", len(unsolved_crossing))
    #
    # for unsolved_crossing in tqdm(crossings.values(), desc = "Solving crossings"):
    #     for blob in unsolved_crossing[::-1]:
    #         print("blob next ", blob)
    #         if blob.frame_number != video._num_frames:
    #             crossing = Crossing(blob, video)
    #             crossing.separate_blobs()
    #             uncrossed_blobs = crossing.assign_blobs(next_or_previous_attribute = 'next')
    #             if uncrossed_blobs is not None:
    #                 for _blob in blob.previous:
    #                     if blob in _blob.next:
    #                         _blob.next.remove(blob)
    #                         _blob.next.extend(uncrossed_blobs)
    #                 print("identities of uncrossed blobs ", [b.identity for b in uncrossed_blobs])
    #                 blobs[blob.frame_number].remove(blob)
    #                 blobs[blob.frame_number].extend(uncrossed_blobs)
    #                 print("new identities in frame ", [b.identity for b in blobs[blob.frame_number]])
    #             else:
    #                 print("no blobs found")


    # blobs_list = ListOfBlobs(blobs_in_video = blobs, path_to_save = video.blobs_path)
    # blobs_list.generate_cut_points(NUM_CHUNKS_BLOB_SAVING)
    # blobs_list.cut_in_chunks()
    # blobs_list.save()
    # frame_by_frame_identity_inspector(video, blobs)
