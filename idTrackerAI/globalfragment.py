from __future__ import absolute_import, division, print_function
import os
import numpy as np
import random
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import logging

from blob import is_a_global_fragment, check_global_fragments

STD_TOLERANCE = 4 ### NOTE set to 1 because we changed the model area to work with the median.
logger = logging.getLogger("__main__.globalfragment")

def detect_beginnings(boolean_array):
    """ detects the frame where the core of a global fragment starts.
    A core of a global fragment is the part of the global fragment where all the
    individuals are visible, i.e. the number of animals in the frame equals the
    number of animals in the video
    :boolean_array: array with True where the number of animals in the frame equals
    the number of animals in the video
    """
    return [i for i in range(0,len(boolean_array)) if (boolean_array[i] and not boolean_array[i-1])]

def compute_model_area_and_body_length(blobs_in_video, number_of_animals, std_tolerance = STD_TOLERANCE):
    """computes the median and standard deviation of all the blobs of the video
    and the median_body_length estimated from the diagonal of the bounding box.
    These values are later used to discard blobs that are not fish and potentially
    belong to a crossing.
    """
    #areas are collected only in global fragments' cores
    areas_and_body_length = np.asarray([(blob.area,blob.estimated_body_length) for blobs_in_frame in blobs_in_video
                                                                                for blob in blobs_in_frame
                                                                                if len(blobs_in_frame) == number_of_animals])
    #areas are collected throughout the entire video
    median_area = np.median(areas_and_body_length[:,0])
    mean_area = np.mean(areas_and_body_length[:,0])
    std_area = np.std(areas_and_body_length[:,0])
    median_body_length = np.median(areas_and_body_length[:,1])
    return ModelArea(mean_area, median_area, std_area), median_body_length

class ModelArea(object):
  def __init__(self, mean, median, std):
    self.median = median
    self.mean = mean
    self.std = std

  def __call__(self, area, std_tolerance = STD_TOLERANCE):
    return (area - self.median) < std_tolerance * self.std

class GlobalFragment(object):
    def __init__(self, list_of_blobs, list_of_fragments, index_beginning_of_fragment, number_of_animals):
        self.index_beginning_of_fragment = index_beginning_of_fragment
        self.individual_fragments_identifiers = [blob.fragment_identifier for blob in list_of_blobs[index_beginning_of_fragment]]
        self.get_list_of_attributes_from_individual_fragments(list_of_fragments)
        self.total_number_of_images = sum(self.number_of_images_per_individual_fragment)
        self.set_minimum_distance_travelled()
        self.number_of_animals = number_of_animals
        self.reset_accumulation_params()
        self._is_unique = False
        self._is_certain = False

    def reset_accumulation_params(self):
        self._used_for_training = False
        self._acceptable_for_training = True
        self._ids_assigned = np.nan * np.ones(self.number_of_animals)
        self._temporary_ids = np.arange(self.number_of_animals) # I initialize the _temporary_ids like this so that I can use the same function to extract images in pretraining and training
        self._score = None
        self._is_unique = False
        self._uniqueness_score = None
        self._repeated_ids = []
        self._missing_ids = []
        self.predictions = [] #stores predictions per portrait in self, organised according to individual fragments.
        self.softmax_probs_median = [] #stores softmax median per individual, per individual fragment

    def get_list_of_attributes_from_individual_fragments(self,list_of_fragments, list_of_attributes = ['distance_travelled', 'number_of_images']):
        [setattr(self,attribute + '_per_individual_fragment',[]) for attribute in list_of_attributes]
        for fragment in list_of_fragments:
            if fragment.identifier in self.individual_fragments_identifiers:
                for attribute in list_of_attributes:
                    getattr(self, attribute + '_per_individual_fragment').append(getattr(fragment, attribute))


    def set_minimum_distance_travelled(self):
        self.minimum_distance_travelled = min(self.distance_travelled_per_individual_fragment)

    @property
    def used_for_training(self):
        return self._used_for_training

    @property
    def acceptable_for_training(self):
        return self._acceptable_for_training

    @property
    def uniqueness_score(self):
        return self._uniqueness_score

    def compute_uniqueness_score(self, P1_individual_fragments):
        """ Computes the distance of the assignation probabilities (P2) per
        individual fragment in the global fragment to the identity matrix of
        dimension number of animals.
        uniqueness_score = 0.0 means that every individual is assigned with
        certainty 1.0 once and only once in the global fragment
        """
        if not self._used_for_training and self.is_unique:
            identity = np.identity(self.number_of_animals)
            P1_mat = np.vstack(P1_individual_fragments) # stack P1 of each individual fragment in the global fragment into a matrix
            perm = np.argmax(P1_mat,axis=1) # get permutation that orders the matrix to match the identity matrix
            P1_mat = P1_mat[:,perm] # apply permutation
            print(P1_mat)
            self._uniqueness_score = np.linalg.norm(P1_mat - identity)

    @property
    def score(self):
        return self._score

    def compute_score(self, P1_individual_fragments, max_distance_travelled):
        if not self._used_for_training and self.is_unique:
            self.compute_uniqueness_score(P1_individual_fragments)
            self._score = self.uniqueness_score**2 + (max_distance_travelled - self.min_distance_travelled)**2

    @property
    def is_unique(self):
        self.check_uniqueness()
        return self._is_unique

    def check_uniqueness(self):
        all_identities = range(self.number_of_animals)
        if len(set(all_identities) - set(self._temporary_ids)) > 0:
            self._is_unique = False
            self.compute_repeated_and_missing_ids(all_identities)
        else:
            self._is_unique = True

    def compute_repeated_and_missing_ids(self, all_identities):
        self._repeated_ids = set([x for x in self._ids_assigned if list(self._ids_assigned).count(x) > 1])
        self._missing_ids = set(all_identities).difference(set(self._ids_assigned))

    def compute_start_end_frame_indices_of_individual_fragments(self, blobs_in_video):
        self.starts_ends_individual_fragments = [blob.compute_fragment_start_end()
            for blob in blobs_in_video[self.index_beginning_of_fragment]]

def order_global_fragments_by_distance_travelled(global_fragments):
    global_fragments = sorted(global_fragments, key = lambda x: x.min_distance_travelled, reverse = True)
    return global_fragments

def order_global_fragments_by_distance_to_the_first_global_fragment(global_fragments):
    index_beginning_of_first_global_fragment = order_global_fragments_by_distance_travelled(global_fragments)[0].index_beginning_of_fragment
    global_fragments = sorted(global_fragments, key = lambda x: np.abs(x.index_beginning_of_fragment - index_beginning_of_first_global_fragment), reverse = False)
    return global_fragments

def give_me_number_of_unique_images_in_global_fragments(global_fragments):
    individual_fragments_used = []
    number_of_images_in_global_fragments = 0

    for global_fragment in global_fragments:

        for i, individual_fragment_identifier in enumerate(global_fragment.individual_fragments_identifiers):

            if individual_fragment_identifier not in individual_fragments_used:
                number_of_images_in_global_fragments += global_fragment.number_of_images_per_individual_fragment[i]
                individual_fragments_used.append(individual_fragment_identifier)

    return number_of_images_in_global_fragments


def give_me_identities_of_global_fragment(global_fragment, blobs_in_video):
    global_fragment._ids_assigned = [blob.identity
        for blob in blobs_in_video[global_fragment.index_beginning_of_fragment]]

def give_me_list_of_global_fragments(blobs_in_video, list_of_fragments, num_animals):
    global_fragments_boolean_array = check_global_fragments(blobs_in_video, num_animals)
    indices_beginning_of_fragment = detect_beginnings(global_fragments_boolean_array)
    return [GlobalFragment(blobs_in_video, list_of_fragments, i, num_animals) for i in indices_beginning_of_fragment]

def filter_global_fragments_by_minimum_number_of_frames(global_fragments,minimum_number_of_frames = 3):
    return [global_fragment for global_fragment in global_fragments
                if np.min(global_fragment.number_of_images_per_individual_fragment) >= minimum_number_of_frames]

def give_me_pre_training_global_fragments(global_fragments, number_of_pretraining_global_fragments = 10):
    indices = np.round(np.linspace(0, len(global_fragments), number_of_pretraining_global_fragments + 1)).astype(int)
    split_global_fragments = [global_fragments[indices[i]:indices[i + 1]] for i in range(len(indices) - 1)]
    ordered_split_global_fragments = [order_global_fragments_by_distance_travelled(global_fragments_in_split)[0]
                                    for global_fragments_in_split in split_global_fragments]
    return ordered_split_global_fragments

def get_images_and_labels_from_global_fragment(global_fragment, individual_fragments_identifiers_already_used = []):
    if not np.isnan(global_fragment._ids_assigned).any() and list(global_fragment._temporary_ids) != list(global_fragment._ids_assigned -1):
        raise ValueError("Temporary ids and assigned ids should match in global fragments used for training")
    images = []
    labels = []
    lengths = []
    individual_fragments_identifiers = []
    for i, portraits in enumerate(global_fragment.portraits):
        if global_fragment.individual_fragments_identifiers[i] not in individual_fragments_identifiers_already_used :
            images.extend(portraits)
            labels.extend([global_fragment._temporary_ids[i]]*len(portraits))
            lengths.append(len(portraits))
            individual_fragments_identifiers.append(global_fragment.individual_fragments_identifiers[i])
    return images, labels, lengths, individual_fragments_identifiers

def get_images_and_labels_from_global_fragments(global_fragments, individual_fragments_identifiers_already_used = []):
    logger.info("Getting images from global fragments")
    images = []
    labels = []
    lengths = []
    candidate_individual_fragments_identifiers = []
    individual_fragments_identifiers_already_used = list(individual_fragments_identifiers_already_used)

    for global_fragment in global_fragments:
        images_global_fragment, \
        labels_global_fragment, \
        lengths_global_fragment, \
        individual_fragments_identifiers = get_images_and_labels_from_global_fragment(global_fragment,
                                                                                        individual_fragments_identifiers_already_used)
        if len(images_global_fragment) != 0:
            images.append(images_global_fragment)
            labels.append(labels_global_fragment)
            lengths.extend(lengths_global_fragment)
            candidate_individual_fragments_identifiers.extend(individual_fragments_identifiers)
            individual_fragments_identifiers_already_used.extend(individual_fragments_identifiers)
    if len(images) != 0:
        return np.concatenate(images, axis = 0), np.concatenate(labels, axis = 0), candidate_individual_fragments_identifiers, np.cumsum(lengths)[:-1]
    else:
        return None, None, candidate_individual_fragments_identifiers, None

def get_number_of_images_in_global_fragments_list(global_fragments_list):
    return sum([global_fragment._total_number_of_portraits for global_fragment in global_fragments_list])

def check_uniquenss_of_global_fragments(global_fragments):
    [check_uniquenss_of_global_fragment(global_fragment) for global_fragment in global_fragments]

def check_uniquenss_of_global_fragment(global_fragment):
    if global_fragment._used_for_training == True and not global_fragment.is_unique:
        logger.debug("is unique %s" %global_fragment.is_unique)
        logger.debug("global_fragment ids %s" %str(global_fragment._temporary_ids))
        logger.debug("global_fragment assigned ids %s" %str(global_fragment._ids_assigned))
        raise ValueError("This global Fragment is not unique")

def subsample_images_for_last_training(images, labels, number_of_animals, number_of_samples = 3000):
    """Before assigning identities to the blobs that are not part of the training set we train the network
    a last time with uncorrelated images from the training set of references. This function subsample the
    entire set of training images to get a 'number_of_samples' balanced set.
    """
    subsampled_images = []
    subsampled_labels = []
    for i in np.unique(labels):
        subsampled_images.append(random.sample(images[np.where(labels == i)[0]],number_of_samples))
        subsampled_labels.append([i] * number_of_samples)
    return np.concatenate(subsampled_images, axis = 0), np.concatenate(subsampled_labels, axis = 0)

""" plotter """
def compute_and_plot_global_fragments_statistics(video, list_of_fragments, global_fragments):
    def get_data_plot_individual_fragments(list_of_fragments):
        number_of_images_in_individual_fragments = []
        distance_travelled_individual_fragments = []
        number_of_images_in_crossing_fragments = []
        for fragment in list_of_fragments:
            if fragment.is_a_fish:
                number_of_images_in_individual_fragments.append(fragment.number_of_images)
                distance_travelled_individual_fragments.append(fragment.distance_travelled)
            elif fragment.is_a_crossing:
                number_of_images_in_crossing_fragments.append(fragment.number_of_images)
        return np.asarray(number_of_images_in_individual_fragments),\
                np.asarray(distance_travelled_individual_fragments),\
                number_of_images_in_crossing_fragments

    def get_data_plot_global_fragments(global_fragments):
        number_of_images_in_shortest_individual_fragment = []
        number_of_images_in_longest_individual_fragment = []
        number_of_images_per_individual_fragment_in_global_fragment = []
        median_number_of_images = []
        minimum_distance_travelled = []
        for global_fragment in global_fragments:
            number_of_images_in_shortest_individual_fragment.append(min(global_fragment.number_of_images_per_individual_fragment))
            number_of_images_in_longest_individual_fragment.append(max(global_fragment.number_of_images_per_individual_fragment))
            number_of_images_per_individual_fragment_in_global_fragment.append(global_fragment.number_of_images_per_individual_fragment)
            median_number_of_images.append(np.median(global_fragment.number_of_images_per_individual_fragment))
            minimum_distance_travelled.append(min(global_fragment.distance_travelled_per_individual_fragment))

        return number_of_images_in_shortest_individual_fragment,\
                number_of_images_in_longest_individual_fragment,\
                number_of_images_per_individual_fragment_in_global_fragment,\
                median_number_of_images,\
                minimum_distance_travelled

    number_of_images_in_individual_fragments, \
    distance_travelled_individual_fragments, \
    number_of_images_in_crossing_fragments =  get_data_plot_individual_fragments(list_of_fragments)
    
    number_of_images_in_shortest_individual_fragment,\
    number_of_images_in_longest_individual_fragment,\
    number_of_images_per_individual_fragment_in_global_fragment,\
    median_number_of_images,\
    minimum_distance_travelled = get_data_plot_global_fragments(global_fragments)
    ''' plotting '''
    plt.ion()
    sns.set_style("ticks")
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    fig, ax_arr = plt.subplots(2,4)
    fig.set_size_inches((screen_x/100,screen_y/100))
    plt.subplots_adjust(hspace = .3, wspace = .5)
    # number of frames in individual fragments
    nbins = 25
    ax = ax_arr[0,0]
    MIN = np.min(number_of_images_in_individual_fragments)
    MAX = np.max(number_of_images_in_individual_fragments)
    hist, bin_edges = np.histogram(number_of_images_in_individual_fragments, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))
    ax.semilogx(bin_edges[:-1], hist, '-ob' ,markersize = 5)
    ax.set_xlabel('number of images')
    ax.set_ylabel('number of individual fragments')
    # distance travelled in individual fragments
    non_zero_indices = np.where(distance_travelled_individual_fragments != 0)[0]
    distance_travelled_individual_fragments_non_zero = distance_travelled_individual_fragments[non_zero_indices]
    ax = ax_arr[0,1]
    MIN = np.min(distance_travelled_individual_fragments_non_zero)
    MAX = np.max(distance_travelled_individual_fragments_non_zero)
    hist, bin_edges = np.histogram(distance_travelled_individual_fragments, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))
    ax.semilogx(bin_edges[:-1], hist, '-ob' ,markersize = 5)
    ax.set_xlabel('distance travelled (pixels)')
    # number of frames vs distance travelled
    ax = ax_arr[0,2]
    ax.plot(np.asarray(number_of_images_in_individual_fragments)[non_zero_indices], distance_travelled_individual_fragments_non_zero, 'bo', alpha = .1, label = 'individual fragment', markersize = 5)
    ax.set_xlabel('num frames')
    ax.set_ylabel('distance travelled (pixels)')
    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')
    # number of frames in shortest individual fragment
    ax = ax_arr[0,3]
    MIN = np.min(number_of_images_in_crossing_fragments)
    MAX = np.max(number_of_images_in_crossing_fragments)
    hist, bin_edges = np.histogram(number_of_images_in_crossing_fragments, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))
    ax.semilogx(bin_edges[:-1],hist, 'ro-', markersize = 5)
    ax.set_xlabel('number of images')
    ax.set_ylabel('number of crossing fragments')
    # plot global fragments
    ax = plt.subplot2grid((2, 4), (1, 0), colspan=4)
    index_order_by_max_num_frames = np.argsort(minimum_distance_travelled)[::-1]
    number_of_images_in_longest_individual_fragment = np.asarray(number_of_images_in_longest_individual_fragment)[index_order_by_max_num_frames]
    number_of_images_in_shortest_individual_fragment = np.asarray(number_of_images_in_shortest_individual_fragment)[index_order_by_max_num_frames]
    number_of_images_per_individual_fragment_in_global_fragment = np.asarray(number_of_images_per_individual_fragment_in_global_fragment)[index_order_by_max_num_frames]
    median_number_of_images = np.asarray(median_number_of_images)[index_order_by_max_num_frames]
    a = ax.semilogy(range(len(global_fragments)), median_number_of_images, color = 'b', linewidth= 2, label = 'median')

    for i in range(len(global_fragments)):
        a = ax.semilogy(i*np.ones(video.number_of_animals),number_of_images_per_individual_fragment_in_global_fragment[i],'o',alpha = .05,color = 'b',markersize=5,label='individual fragment')
    b = ax.semilogy(range(len(global_fragments)), number_of_images_in_longest_individual_fragment, color = 'r', linewidth= 2 ,alpha = .5, label = 'max')
    c = ax.semilogy(range(len(global_fragments)), median_number_of_images, color = 'r', linewidth= 2, label = 'median')
    d = ax.semilogy(range(len(global_fragments)), number_of_images_in_shortest_individual_fragment, color = 'r', linewidth= 2 ,alpha = .5, label = 'min')
    ax.set_xlabel('global fragments ordered by minimum distance travelled (from max to min)')
    ax.set_ylabel('num of frames')
    ax.legend(handles = [c[0],d[0],b[0],a[0]])

    plt.show()
    fig.savefig(os.path.join(video._preprocessing_folder,'global_fragments_summary.pdf'), transparent=True)
    return number_of_images_in_individual_fragments, distance_travelled_individual_fragments
