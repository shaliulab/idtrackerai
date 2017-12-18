from __future__ import absolute_import, division, print_function
import os
import random
import logging
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

from globalfragment import GlobalFragment
from get_predictions import GetPrediction
from assigner import assign, compute_identification_statistics_for_non_accumulated_fragments
from accumulation_manager import AccumulationManager
from constants import CERTAINTY_THRESHOLD

logger = logging.getLogger("__main__.list_of_global_fragments")

class ListOfGlobalFragments(object):
    def __init__(self, global_fragments):
        self.global_fragments = global_fragments
        self.number_of_global_fragments = len(self.global_fragments)

    def reset(self, roll_back_to = None):
        [global_fragment.reset(roll_back_to) for global_fragment in self.global_fragments]

    def order_by_distance_travelled(self):
        self.global_fragments = sorted(self.global_fragments, key = lambda x: x.minimum_distance_travelled, reverse = True)

    @staticmethod
    def give_me_frequencies_first_fragment_accumulated(i, number_of_animals, fragment):
        frequencies = np.zeros(number_of_animals)
        frequencies[i] = fragment.number_of_images
        return frequencies

    @staticmethod
    def abort_knowledge_transfer_on_same_animals(video, net):
        identities = range(video.number_of_animals)
        net.reinitialize_softmax_and_fully_connected()
        logger.info("Identity transfer failed. We proceed by transferring only the convolutional filters.")
        return identities

    def set_first_global_fragment_for_accumulation(self, video, net, accumulation_trial):
        self.order_by_distance_travelled()
        self.first_global_fragment_for_accumulation = self.global_fragments[accumulation_trial]
        if not video.identity_transfer:
            identities = range(video.number_of_animals)
        else:
            if net is None:
                raise ValueError("In order to associate an identity to the same animals a network is needed")
            images, _ = self.first_global_fragment_for_accumulation.get_images_and_labels()
            images = np.asarray(images)
            assigner = assign(net, video, images, False)
            compute_identification_statistics_for_non_accumulated_fragments(
                self.first_global_fragment_for_accumulation.individual_fragments,
                assigner, net.params.number_of_animals)
            # Check certainties of the individual fragments in the global fragment
            # for individual_fragment_identifier in global_fragment.individual_fragments_identifiers:
            [setattr(fragment,'_acceptable_for_training', True) for fragment
            in self.first_global_fragment_for_accumulation.individual_fragments]

            for fragment in self.first_global_fragment_for_accumulation.individual_fragments:
                if AccumulationManager.is_not_certain(fragment, CERTAINTY_THRESHOLD):
                    identities = self.abort_knowledge_transfer_on_same_animals(video, net)
                    break

            P1_array, index_individual_fragments_sorted_by_P1_max_to_min = AccumulationManager.get_P1_array_and_argsort(
                                    self.first_global_fragment_for_accumulation)

            # assign temporary identity to individual fragments by hierarchical P1
            for index_individual_fragment in index_individual_fragments_sorted_by_P1_max_to_min:
                fragment = self.first_global_fragment_for_accumulation.individual_fragments[index_individual_fragment]

                if AccumulationManager.p1_below_random(P1_array, index_individual_fragment, fragment):
                    identities = self.abort_knowledge_transfer_on_same_animals(video, net)
                    break
                else:
                    temporary_id = np.argmax(P1_array[index_individual_fragment,:])
                    if not fragment.check_consistency_with_coexistent_individual_fragments(temporary_id):
                        identities = self.abort_knowledge_transfer_on_same_animals(video, net)
                        break
                    else:
                        P1_array = AccumulationManager.set_fragment_temporary_id(
                                                fragment, temporary_id, P1_array,
                                                index_individual_fragment)

            # Check if the global fragment is unique after assigning the identities
            if not self.first_global_fragment_for_accumulation.is_partially_unique:
                identities = self.abort_knowledge_transfer_on_same_animals(video, net)
                logger.info("Identity transfer is not possible. Identities will be intialized")
            else:
                video._first_global_fragment_knowledge_transfer_identities = [fragment.temporary_id for fragment
                            in self.first_global_fragment_for_accumulation.individual_fragments]
                if video.number_of_animals == video.knowledge_transfer_info_dict['number_of_animals']:
                    identities = video._first_global_fragment_knowledge_transfer_identities
                elif video.number_of_animals < video.knowledge_transfer_info_dict['number_of_animals']:
                    identities = range(video.number_of_animals)
                logger.info("Identities transferred succesfully")

                self.plot_P1s_identity_transfer(video)

        [(setattr(fragment, '_acceptable_for_training', True),
            setattr(fragment, '_temporary_id', identities[i]),
            setattr(fragment, '_frequencies', self.give_me_frequencies_first_fragment_accumulated(i, video.number_of_animals, fragment)),
            setattr(fragment, '_is_certain', True),
            setattr(fragment, '_certainty', 1.),
            setattr(fragment, '_P1_vector', fragment.compute_P1_from_frequencies(fragment.frequencies)))
            for i, fragment in enumerate(self.first_global_fragment_for_accumulation.individual_fragments)]

        return self.first_global_fragment_for_accumulation.index_beginning_of_fragment

    def plot_P1s_identity_transfer(self, video):
        P1_vectors = np.asarray([fragment.P1_vector for fragment in self.first_global_fragment_for_accumulation.individual_fragments])
        certainties = np.asarray([fragment.certainty for fragment in self.first_global_fragment_for_accumulation.individual_fragments])
        indices = np.argmax(P1_vectors, axis = 1)
        P1_vectors_ordered = np.zeros((video.knowledge_transfer_info_dict['number_of_animals'],video.knowledge_transfer_info_dict['number_of_animals']))
        P1_vectors_ordered[indices, :] = P1_vectors
        certainties_ordered = np.zeros(video.knowledge_transfer_info_dict['number_of_animals'])
        certainties_ordered[indices] = certainties

        fig = plt.figure()
        sns.set_style("ticks")
        fig.suptitle('Identity transfer summary')
        ax0 = fig.add_subplot(121)
        im = ax0.imshow(P1_vectors_ordered)
        ax0.invert_yaxis()
        ax0.set_ylabel('fragment')
        ax0.set_xlabel('transferred identity')
        im.set_clim(0.0,1.0)
        cbar = plt.colorbar(im, orientation='horizontal', label = 'probability of assignment')
        pos0 = ax0.get_position()

        ax1 = fig.add_subplot(122)
        pos1 = ax1.get_position()
        ax1.set_position([pos1.x0, pos0.y0, pos1.width, pos0.height - .025])
        ax1.barh(range(len(certainties_ordered)), certainties_ordered, height = .5)
        ax1.set_xlim((0,1))
        ax1.set_xlabel('certainty')
        ax1.set_ylim(ax0.get_ylim())
        ax1.set_yticklabels([])
        sns.despine(ax = ax1, left=False, top = True, bottom=False, right=True)


        fig.savefig(os.path.join(video.session_folder,'identity_transfer_summary.pdf'), transparent=True)


    def order_by_distance_to_the_first_global_fragment_for_accumulation(self, video, accumulation_trial = None):
        self.global_fragments = sorted(self.global_fragments,
                                        key = lambda x: np.abs(x.index_beginning_of_fragment - video.first_frame_first_global_fragment[accumulation_trial]),
                                        reverse = False)

    def compute_maximum_number_of_images(self):
        self.maximum_number_of_images = max([global_fragment.get_total_number_of_images() for global_fragment in self.global_fragments])

    def filter_candidates_global_fragments_for_accumulation(self):
        self.non_accumulable_global_fragments = [global_fragment for global_fragment in self.global_fragments
                    if not global_fragment.candidate_for_accumulation]
        self.global_fragments = [global_fragment for global_fragment in self.global_fragments
                    if global_fragment.candidate_for_accumulation]
        self.number_of_global_fragments = len(self.global_fragments)

    def get_data_plot(self):
        number_of_images_in_shortest_individual_fragment = []
        number_of_images_in_longest_individual_fragment = []
        number_of_images_per_individual_fragment_in_global_fragment = []
        median_number_of_images = []
        minimum_distance_travelled = []
        for global_fragment in self.global_fragments:
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

    def delete_fragments_from_global_fragments(self):
        [setattr(global_fragment,'individual_fragments',None) for global_fragment in self.global_fragments]

    def relink_fragments_to_global_fragments(self, fragments):
        [global_fragment.get_individual_fragments_of_global_fragment(fragments) for global_fragment in self.global_fragments]

    def save(self, global_fragments_path, fragments):
        logger.info("saving list of global fragments at %s" %global_fragments_path)
        self.delete_fragments_from_global_fragments()
        np.save(global_fragments_path,self)
        # After saving the list of globa fragments the individual fragments are deleted and we need to relink them again
        self.relink_fragments_to_global_fragments(fragments)

    @classmethod
    def load(self, path_to_load, fragments):
        logger.info("loading list of global fragments from %s" %path_to_load)
        list_of_global_fragments = np.load(path_to_load).item()
        list_of_global_fragments.relink_fragments_to_global_fragments(fragments)
        return list_of_global_fragments

def detect_beginnings(boolean_array):
    """ detects the frame where the core of a global fragment starts.
    A core of a global fragment is the part of the global fragment where all the
    individuals are visible, i.e. the number of animals in the frame equals the
    number of animals in the video
    :boolean_array: array with True where the number of animals in the frame equals
    the number of animals in the video
    """
    return [i for i in range(0,len(boolean_array)) if (boolean_array[i] and not boolean_array[i-1])]

def check_global_fragments(blobs_in_video, num_animals):
    """Returns an array with True if:
    * each blob has a unique blob intersecting in the past and future
    * number of blobs equals num_animals
    """
    def all_blobs_in_a_fragment(blobs_in_frame):
        # return all([blob.is_in_a_fragment for blob in blobs_in_frame])
        return all([blob.is_an_individual for blob in blobs_in_frame])

    return [all_blobs_in_a_fragment(blobs_in_frame) and len(blobs_in_frame) == num_animals for blobs_in_frame in blobs_in_video]

def create_list_of_global_fragments(blobs_in_video, fragments, num_animals):
    global_fragments_boolean_array = check_global_fragments(blobs_in_video, num_animals)
    indices_beginning_of_fragment = detect_beginnings(global_fragments_boolean_array)
    return [GlobalFragment(blobs_in_video, fragments, i, num_animals) for i in indices_beginning_of_fragment]
