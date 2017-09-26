from __future__ import absolute_import, print_function, division
import sys
import numpy as np
import logging

MAX_FLOAT = sys.float_info[0]
MIN_FLOAT = sys.float_info[3]

logger = logging.getLogger("__main__.statistics_for_assignment")

# def compute_identification_frequencies_individual_fragment(non_shared_information_in_fragment, identities_in_fragment, number_of_animals):
#     """Given a list of identities computes the frequencies based on the
#     network's assignment on the individual fragment.
#     """
#     # Compute frequencies of assignation for each identity
#     # assert len(non_shared_information_in_fragment) == len(identities_in_fragment)
#     # non_shared_information_in_fragment = np.asarray(non_shared_information_in_fragment)
#     # if np.any(non_shared_information_in_fragment == None):
#     #     index = np.where(non_shared_information_in_fragment == None)
#     #     non_shared_information_in_fragment[index] = np.mean([nsi for nsi in non_shared_information_in_fragment if nsi is not None])
#     # elif sum(np.isnan(non_shared_information_in_fragment)):
#     #     index = np.where(np.isnan(non_shared_information_in_fragment))
#     #     non_shared_information_in_fragment[index] = np.nanmean(non_shared_information_in_fragment)
#     #
#     # weighted_frequencies =  np.asarray([np.sum((identities_in_fragment == i) * non_shared_information_in_fragment)
#     #                             for i in range(1, number_of_animals+1)]) # The predictions come from 1 to number_of_animals + 1
#     # if sum(np.isnan(weighted_frequencies)):
#     #     logger.debug("identities_in_fragment: %s", str(identities_in_fragment))
#     #     logger.debug("non_shared_information_in_fragment: %s", str(non_shared_information_in_fragment))
#     #     logger.debug("weighted_frequencies: %s" str(weighted_frequencies))
#     #     raise ValueError("the weighted_frequencies are nan")
#     #
#     # return weighted_frequencies
#     return np.asarray([np.sum(identities_in_fragment == i)
#                             for i in range(1, number_of_animals+1)]) # The predictions come from 1 to number_of_animals + 1

def normalise_frequencies(frequencies):
    return frequencies / np.sum(frequencies)

# def compute_P1_individual_fragment_from_frequencies(frequencies):
#     """Given the frequencies of a individual fragment
#     computer the P1 vector. P1 is the softmax of the frequencies with base 2
#     for each identity.
#     """
#     # Compute numerator of P1 and check that it is not inf
#     numerator = 2.**frequencies
#     if np.any(numerator == np.inf):
#         numerator[numerator == np.inf] = MAX_FLOAT
#     # Compute denominator of P1
#     denominator = np.sum(numerator)
#     # Compute P1 and check that it is not 0. for any identity
#     P1_of_fragment = numerator / denominator
#     if np.all(P1_of_fragment == 0.):
#         P1_of_fragment[P1_of_fragment == 0.] = 1/len(P1_of_fragment) #if all the frequencies are very high then the denominator is very big and all the P1 are 0. so we set then to random.
#     else:
#         P1_of_fragment[P1_of_fragment == 0.] = MIN_FLOAT
#     # Change P1 that are 1. for 0.9999 so that we do not have problems when computing P2
#     # P1_of_fragment[P1_of_fragment == 1.] = 1. - MIN_FLOAT
#     # P1_of_fragment = P1_of_fragment / np.sum(P1_of_fragment)
#     P1_of_fragment[P1_of_fragment == 1.] = 0.999999999999
#
#     return P1_of_fragment

def compute_P2_of_individual_fragment_from_blob(blob, blobs_in_video):
    # logger.debug("****Computing P2")
    coexisting_blobs, _ = blob.get_coexisting_blobs_in_fragment(blobs_in_video)
    coexisting_blobs_P1_vectors = np.asarray([coexisting_blob.P1_vector for coexisting_blob in coexisting_blobs])
    # logger.debug("coexisting_blobs_P1_vectors: %s" %str(coexisting_blobs_P1_vectors))
    numerator = np.asarray(blob.P1_vector) * np.prod(1. - coexisting_blobs_P1_vectors, axis = 0)
    # logger.debug("second factor in P2: %s" %str(np.prod(1. - coexisting_blobs_P1_vectors, axis = 0)))
    # logger.debug("numerator: %s" %str(numerator))
    denominator = np.sum(numerator)
    if denominator == 0:
        # logger.debug("P1 of blob: %s", %str(blob.P1_vector))
        # logger.debug("coexisting_blobs_P1_vectors: %s" %str(coexisting_blobs_P1_vectors))
        # raise ValueError('denominator of P2 is 0')
        P2 = blob.P1_vector
    else:
        P2 = numerator / denominator
    return P2

def is_assignment_ambiguous(P2_vector):
    """Check if P2 has two identical maxima. In that case returns the indices.
    Else return false.
    """
    maxima_indices = np.where(P2_vector == np.max(P2_vector))[0]
    return maxima_indices + 1, len(maxima_indices) > 1
