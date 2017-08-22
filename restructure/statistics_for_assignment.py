from __future__ import absolute_import, print_function, division
import sys
import numpy as np

MAX_FLOAT = sys.float_info[0]
MIN_FLOAT = sys.float_info[3]

def compute_identification_frequencies_individual_fragment(non_shared_information_in_fragment, identities_in_fragment, number_of_animals):
    """Given a list of identities computes the frequencies based on the
    network's assignment on the individual fragment.
    """
    # Compute frequencies of assignation for each identity
    # return np.asarray([np.sum((identities_in_fragment == i) * non_shared_information_in_fragment)
    #                         for i in range(1, number_of_animals+1)]) # The predictions come from 1 to number_of_animals + 1
    return np.asarray([np.sum(identities_in_fragment == i)
                            for i in range(1, number_of_animals+1)]) # The predictions come from 1 to number_of_animals + 1

def normalise_frequencies(frequencies):
    return frequencies / np.sum(frequencies)

def compute_P1_individual_fragment_from_frequencies(frequencies):
    """Given the frequencies of a individual fragment
    computer the P1 vector. P1 is the softmax of the frequencies with base 2
    for each identity.
    """
    # Compute numerator of P1 and check that it is not inf
    numerator = 2.**frequencies
    if np.any(numerator == np.inf):
        numerator[numerator == np.inf] = MAX_FLOAT
    # Compute denominator of P1
    denominator = np.sum(numerator)
    # Compute P1 and check that it is not 0. for any identity
    P1_of_fragment = numerator / denominator
    if np.all(P1_of_fragment == 0.):
        P1_of_fragment[P1_of_fragment == 0.] = 1/len(P1_of_fragment) #if all the frequencies are very high then the denominator is very big and all the P1 are 0. so we set then to random.
    else:
        P1_of_fragment[P1_of_fragment == 0.] = MIN_FLOAT
    # Change P1 that are 1. for 0.9999 so that we do not have problems when computing P2
    P1_of_fragment[P1_of_fragment == 1.] = 1. - MIN_FLOAT

    return P1_of_fragment

def compute_P2_of_individual_fragment_from_blob(blob, blobs_in_video):
    # print("\n****Computing P2")
    coexisting_blobs_P1_vectors = blob.get_P1_vectors_coexisting_fragments(blobs_in_video)
    # print("coexisting_blobs_P1_vectors: ", coexisting_blobs_P1_vectors)
    # Compute P2
    numerator = np.asarray(blob.P1_vector) * np.prod(1. - coexisting_blobs_P1_vectors, axis = 0)
    denominator = np.sum(numerator)
    if denominator == 0:
        print("P1 of blob, ", blob.P1_vector)
        print("coexisting_blobs_P1_vectors, ", coexisting_blobs_P1_vectors)
        raise ValueError('denominator of P2 is 0')

    P2 = numerator / denominator
    return P2

def is_assignment_ambiguous(P2_vector):
    """Check if P2 has two identical maxima. In that case returns the indices.
    Else return false.
    """
    maxima_indices = np.where(P2_vector == np.max(P2_vector))[0]
    return maxima_indices + 1, len(maxima_indices) > 1
