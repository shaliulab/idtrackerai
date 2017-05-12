from __future__ import absolute_import, print_function, division
import sys
import numpy as np

MAX_FLOAT = sys.float_info[0]
MIN_FLOAT = sys.float_info[3]

def compute_identification_frequencies_individual_fragment(identities_in_fragment, number_of_animals):
    """Given a list of identities computes the frequencies based on the
    network's assignment on the individual fragment.
    """
    # Compute frequencies of assignation for each identity
    return np.asarray([np.sum(identities_in_fragment == i)
                            for i in range(1, number_of_animals + 1)])

def normalise_frequencies(frequencies):
    return frequencies / np.sum(frequencies)

def compute_P1_individual_fragment_from_blob(frequencies):
    """Given a blob it computes P1 vector for the individual fragment containing
    the blob. P1 is the softmax of the frequencies with base 2.
    """
    # Compute numerator of P1 and check that it is not inf
    numerator = 2.**frequencies
    if np.any(numerator == np.inf):
        numerator[numerator == np.inf] = MAX_FLOAT
    # Compute denominator of P1
    denominator = np.sum(numerator)
    # Compute P1 and check that it is not 0. for any identity
    P1_of_fragment = numerator / denominator
    if np.any(P1_of_fragment == 0.):
        P1_of_fragment[P1_of_fragment == 0.] = MIN_FLOAT
    if np.any(P1_of_fragment == 0.):
        raise ValueError('P1_of_fragment cannot be 0')
    # Change P1 that are 1. for 0.9999 so that we do not have problems when computing P2
    P1_of_fragment[P1_of_fragment == 1.] = 0.9999
    if np.any(P1_of_fragment == 1.):
        raise ValueError('P1_of_fragment cannot be 1')
    return P1_of_fragment
