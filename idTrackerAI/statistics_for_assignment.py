from __future__ import absolute_import, print_function, division
import numpy as np
from constants import MAX_FLOAT, MIN_FLOAT
import sys
if sys.argv[0] == 'idtrackerdeepApp.py':
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.statistics_for_assignment")

def normalise_frequencies(frequencies):
    return frequencies / np.sum(frequencies)

def compute_P2_of_individual_fragment_from_blob(blob, blobs_in_video):
    coexisting_blobs, _ = blob.get_coexisting_blobs_in_fragment(blobs_in_video)
    coexisting_blobs_P1_vectors = np.asarray([coexisting_blob.P1_vector for coexisting_blob in coexisting_blobs])
    numerator = np.asarray(blob.P1_vector) * np.prod(1. - coexisting_blobs_P1_vectors, axis = 0)
    denominator = np.sum(numerator)
    if denominator == 0:
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
