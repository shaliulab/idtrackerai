from __future__ import absolute_import, print_function, division
import sys
sys.path.append('../')
sys.path.append('../preprocessing')
sys.path.append('../utils')
sys.path.append('../network')
import numpy as np
from tqdm import tqdm
import collections
import logging

from list_of_blobs import ListOfBlobs
from blob import Blob
from video_utils import segmentVideo, filterContoursBySize, getPixelsList, getBoundigBox
from statistics_for_assignment import compute_P2_of_individual_fragment_from_blob


def solve_duplications_loop(video, blobs_in_video, group_size, scope = None):
    if scope == 'to_the_past':
        blobs_in_direction = blobs_in_video[:video.first_frame_first_global_fragment][::-1]
    elif scope == 'to_the_future':
        blobs_in_direction = blobs_in_video[video.first_frame_first_global_fragment:-1]
    possible_identities = set(range(1,group_size+1))

    for blobs_in_frame in tqdm(blobs_in_direction, desc = 'Solving duplications ' + scope):
        duplicated_identities, identities_in_frame, missing_identities, _ = check_for_duplications(blobs_in_frame, possible_identities)
        assign_duplicated_identities_in_frame(blobs_in_video, blobs_in_frame, duplicated_identities, missing_identities, identities_in_frame)
        duplicated_identities, identities_in_frame, missing_identities, _ = check_for_duplications(blobs_in_frame, possible_identities)
        missing_identities = assign_single_unidentified_blob(missing_identities, blobs_in_frame, blobs_in_video)

def solve_duplications_loop(list_of_fragments, scope = None):
    fragments_in_direction = list_of_fragments.get_ordered_list_of_fragments(scope)

    for fragment in fragments_in_direction:
        # print(fragment.is_an_individual, fragment.identity != 0, fragment.is_a_duplication, not fragment.used_for_training, not hasattr(fragment, 'identity_corrected_solving_duplication'))
        if fragment.is_an_individual\
            and fragment.identity != 0\
            and fragment.is_a_duplication\
            and not fragment.used_for_training\
            and fragment.identity_corrected_solving_duplication is None:
            # print("\n***fragment: ", fragment.identifier, fragment.start_end, fragment.assigned_identity, fragment.number_of_images)
            solve_duplication(fragment)

def solve_duplication(fragment):
    fixed_identities = fragment.get_fixed_identities_of_coexisting_fragments()
    missing_identities = fragment.get_missing_identities_in_coexisting_fragments(fixed_identities)

    # print("missing_identities: ", missing_identities)
    # print("fixed_identities: ", fixed_identities)
    print([(f.identifier, f.assigned_identity, f.is_a_duplication, f.used_for_training) for f in fragment.coexisting_individual_fragments])
    number_of_candidate_identities = len(missing_identities)
    # case 1
    if len(missing_identities) == 0:
        # print("case 1")
        if fragment.identity not in fixed_identities:
            fragment._identity_corrected_solving_duplication = fragment.identity
        else:
            fragment._identity_corrected_solving_duplication = 0
    # case 2
    elif len(missing_identities) == 1:
        # print("case 2")
        fragment._identity_corrected_solving_duplication = missing_identities[0]
        # print("final identity: ", fragment.assigned_identity)
    # case 3
    else:
        # print("case 3")
        fragments_with_duplicated_ids = [fragment] + [coexisting_fragment
                for coexisting_fragment in
                fragment.coexisting_individual_fragments
                if coexisting_fragment.assigned_identity == fragment.identity and
                (not coexisting_fragment.used_for_training
                or coexisting_fragment.identity_corrected_solving_duplication is not None
                or coexisting_fragment.user_generated_identity is not None)]
        # get P2 matrix of fragments in fragments_with_duplicated_ids
        P2_matrix = np.asarray([duplicated_fragment.P2_vector for duplicated_fragment in fragments_with_duplicated_ids])
        # print("P2_matrix (no zeros): ", P2_matrix)
        # set to 0 columns in the P2_matrix that correspond to fixed_identities
        P2_matrix[:, np.asarray(fixed_identities).astype("int") - 1] = 0
        # print("P2_matrix (zeros): ",P2_matrix)
        argsort_row_zero = np.argsort(P2_matrix[0,:])[::-1]
        # print("argsort_row_zero: ", argsort_row_zero)
        # random_threshold = 1/fragment.number_of_images if fragment.number_of_images > 1 else 0
        random_threshold = 0
        # print("random_threshold: ", random_threshold)
        for index in argsort_row_zero:
            # print("index: ", index)
            P2_max_col = max(P2_matrix[0,:])
            # print("P2_max_col: ", P2_max_col)
            max_indices_col = np.where(P2_matrix[0,:] == P2_max_col)[0]
            # print("max_indices_col: ", max_indices_col)
            # print("number_of_candidate_identities: ", number_of_candidate_identities)
            if (number_of_candidate_identities > 1) and (P2_max_col == 0 or P2_max_col < random_threshold):
                # print("case 3.1")
                fragment._identity_corrected_solving_duplication = 0
                # print("final identity: ", fragment.assigned_identity)
                break
            elif number_of_candidate_identities == 1:
                # print("case 3.2")
                fragment._identity_corrected_solving_duplication = max_indices_col[0] + 1
                # print("final identity: ", fragment.assigned_identity)
                break
            if len(max_indices_col) == 1 and len(fragments_with_duplicated_ids) > 1:
                # print("case 3.3")
                if P2_max_col > np.max(P2_matrix[1:,max_indices_col[0]]):
                    # print("case 3.3.1")
                    fragment._identity_corrected_solving_duplication = max_indices_col[0] + 1
                    # print("final identity: ", fragment.assigned_identity)
                    break
                elif P2_max_col == np.max(P2_matrix[1:,max_indices_col[0]]):
                    # print("case 3.3.2")
                    fragment_indices = np.where(P2_matrix[:,max_indices_col[0]] == P2_max_col)[0]
                    # print("fragment_indices:", fragment_indices)
                    for fragment_index in fragment_indices:
                        fragments_with_duplicated_ids[fragment_index]._identity_corrected_solving_duplication = 0
                        # print("final identity: ", fragment.assigned_identity)
                    break
                elif P2_max_col < np.max(P2_matrix[1:,max_indices_col[0]]):
                    # print("case 3.3.3")
                    P2_matrix[:,index] = 0
                    number_of_candidate_identities -= 1
            elif len(max_indices_col) == 1 and len(fragments_with_duplicated_ids) == 1:
                # print("case 3.4")
                fragment._identity_corrected_solving_duplication = max_indices_col[0] + 1
                # print("final identity: ", fragment.assigned_identity)
                break
            elif len(max_indices_col) > 1:
                # print("case 3.5")
                fragment._identity_corrected_solving_duplication = 0
                # print("final identity: ", fragment.assigned_identity)
                break

def solve_duplications(list_of_fragments):
    check_for_duplications_last_pass(list_of_fragments.fragments)
    solve_duplications_loop(list_of_fragments, scope = 'to_the_past')
    solve_duplications_loop(list_of_fragments, scope = 'to_the_future')
    check_for_duplications_last_pass(list_of_fragments.fragments)

def mark_fragments_as_duplications(fragments):
    [fragment.set_duplication_flag() for fragment in fragments]

def check_for_duplications_last_pass(fragments):
    duplicated_fragments_start_end = []
    for fragment in fragments:
        overlapping_identities = [fragment.assigned_identity == coexisting_fragment.assigned_identity
                                    for coexisting_fragment in fragment.coexisting_individual_fragments
                                    if (fragment.assigned_identity != 0 and coexisting_fragment.assigned_identity != 0)]
        if sum(overlapping_identities) != 0:
            duplicated_fragments_start_end.append((fragment.identifier, fragment.start_end))

    print("start end tuples of fragments with duplicated identity ",duplicated_fragments_start_end)
    return duplicated_fragments_start_end
