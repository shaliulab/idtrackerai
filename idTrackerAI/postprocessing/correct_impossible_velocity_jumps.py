from __future__ import absolute_import, print_function, division
import sys
sys.path.append('./')
sys.path.append('./preprocessing')
sys.path.append('./utils')
sys.path.append('./network')
sys.path.append('./plots')
import numpy as np
from tqdm import tqdm
import collections
import logging

from list_of_blobs import ListOfBlobs
from assigner import assign
from id_CNN import ConvNetwork
from network_params import NetworkParams
from blob import Blob
from compute_velocity_model import compute_velocity_from_list_of_fragments, compute_model_velocity
from get_trajectories import produce_trajectories
from plot_individual_velocity import plot_individual_trajectories_velocities_and_accelerations

VEL_PERCENTILE = 99 #percentile used to model velocity jumps
P2_CERTAINTY_THRESHOLD = .9
VELOCITY_TOLERANCE = 1.5

def reassign(fragment, fragments, impossible_velocity_threshold):
    def get_available_and_non_available_identities(fragment):
        non_available_identities = set([coexisting_fragment.assigned_identity for coexisting_fragment in fragment.coexisting_individual_fragments])
        available_identities = set(range(1, fragment.number_of_animals + 1)) - non_available_identities
        if fragment.assigned_identity is not None and fragment.assigned_identity != 0:
            available_identities = available_identities | set([fragment.assigned_identity])
        if 0 in non_available_identities: non_available_identities.remove(0)
        non_available_identities = np.array(list(non_available_identities))
        return non_available_identities, available_identities


    def get_candidate_identities_by_minimum_speed(fragment, fragments, available_identities, impossible_velocity_threshold):
        speed_of_candidate_identities = []
        for identity in available_identities:
            fragment._user_generated_identity = identity
            neighbour_fragment_past = fragment.get_neighbour_fragment(fragments, 'to_the_past')
            neighbour_fragment_future = fragment.get_neighbour_fragment(fragments, 'to_the_future')
            velocities_between_fragments = compute_velocities_consecutive_fragments(neighbour_fragment_past, fragment, neighbour_fragment_future)

            if np.all(np.isnan(velocities_between_fragments)):
                speed_of_candidate_identities.append(impossible_velocity_threshold)
            else:
                speed_of_candidate_identities.append(np.nanmin(velocities_between_fragments))
        fragment._user_generated_identity = None
        argsort_identities_by_speed = np.argsort(speed_of_candidate_identities)
        return np.asarray(list(available_identities))[argsort_identities_by_speed], np.asarray(speed_of_candidate_identities)[argsort_identities_by_speed]

    def get_candidate_identities_by_above_random_P2(fragment, fragments, non_available_identities, available_identities, impossible_velocity_threshold):
        P2_vector = fragment.P2_vector
        if len(non_available_identities) > 0:
            P2_vector[non_available_identities - 1] = 0
        if np.all(P2_vector == 0):
            candidate_identities_speed, _ = get_candidate_identities_by_minimum_speed(fragment, fragments, available_identities, impossible_velocity_threshold)
            return candidate_identities_speed
        else:
            if fragment.number_of_images == 1:
                random_threshold  = 1/fragment.number_of_animals
            else:
                random_threshold = 1/fragment.number_of_images
            return np.where(P2_vector > random_threshold)[0] + 1

    non_available_identities, available_identities = get_available_and_non_available_identities(fragment)
    if len(available_identities) == 1:
        candidate_id = list(available_identities)[0]
    elif len(available_identities) == 0:
        candidate_id = fragment.assigned_identity
    else:
        candidate_identities_speed, speed_of_candidate_identities = get_candidate_identities_by_minimum_speed(fragment, fragments, available_identities, impossible_velocity_threshold)
        candidate_identities_P2 = get_candidate_identities_by_above_random_P2(fragment,
                                                                    fragments,
                                                                    non_available_identities,
                                                                    available_identities,
                                                                    impossible_velocity_threshold)
        candidate_identities = []
        candidate_speeds = []
        for candidate_id, candidate_speed in zip(candidate_identities_speed, speed_of_candidate_identities):
            if candidate_id in candidate_identities_P2:
                candidate_identities.append(candidate_id)
                candidate_speeds.append(candidate_speed)
        if len(candidate_identities) == 0:
            candidate_id = 0
        elif len(candidate_identities) == 1:
            if candidate_speeds[0] < impossible_velocity_threshold:
                candidate_id = candidate_identities[0]
            else:
                candidate_id = 0
        elif len(candidate_identities) > 1:
            if len(np.where(candidate_speeds == np.min(candidate_speeds))[0]) == 1:
                if candidate_speeds[0] < impossible_velocity_threshold:
                    candidate_id = candidate_identities[0]
                else:
                    candidate_id = 0
            else:
                candidate_id = 0

    if fragment._identity_corrected_solving_duplication is not None:
        fragment._identity_corrected_solving_duplication = candidate_id
    else:
        fragment._identity = candidate_id

def compute_velocities_consecutive_fragments(neighbour_fragment_past, fragment, neighbour_fragment_future):
    velocities = [np.nan, np.nan]
    if neighbour_fragment_past is not None:
        velocities[0] = fragment.compute_border_velocity(neighbour_fragment_past)
    if neighbour_fragment_future is not None:
        velocities[1] = neighbour_fragment_future.compute_border_velocity(fragment)
    return velocities

def correct_impossible_velocity_jumps_loop(video, list_of_fragments, scope = None):
    fragments_in_direction = list_of_fragments.get_ordered_list_of_fragments(scope, video.first_frame_first_global_fragment)
    impossible_velocity_threshold = video.velocity_threshold * VELOCITY_TOLERANCE

    for fragment in tqdm(fragments_in_direction, desc = 'Correcting impossible velocity jumps ' + scope):
        if fragment.is_an_individual and fragment.assigned_identity != 0:

            neighbour_fragment_past = fragment.get_neighbour_fragment(list_of_fragments.fragments, 'to_the_past')
            neighbour_fragment_future = fragment.get_neighbour_fragment(list_of_fragments.fragments, 'to_the_future')
            velocities_between_fragments = compute_velocities_consecutive_fragments(neighbour_fragment_past, fragment, neighbour_fragment_future)

            if all(velocity > impossible_velocity_threshold for velocity in velocities_between_fragments):
                # print("\nidentity: ", fragment.assigned_identity)
                if neighbour_fragment_past.identity_is_fixed or neighbour_fragment_future.identity_is_fixed:
                    reassign(fragment, list_of_fragments.fragments, impossible_velocity_threshold)
                else:
                    neighbour_fragment_past_past = neighbour_fragment_past.get_neighbour_fragment(list_of_fragments.fragments, 'to_the_past')
                    velocity_in_past = compute_velocities_consecutive_fragments(neighbour_fragment_past_past, neighbour_fragment_past, fragment)[0]
                    neighbour_fragment_future_future = neighbour_fragment_future.get_neighbour_fragment(list_of_fragments.fragments, 'to_the_future')
                    velocity_in_future = compute_velocities_consecutive_fragments(fragment, neighbour_fragment_future, neighbour_fragment_future_future)[1]
                    if velocity_in_past < impossible_velocity_threshold or velocity_in_future < impossible_velocity_threshold:
                        reassign(fragment, list_of_fragments.fragments, impossible_velocity_threshold)
                # print("corrected identity: ", fragment.assigned_identity)
            elif velocities_between_fragments[0] > impossible_velocity_threshold:
                # print("\nidentity: ", fragment.assigned_identity)
                if neighbour_fragment_past.identity_is_fixed:
                    reassign(fragment, list_of_fragments.fragments, impossible_velocity_threshold)
                else:
                    reassign(neighbour_fragment_past, list_of_fragments.fragments, impossible_velocity_threshold)
                # print("corrected identity: ", fragment.assigned_identity)
            elif velocities_between_fragments[1] > impossible_velocity_threshold:
                # print("\nidentity: ", fragment.assigned_identity)
                if neighbour_fragment_future.identity_is_fixed:
                    reassign(fragment, list_of_fragments.fragments, impossible_velocity_threshold)
                else:
                    reassign(neighbour_fragment_future, list_of_fragments.fragments, impossible_velocity_threshold)
                # print("corrected identity: ", fragment.assigned_identity)


def correct_impossible_velocity_jumps(video, list_of_fragments):
    correct_impossible_velocity_jumps_loop(video, list_of_fragments, scope = 'to_the_past')
    correct_impossible_velocity_jumps_loop(video, list_of_fragments, scope = 'to_the_future')
