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

from blob import ListOfBlobs
from assigner import assign
from id_CNN import ConvNetwork
from network_params import NetworkParams
from blob import Blob
from compute_velocity_model import compute_velocity_from_list_of_blobs, compute_model_velocity
from get_trajectories import produce_trajectories
from plot_individual_velocity import plot_individual_trajectories_velocities_and_accelerations

VEL_PERCENTILE = 99 #percentile used to model velocity jumps
P2_CERTAINTY_THRESHOLD = .9
VELOCITY_TOLERANCE = 1.5

class ImpossibleJump(object):
    def __init__(self, video_object = None,
                        blobs_in_video = None,
                        blob_extreme_past = None,
                        blobs_border_past = None,
                        speeds_at_border_past = None,
                        blob_extreme_future = None,
                        blobs_border_future = None,
                        speeds_at_border_future = None):
        self.video = video_object
        self.blobs_in_video = blobs_in_video
        self.blob_extreme_past = blob_extreme_past
        self.blobs_border_past = blobs_border_past
        self.speeds_at_border_past = speeds_at_border_past
        self.blob_extreme_future = blob_extreme_future
        self.blobs_border_future = blobs_border_future
        self.speeds_at_border_future = speeds_at_border_future

    @property
    def jump_in_future(self):
        return np.any(self.speeds_at_border_future > self.video.velocity_threshold)

    @property
    def jump_in_past(self):
        return np.any(self.speeds_at_border_past > self.video.velocity_threshold)

    @property
    def jump_in_past_and_future(self):
        return self.jump_in_past and self.jump_in_future

    @staticmethod
    def get_assigned_and_corrected_identity(blob):
        if blob._user_generated_identity is not None:
            return blob._user_generated_identity
        elif blob._identity_corrected_solving_duplication is not None:
            return blob._identity_corrected_solving_duplication
        elif blob.identity is not None:
            return blob.identity
        else:
            return None

    def get_identities_assigned_and_corrected_in_frame(self, blobs_in_frame):
        identities_in_frame = []
        for blob in blobs_in_frame:
            blob_identity = self.get_assigned_and_corrected_identity(blob)
            if blob_identity is not None:
                identities_in_frame.append(blob_identity)
        return identities_in_frame

    def get_available_and_non_available_identities(self, blob):
        """This function does not allow a segmentation with maximal number of
        blobs bigger than the number of animals. If that happens the available
        identities set could be empty.
        """
        non_available_identities = set(self.get_identities_assigned_and_corrected_in_frame(self.blobs_in_video[blob.frame_number]))
        if 0 in non_available_identities:
            non_available_identities.remove(0)
        coexisting_identities, _ = blob.get_fixed_identities_of_coexisting_fragments(self.blobs_in_video)
        coexisting_identities = set(coexisting_identities)
        if 0 in coexisting_identities:
            coexisting_identities.remove(0)
        print("from get_available_identities. non_available_identities ", non_available_identities)
        print("from get_available_identities. coexisting_identities ", coexisting_identities)
        non_available_identities = coexisting_identities | non_available_identities
        print("from get_available_identities. non_available_identities U coexisting_identities ", non_available_identities)
        available_identities = set(range(1, self.video.number_of_animals + 1)) - non_available_identities
        print("from get_available_identities. available_identities ", available_identities)
        available_identities = available_identities | set([blob.identity])
        non_available_identities.remove(blob.identity)
        return non_available_identities, available_identities

    # def get_blob_to_reassing(self):
    #     candidate_blobs = [self.blob_extreme_past] + [blob for blob in self.blobs_border_past if not blob.is_fixed ] + [blob for blob in self.blobs_border_future if not blob.is_fixed ]
    #     for candidate_blob in candidate_blobs:
    #         self.check_velocity_border_fragments(candidate_blob)

    def check_velocity_border_fragments_past(self, blob):
        print("--Previous")
        return check_velocity_neighbour_fragment(self.video, self.blobs_in_video, blob, direction = 'previous')

    def check_velocity_border_fragments_future(self, blob):
        print("--Next")
        return check_velocity_neighbour_fragment(self.video, self.blobs_in_video, blob, direction = 'next')

    def check_velocity_border_fragments(self, blob):
        blob_extreme_past, blobs_border_past, speeds_at_border_past = self.check_velocity_border_fragments_previous(blob)
        blob_extreme_future, blobs_border_future, speeds_at_border_future = self.check_velocity_border_fragments_next(blob)

    def get_blob_to_reassign_past(self):
        #we consider the blobs in the past without a fixed identity
        candidate_blobs = np.asarray([blob for blob in self.blobs_border_past if not blob.is_fixed])
        #for those blobs we check the velocity in borders
        for candidate_blob in candidate_blobs:
            blob_extreme_past, blobs_border_past, speeds_at_border_past = self.check_velocity_border_fragments_past(candidate_blob)
            print("check border in the past of candidate blob: ", speeds_at_border_past > self.video.velocity_threshold)
            if speeds_at_border_past > self.video.velocity_threshold:
                return candidate_blob
            else:
                return self.blob_extreme_past

    def get_blob_to_reassign_future(self):
        #we consider the blobs in the past without a fixed identity
        candidate_blobs = np.asarray([blob for blob in self.blobs_border_future if not blob.is_fixed])
        #for those blobs we check the velocity in borders
        for candidate_blob in candidate_blobs:
            blob_extreme_future, blobs_border_future, speeds_at_border_future = self.check_velocity_border_fragments_future(candidate_blob)
            print("check border in the future of candidate blob: ", speeds_at_border_future > self.video.velocity_threshold)
            if speeds_at_border_future > self.video.velocity_threshold:
                return candidate_blob
            else:
                return self.blob_extreme_future

    def get_candidate_identities_by_above_random_P2(self, blob, non_available_identities):
        P2_vector = blob._P2_vector
        # print("P2 vector ", P2_vector)
        P2_vector[non_available_identities - 1] = 0
        # print("P2 vector after removing zeros ", P2_vector)
        if np.all(P2_vector == 0):
            candidate_identities_speed = self.get_candidate_identities_by_minimum_speed(blob)
            return candidate_identities_speed
        else:
            if np.sum(blob._frequencies_in_fragment) == 1:
                random_threshold  = 1/self.video.number_of_animals
            else:
                random_threshold = 1/np.sum(blob._frequencies_in_fragment)

            return np.where(P2_vector > random_threshold)[0] + 1

    def get_candidate_identities_by_minimum_speed(self, blob, available_identities):

        original_identity = blob.identity
        speed_of_candidate_identities = []
        for identity in available_identities:
            speeds_of_identity = []
            blob._identity = identity
            _, _, speeds_at_border_past = check_velocity_neighbour_fragment(self.video, self.blobs_in_video, blob, direction = 'previous')
            _, _, speeds_at_border_future = check_velocity_neighbour_fragment(self.video, self.blobs_in_video, blob, direction = 'next')
            speeds_of_identity.extend(speeds_at_border_past)
            speeds_of_identity.extend(speeds_at_border_future)
            if len(speeds_of_identity) != 0:
                speed_of_candidate_identities.append(np.min(speeds_of_identity))
            else:
                speed_of_candidate_identities.append(VELOCITY_TOLERANCE * self.video.velocity_threshold)
        blob._identity = original_identity
        argsort_identities_by_speed = np.argsort(speed_of_candidate_identities)
        print("available_identities", available_identities)
        print("speed_of_candidate_identities, ", speed_of_candidate_identities)
        return np.asarray(list(available_identities))[argsort_identities_by_speed], np.asarray(speed_of_candidate_identities)[argsort_identities_by_speed]

    def reassign(self, blob):
        non_available_identities, available_identities = self.get_available_and_non_available_identities(blob)
        non_available_identities = np.array(list(non_available_identities))
        print("available identities ",available_identities)
        print("non available identities ",non_available_identities)
        if len(available_identities) == 1:
            print("There is a single id available!")
            candidate_id = list(available_identities)[0]
            print("id = ", candidate_id)
        elif len(available_identities) == 0:
            print("There are no ids available!")
            print("blob is a jump ", blob.is_a_jump)
            print("blob is a ghost crossing ", blob.is_a_ghost_crossing)
            candidate_id = blob.identity
            print("id = ", candidate_id)
        else:
            candidate_identities_speed, speed_of_candidate_identities = self.get_candidate_identities_by_minimum_speed(blob, available_identities)
            print("candidate_identities_speed, ", candidate_identities_speed)
            print("speed_of_candidate_identities, ", speed_of_candidate_identities)
            candidate_identities_P2 = self.get_candidate_identities_by_above_random_P2(blob, non_available_identities)
            print("candidate_identities_P2, ", candidate_identities_P2 )
            candidate_identities = []
            candidate_speeds = []
            for candidate_id, candidate_speed in zip(candidate_identities_speed, speed_of_candidate_identities):
                if candidate_id in candidate_identities_P2:
                    candidate_identities.append(candidate_id)
                    candidate_speeds.append(candidate_speed)
            print("candidate_identities, ", candidate_identities)
            print("candidate_speeds, ", candidate_speeds)
            if len(candidate_identities) == 0:
                print("There are not candidate identities that are available by P2_vector and speed")
                candidate_id = 0
            elif len(candidate_identities) == 1:
                print("There is a single identity that is available by P2_vector and speed and it passe")
                if candidate_speeds[0] < VELOCITY_TOLERANCE * self.video.velocity_threshold:
                    print("It passes the velocity tolerance")
                    candidate_id = candidate_identities[0]
                else:
                    print("It does not pass the velocity tolerance")
                    candidate_id = 0
            elif len(candidate_identities) > 1:
                print("There are several identities that are available by P2_vector and speed")
                if len(np.where(candidate_speeds == np.min(candidate_speeds))[0]) == 1:
                    print("The minimum speed is unique")
                    if candidate_speeds[0] < VELOCITY_TOLERANCE * self.video.velocity_threshold:
                        print("It passes the velocity tolerance")
                        candidate_id = candidate_identities[0]
                    else:
                        print("It does not pass the velocity tolerance")
                        candidate_id = 0
                else:
                    print("The minimum speed is degenerated")
                    candidate_id = 0
            print("candidate_id, ", candidate_id)

        print("identity update: ", blob.identity, " --> ", candidate_id)
        blob._identity = candidate_id
        number_of_images_in_fragment = len(blob.identities_in_fragment())
        print("number_of_images_in_fragment, ", number_of_images_in_fragment)
        blob.update_identity_in_fragment(candidate_id, number_of_images_in_fragment = number_of_images_in_fragment)
        return blob

    def solve(self):
        if self.jump_in_past_and_future:
            print("blobs_border_past, ", self.blobs_border_past)
            print("blobs_boder_future, ", self.blobs_border_future)
            print("blobs fixed in past, ", [blob.is_fixed for blob in self.blobs_border_past])
            print("blobs fixed in future, ", [blob.is_fixed for blob in self.blobs_border_future])
            if np.all([blob.is_fixed for blob in self.blobs_border_past]) or np.all([blob.is_fixed for blob in self.blobs_border_future]):
                #the identity of the fragment containing the blob we are considering
                #is surely wrong. Hence, we reassign it
                self.reassign(self.blob_extreme_past) #we reassign the fragment (blob_extreme_future would also do)
            else:
                print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
                print("not all are fixed past or future")
                blob_to_reassign_past = self.get_blob_to_reassign_past()
                blob_to_reassign_future = self.get_blob_to_reassign_future()

                if blob_to_reassign_past is self.blob_extreme_past:
                    print("The blob to reassign is the current one [past]")
                    self.reassign(self.blob_extreme_past)
                elif blob_to_reassign_future is self.blob_extreme_future:
                    print("The blob to reassign is the current one [future]")
                    self.reassign(self.blob_extreme_future)
                else:
                    print("This blob does not need to be fixed. A consecutive blob will be fixed later if needed")



                # blob_to_reassign = self.get_blob_to_reassing()
        elif self.jump_in_past:
            if np.all([blob.is_fixed for blob in self.blobs_border_past]):
                self.reassign(self.blob_extreme_past)
            else:
                print("--------------------------------------------------------")
                blob_to_reassign = self.get_blob_to_reassign_past()
                self.reassign(blob_to_reassign)
                print("not all are fixed past")
        elif self.jump_in_future:
            if np.all([blob.is_fixed for blob in self.blobs_border_future]):
                self.reassign(self.blob_extreme_future)
            else:
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                blob_to_reassign = self.get_blob_to_reassign_future()
                self.reassign(blob_to_reassign)
                print("not all are fixed future")




def give_me_extreme_blob_of_current_fragment(blob, direction):
    cur_fragment_identifier = blob.fragment_identifier
    blob_to_return = blob

    print("(out)Getting to the extreme of the fragment. Direction:. Frame number. Id. FragID ",(direction, blob_to_return.frame_number, blob_to_return.identity, blob_to_return.fragment_identifier))
    while len(getattr(blob_to_return, direction)) == 1 and getattr(blob_to_return, direction)[0].is_a_fish and getattr(blob_to_return, direction)[0].fragment_identifier == cur_fragment_identifier:
        blob_to_return = getattr(blob_to_return, direction)[0]
        print("(in)Getting to the extreme of the fragment. Direction:. Frame number. Id. FragID ",(direction, blob_to_return.frame_number, blob_to_return.identity, blob_to_return.fragment_identifier))

    return blob_to_return

def check_velocity_neighbour_fragment(video, blobs_in_video, blob, direction = None):
    blob_extreme = give_me_extreme_blob_of_current_fragment(blob, direction)
    if direction == 'next':
        frame_number = blob_extreme.frame_number + 1
    elif direction == 'previous':
        frame_number = blob_extreme.frame_number - 1
    else:
        raise ValueError("Check you direction smart goose!")

    print("border frame:", frame_number)
    blobs_border_frame_same_identity = [blob for blob in blobs_in_video[frame_number] if blob.identity == blob_extreme.identity]
    speeds_at_border = []
    print("blobs_border_frame_same_identity in ",direction, " :", blobs_border_frame_same_identity)
    if len(blobs_border_frame_same_identity) > 0:
        for boder_blob in blobs_border_frame_same_identity:
            individual_velocities = np.diff([blob_extreme.centroid, boder_blob.centroid], axis = 0)
            print("individual_velocities: ", individual_velocities)
            speed_at_border = np.linalg.norm(individual_velocities)
            print("individual speed in %s direction: %.4f. The velocity threshold is %.4f" %(direction, speed_at_border, video.velocity_threshold))
            speeds_at_border.append(speed_at_border)
        return blob_extreme, blobs_border_frame_same_identity, speeds_at_border
    else:
        logging.warn("There is no available %s fragment" %direction)
        return blob_extreme, [], []


def solve_impossible_jumps_for_blobs_in_frame(video, blobs_in_video, blobs_in_frame, individual_fragments_checked, direction):

    for blob in blobs_in_frame:
        blob_identity = ImpossibleJump.get_assigned_and_corrected_identity(blob)
        if blob_identity != 0 and blob.is_a_fish and blob.fragment_identifier not in individual_fragments_checked and not blob.assigned_during_accumulation:
            print("\nfragment_identifier: ", blob.fragment_identifier)
            print("--Previous")
            blob_extreme_past, blobs_border_past, speeds_at_border_past = check_velocity_neighbour_fragment(video, blobs_in_video, blob, direction = 'previous')
            print("--Next")
            blob_extreme_future, blobs_border_future, speeds_at_border_future = check_velocity_neighbour_fragment(video, blobs_in_video, blob, direction = 'next')
            if (np.any(speeds_at_border_past > video.velocity_threshold) or np.any(speeds_at_border_future > video.velocity_threshold)) and\
                (len(blobs_border_past) != 0 or len(blobs_border_future) != 0):
                impossible_jump = ImpossibleJump(video_object = video,
                                                    blobs_in_video = blobs_in_video,
                                                    blob_extreme_past = blob_extreme_past,
                                                    blobs_border_past = blobs_border_past,
                                                    speeds_at_border_past = speeds_at_border_past,
                                                    blob_extreme_future = blob_extreme_future,
                                                    blobs_border_future = blobs_border_future,
                                                    speeds_at_border_future = speeds_at_border_future)
                impossible_jump.solve()
            else:
                print("It is not an impossible velocity jump")
            individual_fragments_checked.append(blob.fragment_identifier)
    return individual_fragments_checked

def correct_impossible_velocity_jumps_loop(video, blobs_in_video, direction = None):
    if direction == 'previous':
        blobs_in_direction = blobs_in_video[:video.first_frame_for_validation][::-1]
    elif direction == 'next':
        blobs_in_direction = blobs_in_video[video.first_frame_for_validation:-1]
    possible_identities = set(range(1,video.number_of_animals+1))

    individual_fragments_checked = []
    for blobs_in_frame in tqdm(blobs_in_direction, desc = 'Correcting impossible velocity jumps ' + direction):
        print('\n *** frame, ', blobs_in_frame[0].frame_number)
        individual_fragments_checked = solve_impossible_jumps_for_blobs_in_frame(video, blobs_in_video, blobs_in_frame, individual_fragments_checked, direction)

def correct_impossible_velocity_jumps(video, blobs):
    correct_impossible_velocity_jumps_loop(video, blobs, direction = 'previous')
    correct_impossible_velocity_jumps_loop(video, blobs, direction = 'next')

def fix_identity_of_blobs_list(list_of_blobs, method = 'accumulation'):
    if method == 'accumulation':
        [blob.update_attributes_in_fragment(['is_fixed'],[True]) if blob.assigned_during_accumulation
            else blob.update_attributes_in_fragment(['is_fixed'],[False])
            for blob in list_of_blobs if not hasattr(blob,'is_fixed')]
    elif method == 'P2_vector':
        [blob.update_attributes_in_fragment(['is_fixed'],[True]) if np.max(blob._P2_vector) > P2_CERTAINTY_THRESHOLD
            else blob.update_attributes_in_fragment(['is_fixed'],[False])
            for blob in list_of_blobs if not hasattr(blob,'is_fixed')]

def fix_identity_of_blobs_in_video(blobs_in_video):
    for blobs_in_frame in tqdm(blobs_in_video, desc = 'Fixing identity of certain blobs'):
        fix_identity_of_blobs_list(blobs_in_frame, method = 'accumulation')


if __name__ == "__main__":
    from GUI_utils import frame_by_frame_identity_inspector
    NUM_CHUNKS_BLOB_SAVING = 10

    #load video and list of blobs
    # video = np.load('/home/chronos/Desktop/IdTrackerDeep/videos/8zebrafish_conflicto/session_4/video_object.npy').item()
    video = np.load('/home/themis/Desktop/IdTrackerDeep/videos/idTrackerDeep_Sex/mixedGroup_28indiv_first/session_rr05_no_ghost/video_object.npy').item()
    number_of_animals = video.number_of_animals
    # list_of_blobs_path = '/home/chronos/Desktop/IdTrackerDeep/videos/8zebrafish_conflicto/session_4/preprocessing/blobs_collection.npy'
    list_of_blobs_path = '/home/themis/Desktop/IdTrackerDeep/videos/idTrackerDeep_Sex/mixedGroup_28indiv_first/session_rr05_no_ghost/preprocessing/blobs_collection.npy'
    list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
    blobs = list_of_blobs.blobs_in_video
    # individual_speeds = get_speed(blobs)
    # jumps_identities, jumps_frame_numbers, velocities = get_frames_with_impossible_speed(video, blobs, individual_speeds)
    fix_identity_of_blobs_in_video(blobs)
    correct_impossible_velocity_jumps(video, blobs)

    trajectories = produce_trajectories(blobs, len(blobs), video.number_of_animals)

    plot_individual_trajectories_velocities_and_accelerations(trajectories['centroid'])

    # for identity, frame_number in zip(jumps_identities, jumps_frame_numbers):
    #     impossible_jump = ImpossibleJump(blobs, identity, frame_number, number_of_animals, velocity_threshold = video.velocity_threshold)
    #     impossible_jump.correct_impossible_jumps()
