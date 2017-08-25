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

from blob import ListOfBlobs
from assigner import assign
from id_CNN import ConvNetwork
from network_params import NetworkParams
from blob import Blob
from compute_velocity_model import compute_velocity_from_list_of_blobs, compute_model_velocity

VEL_PERCENTILE = 99 #percentile used to model velocity jumps


def get_speed(blobs_in_video):
    number_of_animals = blobs_in_video[0][0].number_of_animals
    number_of_frames = len(blobs_in_video)
    centroid_trajectories = np.ones((number_of_animals,number_of_frames, 2))*np.NaN

    for frame_number, blobs_in_frame in enumerate(tqdm(blobs_in_video, desc = "Computing trajectories")):
        for blob in blobs_in_frame:
            if blob.user_generated_identity is not None:
                blob_identity = blob.user_generated_identity
            elif blob._identity_corrected_solving_duplication is not None:
                blob_identity = blob._identity_corrected_solving_duplication
            else:
                blob_identity = blob.identity

            if (blob_identity is not None) and (blob_identity != 0):
                centroid_trajectories[blob_identity-1, frame_number, :] = blob.centroid

    individual_velocities = np.diff(centroid_trajectories, axis = 1)
    individual_speeds = np.linalg.norm(individual_velocities, axis = 2)
    individual_speeds[np.isnan(individual_speeds)] = -1

    return individual_speeds

def get_frames_with_impossible_speed(video, blobs_in_video, individual_speeds):
    if not hasattr(video, "velocity_threshold"):
        video.velocity_threshold = compute_model_velocity(blobs_in_video, video.number_of_animals, percentile = VEL_PERCENTILE)
    jumps_identities, jumps_frame_numbers = np.where(individual_speeds > video.velocity_threshold)
    velocities = individual_speeds[jumps_identities, jumps_frame_numbers]
    return jumps_identities + 1, jumps_frame_numbers, velocities

class ImpossibleJump(object):
    def __init__(self, blobs_in_video, impossible_jump_identity, impossible_jump_frame, number_of_animals, velocity_threshold = None):
        self.blobs_in_video = blobs_in_video
        self.blob = [blob for blob in self.blobs_in_video[impossible_jump_frame] if blob.identity == impossible_jump_identity]
        self.number_of_animals = number_of_animals
        self.velocity_threshold = velocity_threshold

    def compare_fragment_identifiers(self):
        if len(self.blob) == 0: return False
        assert len(self.blob) == 1
        self.blob = self.blob[0]
        if self.blob.is_a_fish_in_a_fragment:
            if not self.blob.previous[0].is_a_crossing and not self.blob.next[0].is_a_crossing:
                return False
        next_fragment_identifiers = [blob_next_frame.fragment_identifier
                                    for blob_next_frame in self.blobs_in_video[self.blob.frame_number + 1]
                                    if blob_next_frame.identity == self.blob.identity]
        if len(next_fragment_identifiers) == 1:
            next_fragment_identifier = next_fragment_identifiers[0]
            self.blob_next_frame = blob_next_frame
            return self.blob.fragment_identifier != next_fragment_identifier
        else:
            print("weird next_fragment_identifiers length: ", len(next_fragment_identifiers))
            return False

    def check_that_is_an_impossible_jump(self):
        return self.compare_fragment_identifiers() and not self.blob.is_a_ghost_crossing

    @staticmethod
    def get_P2_vectors(blob1, blob2):
        return blob1._P2_vector, blob2._P2_vector

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
        available_identities = set(range(1, self.number_of_animals + 1)) - non_available_identities
        print("from get_available_identities. available_identities ", available_identities)
        return non_available_identities, available_identities

    def give_me_extreme_blob_of_current_fragment(self, blob, direction):
        cur_fragment_identifier = blob.fragment_identifier
        blob_to_return = blob

        logging.info("(out)Getting to the extreme of the fragment. Direction: %s. Frame number %i. Id %i. FID %i" %(direction, blob_to_return.frame_number, blob_to_return.identity, blob_to_return.fragment_identifier))
        print(len(blob.next[0].previous) == 1)
        while len(getattr(blob_to_return, direction)) == 1 and getattr(blob_to_return, direction)[0].fragment_identifier == cur_fragment_identifier:
            blob_to_return = blob_to_return.next[0]
            print("(in)Getting to the extreme of the fragment. Direction:. Frame number. Id. FID ",(direction, blob_to_return.frame_number, blob_to_return.identity, blob_to_return.fragment_identifier))

        return blob_to_return

    def check_velocity_neighbour_fragment(self, blob, direction = None):
        last_blob_in_fragment = self.give_me_extreme_blob_of_current_fragment(blob, direction)
        if direction == 'next':
            frame_number = last_blob_in_fragment.frame_number + 1
        elif direction == 'previous':
            frame_number = last_blob_in_fragment.frame_number - 1
        else:
            raise ValueError("Check you direction smart goose!")

        blob_next_frame_same_identity = [blob for blob in self.blobs_in_video[frame_number] if blob.identity == last_blob_in_fragment.identity]
        if len(blob_next_frame_same_identity) > 0:
            individual_velocities = np.diff([last_blob_in_fragment.centroid, blob_next_frame_same_identity[0].centroid], axis = 1)
            individual_speed = np.linalg.norm(individual_velocities)
            logging.info("individual speed in %s direction: %.4f. The velocity threshold is %.4f" %(direction, individual_speed, self.velocity_threshold))
            return individual_speed < self.velocity_threshold
        else:
            logging.warn("There is no available %s fragment" %direction)
            return False

    def reassign_blob(self, blob):
        print("__reassigning blob__")
        non_available_identities, available_identities = self.get_available_and_non_available_identities(blob)
        non_available_identities = np.array(list(non_available_identities))
        print("available identities ",available_identities)
        print("non available identities ",non_available_identities)
        if len(available_identities) == 1:
            print("There is a single id available!")
            candidate_id = list(available_identities)[0]
            print("id = ", candidate_id)
        else:
            print("Choosing id according to P2 maximum")
            P2_vector = blob._P2_vector
            print("P2 vector ", P2_vector)
            P2_vector[non_available_identities - 1] = 0
            print("P2 vector after removing zeros ", P2_vector)
            # XXX check if P2s are all zero!!!!!!!!!
            candidate_id = np.argmax(P2_vector) + 1
            print("id = ", candidate_id)

        print("checking next fragment: ", self.check_velocity_neighbour_fragment(blob, direction = 'next'))
        print("checking previous fragment: ", self.check_velocity_neighbour_fragment(blob, direction = 'previous'))
        print("candidate_id ", candidate_id)
        print("candidate available? ", candidate_id in available_identities)
        if candidate_id in available_identities and \
            self.check_velocity_neighbour_fragment(blob, direction = 'next') and \
            self.check_velocity_neighbour_fragment(blob, direction = 'previous'):
            print("frame number ", blob.frame_number)
            print("blob identity ", blob.identity)
            blob.old_identity = blob.identity
            blob._identity = candidate_id
            print("blob new identity ", blob.identity)
            blob.update_identity_in_fragment(candidate_id)
            return blob
        return None


    @staticmethod
    def give_me_the_blob_to_correct(list_of_blobs):
        P2_maxima = [np.max(blob._P2_vector) for blob in list_of_blobs]
        minimal_P2s_indices = np.where(P2_maxima == np.min(P2_maxima))[0]
        if len(minimal_P2s_indices) == 1:
            return list_of_blobs[minimal_P2s_indices[0]]
        else:
            raise NotImplementedError("the P2_vecs has same min max")


    def correct_impossible_jumps(self):
        if self.check_that_is_an_impossible_jump():
            #evaluate P2 of the two frames
            P2_vector_blob, P2_vector_blob_after_jump = self.get_P2_vectors(self.blob, self.blob_next_frame)
            # print("frame number starting jump ", self.blob.frame_number)
            # print("blob P2s ", P2_vector_blob)
            # print("blob next frame P2 ", P2_vector_blob_after_jump)
            # print("________________________________")
            if self.blob.assigned_during_accumulation and self.blob_next_frame.assigned_during_accumulation:
                logging.warn("A velocity jump occured in subsequent frames during accumulation")
            else:
                blob_to_correct = self.give_me_the_blob_to_correct([self.blob, self.blob_next_frame])
                self.reassign_blob(blob_to_correct)



# old check method
# def check(self):
    # if self.compare_fragment_identifiers() and not self.blob.is_a_ghost_crossing:
    #     print("It is an impossible jump")
    #     print("frame number ", self.blob.frame_number)
    #     print("identity ", self.blob.identity)
    #     print("fish_in_a_fragment ", self.blob.is_a_fish_in_a_fragment)
    #     print("jump ", self.blob.is_a_jump)
    #     print("ghost crossing ", self.blob.is_a_ghost_crossing)
    #     print("_______________________________________")
    #     return True
    # return False



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
    individual_speeds = get_speed(blobs)
    jumps_identities, jumps_frame_numbers, velocities = get_frames_with_impossible_speed(video, blobs, individual_speeds)

    for identity, frame_number in zip(jumps_identities, jumps_frame_numbers):
        impossible_jump = ImpossibleJump(blobs, identity, frame_number, number_of_animals, velocity_threshold = video.velocity_threshold)
        impossible_jump.correct_impossible_jumps()
