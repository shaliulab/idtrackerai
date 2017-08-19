from __future__ import absolute_import, print_function, division
import sys
sys.path.append('../')
sys.path.append('../preprocessing')
sys.path.append('../utils')
sys.path.append('../network')
import numpy as np
from tqdm import tqdm
import collections
from blob import ListOfBlobs
from blob import Blob
from video_utils import segmentVideo, filterContoursBySize, getPixelsList, getBoundigBox
from globalfragment import order_global_fragments_by_distance_travelled

class Duplication(object):
    def __init__(self, blobs_in_frame_with_duplication = None, duplicated_identities = None, missing_identities = None):
        self.blobs_in_frame = blobs_in_frame_with_duplication
        self.identities_to_be_reassigned = duplicated_identities
        #all non duplicated identities in the frame are not available. This list will be
        #updated later on
        self.possible_identities = range(1, self.blobs_in_frame[0].number_of_animals+1)
        self.missing_identities = missing_identities
        # print("identities_to_be_reassigned", self.identities_to_be_reassigned)
        self.identities_assigned_during_accumulation = [blob.identity for blob in self.blobs_in_frame if blob.assigned_during_accumulation]
        # print("identities_assigned_during_accumulation ", self.identities_assigned_during_accumulation)
        self.identities_corrected_in_frame = [blob._identity_corrected_solving_duplication
                                                for blob in self.blobs_in_frame
                                                if blob._identity_corrected_solving_duplication is not None]
        # print("identities_corrected_in_frame ", self.identities_corrected_in_frame)
        self.non_available_identities = set(self.possible_identities) - set(self.missing_identities)
        ids_to_be_removed = set(self.identities_to_be_reassigned) - set(self.identities_assigned_during_accumulation)
        self.non_available_identities = self.non_available_identities - ids_to_be_removed
        self.non_available_identities = self.non_available_identities | set(self.identities_corrected_in_frame)
        self.non_available_identities = list(self.non_available_identities)
        # print("non_available_identities ", self.non_available_identities)

    def assign_unique_identities(self):
        all_blobs_to_reassign = []

        # self.available_identities = list(set(self.possible_identities) - set(self.non_available_identities))
        for identity in self.identities_to_be_reassigned:
            # print("\nsolving identity ", identity)
            self.available_identities = list(set(self.possible_identities) - set(self.non_available_identities))
            # print("available identities: ", self.available_identities)
            self.blobs_to_reassign = self.get_blobs_with_same_identity(identity)
            # print("number of blobs with same identity: ", len(self.blobs_to_reassign))
            self.assign()
            all_blobs_to_reassign.extend(self.blobs_to_reassign)

        return all_blobs_to_reassign

    def get_blobs_with_same_identity(self, identity):
        """We do not reassign blobs used as references
        """
        return [blob for blob in self.blobs_in_frame
                if blob.identity == identity and not blob._assigned_during_accumulation
                    and blob._identity_corrected_solving_duplication is None]

    @staticmethod
    def get_P2_matrix(blobs_list):
        return np.asarray([blob.P2_vector for blob in blobs_list])

    @staticmethod
    def sort_P2_matrix(P2_matrix):
        P2_ids = np.flip(np.argsort(P2_matrix, axis = 1), axis = 1) + 1
        corresponding_P2s = np.flip(np.sort(P2_matrix, axis = 1), axis = 1)
        return np.squeeze(np.asarray(P2_ids.T)), np.squeeze(np.asarray(corresponding_P2s.T))

    def set_to_0_non_available_ids(self,P2_matrix):
        no_available_now = np.asarray(self.non_available_identities)-1
        P2_matrix[:,no_available_now.astype(int)] = 0.
        return P2_matrix

    def assign(self):
        number_of_blobs_to_reassign = len(self.blobs_to_reassign)
        P2_matrix = self.get_P2_matrix(self.blobs_to_reassign)
        print("P2 matrix ", P2_matrix)
        P2_matrix = self.set_to_0_non_available_ids(P2_matrix) # Set the probabilities fo the non availabie identieies to 0
        print("P2 matrix ", P2_matrix)
        assigned_identities = []
        index_of_blobs_assigned = []

        counter = 0
        while len(assigned_identities) < number_of_blobs_to_reassign:
            # print("ids assigned: ", len(assigned_identities), ", blobs to reasign: ", number_of_blobs_to_reassign)
            if number_of_blobs_to_reassign == 1 and len(self.missing_identities) == 1:
                candidate_id = self.missing_identities[0]
                self.available_identities.remove(candidate_id)
                self.missing_identities.remove(candidate_id)
                assigned_identities.append(candidate_id)
                index_of_blobs_assigned.append(0)
                self.blobs_to_reassign[0]._identity_corrected_solving_duplication = candidate_id
            else:
                P2_max = np.max(P2_matrix,axis = 1) # Take the best value for of P2 for each blob
                max_indices = np.where(P2_max == np.max(P2_max))[0]
                if len(max_indices) == 1: # There is a blob that has a better P2max than the rest
                    # print("there is a unique maxima")
                    index_blob = np.argmax(P2_max)
                    P2_max_blob = np.max(P2_max)
                    candidate_id = np.argmax(P2_matrix[index_blob,:]) + 1
                    print("candidate_id: ", candidate_id)
                    print("P2_max_blob: ", P2_max_blob)
                    if candidate_id in self.available_identities and P2_max_blob > 1/np.sum(self.blobs_to_reassign[index_blob]._frequencies_in_fragment):
                        # print("id is available and P2 above random")
                        # print("we assign the candidate id", candidate_id)
                        # Assign the candidate_id if it is available and the probability is less than random
                        self.blobs_to_reassign[index_blob]._identity_corrected_solving_duplication = candidate_id
                        P2_matrix[:, candidate_id-1] = 0
                        P2_matrix[index_blob, :] = 0
                        self.available_identities.remove(candidate_id)
                        if candidate_id in self.missing_identities:
                            self.missing_identities.remove(candidate_id)
                        assigned_identities.append(candidate_id)
                        index_of_blobs_assigned.append(index_blob)
                        # print("P2 matrix ", P2_matrix)
                    elif candidate_id in self.available_identities and P2_max_blob < 1/np.sum(self.blobs_to_reassign[index_blob]._frequencies_in_fragment):
                        # print("id is available and P2 below random")
                        if len(self.available_identities) > 1 :
                            # print("there are other available")
                            # print("we assign 0")
                            # Assing the id to 0 because otherwise I would be assigning randomly
                            self.blobs_to_reassign[index_blob]._identity_corrected_solving_duplication = 0
                            assigned_identities.append(0)
                            index_of_blobs_assigned.append(index_blob)
                            P2_matrix[index_blob, :] = 0
                            # print("P2 matrix ", P2_matrix)
                        elif len(self.available_identities) == 1:
                            # print("is the last missing identity")
                            # print("missing identities ", self.missing_identities)
                            # print("available identities ", self.available_identities)
                            # print("candidate_id ", candidate_id)
                            self.blobs_to_reassign[index_blob]._identity_corrected_solving_duplication = candidate_id
                            assigned_identities.append(candidate_id)
                            index_of_blobs_assigned.append(index_blob)
                        else:
                            raise ValueError("condition no considered")
                    else:
                        raise ValueError("condition no considered")
                elif len(max_indices) > 1 and np.any(P2_max != 0):
                    # print("P2_max is degenerated")
                    # if there are duplicated maxima, set the id of those blobs to 0 and put P2_matrix of thos ids to 0
                    for max_index in max_indices:
                        candidate_id = np.argmax(P2_matrix[max_index,:])+1
                        # print("candidate_id: ", candidate_id-1)
                        P2_matrix[max_index, :] = 0
                        self.blobs_to_reassign[max_index]._identity_corrected_solving_duplication = 0
                        assigned_identities.append(0)
                        index_of_blobs_assigned.append(max_index)
                        # print("we assign to 0")
                        if candidate_id in self.available_identities:
                            P2_matrix[:, candidate_id-1] = 0
                            self.available_identities.remove(candidate_id)
                        # print("P2 matrix ", P2_matrix)
                elif len(max_indices) > 1 and np.all(P2_max == 0):
                    # print("all P2_max are 0 ")
                    index_of_blobs_to_be_assigned = list(set(range(number_of_blobs_to_reassign)).difference(set(index_of_blobs_assigned)))
                    # print("indices to be reassign ", index_of_blobs_to_be_assigned)
                    # print("available_identities ", self.available_identities)
                    # print("missing identities ", self.missing_identities)
                    if len(index_of_blobs_to_be_assigned) == 1 and len(self.available_identities) == 1 and len(self.missing_identities) == 1:
                        # if is the last blob to be assigned, assign the missing identity to that blob
                        self.blobs_to_reassign[index_of_blobs_to_be_assigned[0]]._identity_corrected_solving_duplication = self.missing_identities[0]
                        assigned_identities.append(self.missing_identities[0])
                        index_of_blobs_assigned.append(index_of_blobs_to_be_assigned[0])
                    elif len(index_of_blobs_to_be_assigned) == 1 and len(self.available_identities) == 1 and len(self.missing_identities) > 1:
                        # it is the last one of this duplication but there is another duplication and there are more than one missing identities
                        # we assing the identity to 0 because we do not know who it is
                        self.blobs_to_reassign[index_of_blobs_to_be_assigned[0]]._identity_corrected_solving_duplication = 0
                        assigned_identities.append(0)
                        index_of_blobs_assigned.append(index_of_blobs_to_be_assigned[0])
                    elif len(index_of_blobs_to_be_assigned) >= 1 and len(self.available_identities) > 1:
                        # there are more than one blobs to be assigned and al P2_max are 0. I assing them al to 0
                        for index_blob in index_of_blobs_to_be_assigned:
                             self.blobs_to_reassign[index_blob]._identity_corrected_solving_duplication = 0
                             assigned_identities.append(0)
                             index_of_blobs_assigned.append(index_blob)
                    else:
                        raise ValueError("condition no considered")
                else:
                    raise ValueError("condition not considered")
            counter += 1
            if counter > 10000:
                raise ValueError('Got trapped in the loop')

def get_first_frame(video, global_fragments):
    if not hasattr(video, 'first_frame_for_validation'):
        max_distance_travelled_global_fragment = order_global_fragments_by_distance_travelled(global_fragments)[0]
        video.first_frame_for_validation = max_distance_travelled_global_fragment.index_beginning_of_fragment

def get_identities_assigned_and_corrected_in_frame(blobs_in_frame):
    identities_in_frame = []
    for blob in blobs_in_frame:
        if blob._user_generated_identity is not None:
            identities_in_frame.append(blob._user_generated_identity)
        elif blob._identity_corrected_solving_duplication is not None:
            identities_in_frame.append(blob._identity_corrected_solving_duplication)
        else:
            identities_in_frame.append(blob.identity)
    return identities_in_frame

def assign_single_unidentified_blob(missing_identities, blobs_in_frame):
    if len(missing_identities) == 1:
        blob_with_0_id = [blob for blob in blobs_in_frame if blob._identity_corrected_solving_duplication == 0]
        if len(blob_with_0_id) == 1:
            # print("assigning unidentified blob with identity %i" %missing_identities[0])
            blob_with_0_id = blob_with_0_id[0]
            blob_with_0_id.update_identity_in_fragment(missing_identities[0], duplication_solved = True)
    return missing_identities

def assign_duplicated_identities_in_frame(blobs_in_frame, duplicated_identities, missing_identities):
    if len(duplicated_identities) > 0:

        frame  = Duplication(blobs_in_frame_with_duplication = blobs_in_frame,
                            duplicated_identities = duplicated_identities,
                            missing_identities = missing_identities)
        blobs_to_reassign = frame.assign_unique_identities()

        for blob in blobs_in_frame:
            for duplicated_blob in blobs_to_reassign:
                if blob is duplicated_blob and blob.identity != duplicated_blob._identity_corrected_solving_duplication:
                    blob.update_identity_in_fragment(duplicated_blob._identity_corrected_solving_duplication, duplication_solved = True)

def check_for_duplications(blobs_in_frame, possible_identities):
    identities_in_frame = get_identities_assigned_and_corrected_in_frame(blobs_in_frame)
    duplicated_identities = set([x for x in identities_in_frame if identities_in_frame.count(x) > 1 and x != 0])
    missing_identities = list(possible_identities.difference(identities_in_frame))
    if 0 in missing_identities:
        missing_identities.remove(0)
    print("identities in frame: ", identities_in_frame)
    print("duplicated identities: ", duplicated_identities)
    print("missing identities: ", missing_identities)
    print([(blob.identity, blob._identity_corrected_solving_duplication, blob.assigned_during_accumulation) for blob in blobs_in_frame])
    return duplicated_identities, identities_in_frame, missing_identities

def solve_duplications_loop(video, blobs, group_size, scope = None):
    if scope == 'to_the_past':
        blobs = blobs[:video.first_frame_for_validation][::-1]
    elif scope == 'to_the_future':
        blobs = blobs[video.first_frame_for_validation:-1]
    possible_identities = set(range(1,group_size+1))

    for blobs_in_frame in tqdm(blobs, desc = 'Solving duplications'):
        print("\n*******solving frame with duplications...")
        print("frame ", blobs_in_frame[0].frame_number)
        print([(blob.identity, blob._identity_corrected_solving_duplication, blob.assigned_during_accumulation) for blob in blobs_in_frame])
        duplicated_identities, identities_in_frame, missing_identities = check_for_duplications(blobs_in_frame, possible_identities)
        missing_identities = assign_single_unidentified_blob(missing_identities, blobs_in_frame)
        assign_duplicated_identities_in_frame(blobs_in_frame, duplicated_identities, missing_identities)

    for blobs_in_frame in tqdm(blobs, desc = 'Checking that there are no more duplication'):
        # print("\n*******checking that there are no more duplications...")
        # print("frame ", blobs_in_frame[0].frame_number)
        # duplicated_identities, identities_in_frame, missing_identities = check_for_duplications(blobs_in_frame, possible_identities)
        # missing_identities = assign_single_unidentified_blob(missing_identities, blobs_in_frame)
        duplicated_identities, identities_in_frame, _ = check_for_duplications(blobs_in_frame, possible_identities)
        if len(duplicated_identities) > 0:
            print("identities_in_frame, ",  identities_in_frame)
            raise ValueError("Not all the duplications have been removed")

def solve_duplications(video, blobs, global_fragments, group_size):

    get_first_frame(video, global_fragments)
    solve_duplications_loop(video, blobs, group_size, scope = 'to_the_past')
    solve_duplications_loop(video, blobs, group_size, scope = 'to_the_future')


# [(blob.identity, blob._identity_corrected_solving_duplication, blob.assigned_during_accumulation) for blob in blobs[30626]]
